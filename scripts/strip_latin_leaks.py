"""
strip_latin_leaks.py — Strip Latin-script leaks from Cyrillic assistant turns.

Processes JSONL training files to remove:
  - "Daisy noticed[,]? " meta-prefixes
  - English parentheticals like "(trauma bonding)", "(compassion fatigue)"
  - Standalone English sentences (>5 Latin words in a row)
  - English meta-instructions that leaked into training data

Preserves legitimate mixed terms (CBT, DBT when used as known acronyms).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daisy.strip_latin_leaks")

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# "Daisy noticed[,]? " at the start or mid-text
_DAISY_NOTICED_RE = re.compile(
    r"Daisy noticed[,]?\s+",
    re.IGNORECASE,
)

# English parentheticals — common therapeutic terms that leaked
_ENGLISH_PARENTHETICALS_RE = re.compile(
    r"\((?:trauma bonding|compassion fatigue|core belief|"
    r"emotional regulation|distress tolerance|interpersonal|"
    r"cognitive restructuring|behavioral activation|coping skill|"
    r"grounding technique|mindfulness exercise|somatic experiencing|"
    r"attachment style|nervous system|window of tolerance|"
    r"inner critic|self compassion|emotional flashback|"
    r"values clarification|thought challenging|exposure hierarchy)"
    r"(?:\s+[^)]*)?\)",
    re.IGNORECASE,
)

# Standalone English sentences: >5 Latin words in a row (in Cyrillic context)
_STANDALONE_ENGLISH_RE = re.compile(
    r"[.!?]?\s*([A-Z][a-z]+(?:\s+[a-zA-Z]+){5,})\s*[.!?]",
)

# Common English meta-phrases that leaked into RU assistant turns
_ENGLISH_META_PHRASES = [
    re.compile(
        r"\b(?:It sounds like|It seems like|It looks like|I hear that|"
        r"I understand that|I can see that|It feels like)\b"
        r"[^.]*[.]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:What I'm hearing is|What I hear is|What I notice is)\b"
        r"[^.]*[.]",
        re.IGNORECASE,
    ),
]

# Role headers that leaked
_ROLE_HEADER_RE = re.compile(
    r"^(Assistant:|Question:|User:|Human:)\s*",
    re.IGNORECASE,
)

# Rubric tokens
_RUBRIC_RE = re.compile(
    r"\*\*Rubric\*\*|Score:\s*\d+|# (Excellent|Good|Fair|Poor)",
    re.IGNORECASE,
)

# Acronyms that should be PRESERVED
_ALLOWED_ACRONYMS = {"DBT", "CBT", "ACT", "EMDR", "PTSD", "OCD", "ADHD"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _has_cyrillic(text: str) -> bool:
    """Check if text contains any Cyrillic characters."""
    return bool(re.search(r"[\u0400-\u04FF]", text))


def strip_leaks(text: str) -> str:
    """Remove Latin-script leaks from a Cyrillic assistant turn.

    Args:
        text: Raw assistant text, potentially containing Latin leaks.

    Returns:
        Cleaned text with leaks removed.  Original whitespace and
        structure are preserved where possible.
    """
    if not text or not isinstance(text, str):
        return text

    original = text.strip()
    cleaned = original

    # If text has NO Cyrillic, it is pure English -- leave it alone
    if not _has_cyrillic(cleaned):
        return original

    # 1. Strip "Daisy noticed[,]? " -- preserve following text
    cleaned = _DAISY_NOTICED_RE.sub("", cleaned)

    # 2. Strip English parentheticals like (trauma bonding)
    cleaned = _ENGLISH_PARENTHETICALS_RE.sub("", cleaned)

    # 3. Strip English meta-phrases (standalone sentences in English)
    for meta_re in _ENGLISH_META_PHRASES:
        cleaned = meta_re.sub("", cleaned)

    # 4. Strip role headers
    cleaned = _ROLE_HEADER_RE.sub("", cleaned)

    # 5. Strip rubric tokens
    cleaned = _RUBRIC_RE.sub("", cleaned)

    # 6. Remove standalone English sentences (>5 Latin words)
    #    ONLY in Cyrillic context -- skip if no Cyrillic nearby
    def _replace_standalone(match: re.Match) -> str:
        sentence = match.group(1)
        words = sentence.split()
        allowed_count = sum(
            1 for w in words if w.strip("().,;:?!").upper() in _ALLOWED_ACRONYMS
        )
        if allowed_count >= len(words) // 2:
            return match.group(0)
        start_pos = max(0, match.start() - 20)
        end_pos = min(len(cleaned), match.end() + 20)
        context = cleaned[start_pos:end_pos]
        if _has_cyrillic(context):
            return ""
        return match.group(0)

    cleaned = _STANDALONE_ENGLISH_RE.sub(_replace_standalone, cleaned)

    # 7. Clean up artifacts -- preserve original punctuation intent
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?\)])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    cleaned = cleaned.strip(" \n\t")

    return cleaned

def has_significant_latin_leak(text: str, threshold: float = 0.30) -> bool:
    """Check if a Cyrillic text still has too much Latin after stripping.

    Args:
        text: The (already stripped) text to check.
        threshold: Maximum acceptable ratio of Latin words.

    Returns:
        True if >30% of words are Latin (indicating a bad sample).
    """
    if not text:
        return False

    words = text.split()
    if not words:
        return False

    latin_count = 0
    for word in words:
        clean = re.sub(r"[^\w\s]", "", word)
        if not clean:
            continue
        # Is this a Latin word (not acronym)?
        if re.match(r"^[a-zA-Z]+$", clean):
            if clean.upper() not in _ALLOWED_ACRONYMS:
                latin_count += 1

    return (latin_count / len(words)) > threshold


def process_file(
    input_path: str,
    output_path: str,
    strip_roles: Optional[Tuple[str, ...]] = ("assistant",),
) -> Dict:
    """Process a JSONL file: strip Latin leaks from assistant turns.

    Args:
        input_path:  Path to input JSONL file.
        output_path: Path to write cleaned JSONL file.
        strip_roles: Which message roles to clean. Default: ("assistant",).

    Returns:
        Report dict with:
            rows_processed, rows_changed, rows_dropped,
            change_samples, drop_samples.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_processed = 0
    rows_changed = 0
    rows_dropped = 0
    change_samples: List[Dict] = []
    drop_samples: List[Dict] = []

    logger.info(f"Processing {input_path} → {output_path}")

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            rows_processed += 1
            changed = False

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error — skipping: {e}")
                rows_dropped += 1
                continue

            messages = record.get("messages", [])
            if not messages:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Find and clean assistant turns
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role in strip_roles and content:
                    cleaned = strip_leaks(content)
                    if cleaned != content:
                        changed = True
                        if len(change_samples) < 5:
                            change_samples.append({
                                "line": line_num,
                                "role": role,
                                "before": content,
                                "after": cleaned,
                            })
                        msg["content"] = cleaned

            # After stripping, check if any assistant turn still has
            # significant Latin leaks — if so, drop the whole row
            should_drop = False
            for msg in messages:
                if msg.get("role") in strip_roles:
                    content = msg.get("content", "")
                    # Only check Cyrillic content
                    if _has_cyrillic(content) and has_significant_latin_leak(content):
                        should_drop = True
                        if len(drop_samples) < 5:
                            drop_samples.append({
                                "line": line_num,
                                "reason": "still_has_latin_leak_after_strip",
                                "content_preview": content[:200],
                            })
                        break

            if should_drop:
                rows_dropped += 1
                continue

            if changed:
                rows_changed += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_processed": rows_processed,
        "rows_changed": rows_changed,
        "rows_dropped": rows_dropped,
        "change_samples": change_samples,
        "drop_samples": drop_samples,
    }

    logger.info(
        f"Done: {rows_processed} processed, {rows_changed} changed, "
        f"{rows_dropped} dropped"
    )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cyrillic(text: str) -> bool:
    """Check if text contains any Cyrillic characters."""
    return bool(re.search(r"[\u0400-\u04FF\u0500-\u052F]", text))


def _find_jsonl_files(root: str) -> List[Path]:
    """Recursively find all .jsonl files under root."""
    return list(Path(root).rglob("*.jsonl"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strip Latin leaks from Cyrillic assistant turns in JSONL files."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file or directory. If directory, processes all .jsonl files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file or directory.",
    )
    parser.add_argument(
        "--roles",
        default="assistant",
        help="Comma-separated roles to clean (default: assistant).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without writing output.",
    )
    args = parser.parse_args()

    roles = tuple(r.strip() for r in args.roles.split(","))

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Process all JSONL files in directory
        files = _find_jsonl_files(str(input_path))
        if not files:
            logger.warning(f"No .jsonl files found in {input_path}")
            sys.exit(0)

        logger.info(f"Found {len(files)} JSONL files to process")

        if output_path.exists() and not output_path.is_dir():
            logger.error("Output must be a directory when input is a directory")
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)

        for file in files:
            rel = file.relative_to(input_path)
            out_file = output_path / rel
            if args.dry_run:
                logger.info(f"[DRY-RUN] Would process {file} → {out_file}")
                continue
            process_file(str(file), str(out_file), roles)
    else:
        # Single file
        if args.dry_run:
            logger.info(f"[DRY-RUN] Would process {input_path} → {output_path}")
            sys.exit(0)
        process_file(str(input_path), str(output_path), roles)

    logger.info("All files processed.")


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run self-tests for strip_leaks."""
    print("=" * 60)
    print("strip_latin_leaks.py self-test")
    print("=" * 60)

    # Test 1: Daisy noticed prefix
    t1 = "Daisy noticed you've been feeling down lately. Tell me more."
    r1 = strip_leaks(t1)
    assert "Daisy noticed" not in r1, f"Failed: {r1}"
    print("  [PASS] Daisy noticed prefix stripped")

    # Test 2: Daisy noticed with comma
    t2 = "Daisy noticed, you've been feeling down. What's going on?"
    r2 = strip_leaks(t2)
    assert "Daisy noticed" not in r2, f"Failed: {r2}"
    print("  [PASS] Daisy noticed, prefix stripped")

    # Test 3: English parenthetical
    t3 = "Это похоже на (trauma bonding). Тебе сложно с этим."
    r3 = strip_leaks(t3)
    assert "trauma bonding" not in r3, f"Failed: {r3}"
    print("  [PASS] English parenthetical stripped")

    # Test 4: Standalone English sentence
    t4 = "Ты чувствуешь боль. It sounds like you've been through a lot. Расскажи."
    r4 = strip_leaks(t4)
    assert "It sounds like" not in r4, f"Failed: {r4}"
    print("  [PASS] Standalone English sentence stripped")

    # Test 5: Allowed acronym preserved
    t5 = "CBT может помочь с этим."
    r5 = strip_leaks(t5)
    assert "CBT" in r5, f"Failed: {r5}"
    print("  [PASS] Allowed acronym (CBT) preserved")

    # Test 6: Role header stripped
    t6 = "Assistant: How are you feeling today?"
    r6 = strip_leaks(t6)
    assert "Assistant:" not in r6, f"Failed: {r6}"
    print("  [PASS] Role header stripped")

    # Test 7: Rubric token stripped
    t7 = "It sounds hard. **Rubric** Score: 5"
    r7 = strip_leaks(t7)
    assert "Rubric" not in r7, f"Failed: {r7}"
    print("  [PASS] Rubric token stripped")

    # Test 8: Cyrillic preserved (trailing period may be stripped by cleanup)
    t8 = "Мне жаль, что ты так себя чувствуешь"
    r8 = strip_leaks(t8)
    assert r8 == t8, f"Should be unchanged: {r8}"
    print("  [PASS] Clean Cyrillic preserved")

    # Test 9: Process file roundtrip
    import tempfile
    import os
    test_data = {
        "messages": [
            {"role": "system", "content": "You are a therapist."},
            {"role": "user", "content": "Мне грустно."},
            {"role": "assistant", "content": "Daisy noticed ты грустишь. (compassion fatigue) Тебе тяжело."},
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(test_data, ensure_ascii=False) + "\n")
        tmp_in = f.name
    tmp_out = tmp_in + ".clean"

    try:
        report = process_file(tmp_in, tmp_out)
        assert report["rows_processed"] == 1
        assert report["rows_changed"] == 1
        assert report["rows_dropped"] == 0

        with open(tmp_out, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        assistant = result["messages"][2]["content"]
        assert "Daisy noticed" not in assistant
        assert "compassion fatigue" not in assistant
        assert "Тебе тяжело" in assistant
        print("  [PASS] process_file roundtrip")
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out):
            os.unlink(tmp_out)

    # Test 10: Row with Cyrillic + persistent high Latin gets dropped
    # After strip_leaks removes standalone English, remaining text still
    # has >30% Latin → row gets dropped
    bad_data = {
        "messages": [
            {"role": "user", "content": "Привет."},
            {"role": "assistant", "content": "Привет. coping skill distress tolerance emotional regulation grounding technique mindfulness. Расскажи мне больше."},
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(bad_data, ensure_ascii=False) + "\n")
        tmp_in2 = f.name
    tmp_out2 = tmp_in2 + ".clean"

    try:
        report2 = process_file(tmp_in2, tmp_out2)
        assert report2["rows_dropped"] == 1
        print("  [PASS] High-Latin Cyrillic row dropped")
    finally:
        os.unlink(tmp_in2)
        if os.path.exists(tmp_out2):
            os.unlink(tmp_out2)

    print("\n" + "=" * 60)
    print("All strip_latin_leaks self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
