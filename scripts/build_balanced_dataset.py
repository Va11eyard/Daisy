"""
build_balanced_dataset.py — Build training dataset with balanced EN/RU/KK mix.

Aggregates data from multiple sources, balances to target locale mix,
validates quality constraints, and outputs a shuffled training JSONL.

Target mix: EN 40%, RU 35%, KK 25% (minimums: ≥20% RU, ≥15% KK)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daisy.build_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MIX = {"en": 0.40, "ru": 0.35, "kk": 0.25}

MINIMUM_LOCALE_FRACTIONS = {
    "ru": 0.35,  # plan gate: ≥35% RU assistant turns
    "kk": 0.15,  # ≥15% KK
    "en": 0.25,  # remainder after RU/KK uplift
}

# Known canned responses to check for duplication
_CANNED_GREETINGS = [
    "hey -- i'm glad you're here. what's on your mind today?",
    "hey — i'm glad you're here. what's on your mind today?",
    "hey - i'm glad you're here. what's on your mind?",
    "i'm glad you're here. what's on your mind?",
    "i'm glad you're here. how can i help you today?",
    "hello! i'm glad you're here. what brings you in today?",
    "hey there! i'm glad you're here. what would you like to talk about?",
    "hey, i'm glad you're here. what's going on?",
]

# Polish diacritics that shouldn't appear in Cyrillic text
_POLISH_DIACRITICS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

# Acronyms that are OK to keep
_ALLOWED_ACRONYMS = {"DBT", "CBT", "ACT", "EMDR", "PTSD", "OCD", "ADHD"}

_IM_START_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>|<\|im_start\|>(system|user|assistant)\n(.*?)<\|redacted_im_end\|>",
    re.DOTALL,
)


def _messages_from_record(record: Dict) -> List[Dict]:
    """Return ChatML messages from either messages[] or text field."""
    messages = record.get("messages", record.get("conversations", []))
    if messages:
        return messages
    text = record.get("text") or ""
    if not text:
        return []
    parsed: List[Dict] = []
    for m in _IM_START_RE.finditer(text):
        role = m.group(1) or m.group(3)
        content = (m.group(2) or m.group(4) or "").strip()
        if role and content:
            parsed.append({"role": role, "content": content})
    return parsed


def _dedupe_by_assistant(records: List[Dict], max_per_response: int = 1) -> List[Dict]:
    """Drop duplicate assistant replies (template synth oversampling)."""
    seen: Counter = Counter()
    out: List[Dict] = []
    for record in records:
        messages = _messages_from_record(record)
        assistant = next(
            (m.get("content", "") for m in messages if m.get("role") == "assistant"),
            "",
        )
        key = assistant.strip().lower()[:240]
        if not key:
            out.append(record)
            continue
        if seen[key] >= max_per_response:
            continue
        seen[key] += 1
        out.append(record)
    logger.info("Deduped %d -> %d rows (max_per_response=%d)", len(records), len(out), max_per_response)
    return out


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _detect_locale(messages: List[Dict]) -> str:
    """Detect the locale of a conversation from assistant turns.

    Uses a heuristic: checks the language of assistant content.
    Falls back to 'en' if undetectable.
    """
    assistant_texts = [
        m.get("content", "") for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_texts:
        return "en"

    combined = " ".join(assistant_texts)

    # Check for Kazakh-specific characters
    if re.search(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]", combined):
        return "kk"

    # Check for Cyrillic (Russian)
    if re.search(r"[а-яА-ЯёЁ]", combined):
        return "ru"

    # Default to English
    return "en"


def _read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file or all JSONL files in a directory."""
    records: List[Dict] = []
    p = Path(path)
    if not p.exists():
        logger.warning(f"Source file not found: {path}")
        return records

    paths: list[Path]
    if p.is_dir():
        paths = sorted(p.rglob("*.jsonl"))
        if not paths:
            logger.warning(f"No JSONL files in directory: {path}")
            return records
    else:
        paths = [p]

    for file_path in paths:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"{file_path}:{line_num} JSON error — skipping: {e}")

        logger.info(f"Read {len(records)} records cumulative from {file_path}")

    return records


def _count_assistant_turns(records: List[Dict]) -> Dict[str, int]:
    """Count assistant turns by detected locale."""
    counts = Counter()
    for record in records:
        locale = record.get("locale", "")
        if not locale:
            messages = _messages_from_record(record)
            locale = _detect_locale(messages) if messages else "en"
            record["locale"] = locale
        counts[locale] += 1
    return dict(counts)


def build_dataset(
    en_sources: List[str],
    ru_sources: List[str],
    kk_sources: List[str],
    target_mix: Optional[Dict[str, float]] = None,
    output_path: str = "data/train_v15.jsonl",
) -> Dict:
    """Build a balanced training dataset from locale-specific sources.

    Args:
        en_sources:  List of paths to English JSONL files.
        ru_sources:  List of paths to Russian JSONL files.
        kk_sources:  List of paths to Kazakh JSONL files.
        target_mix:  Target locale fraction dict, e.g. {"en": 0.40, "ru": 0.35, "kk": 0.25}.
        output_path: Where to write the output JSONL file.

    Returns:
        Report dict with:
            row_counts: dict of locale → count
            locale_mix: dict of locale → fraction
            quality_metrics: dict of validation results
            output_path: path to output file
    """
    target_mix = target_mix or DEFAULT_TARGET_MIX.copy()

    # --- Read all sources ---
    en_records: List[Dict] = []
    ru_records: List[Dict] = []
    kk_records: List[Dict] = []

    for src in en_sources:
        en_records.extend(_read_jsonl(src))
    for src in ru_sources:
        ru_records.extend(_read_jsonl(src))
    for src in kk_sources:
        kk_records.extend(_read_jsonl(src))

    # Tag all records with locale
    for r in en_records:
        r["locale"] = "en"
    for r in ru_records:
        r["locale"] = "ru"
    for r in kk_records:
        r["locale"] = "kk"

    logger.info(
        f"Raw counts: EN={len(en_records)}, RU={len(ru_records)}, "
        f"KK={len(kk_records)}"
    )

    # --- Balance to target mix ---
    total_target = sum(target_mix.values())
    total_available = len(en_records) + len(ru_records) + len(kk_records)

    if total_available == 0:
        raise ValueError("No data available from any source")

    # Calculate target counts
    en_target = int(total_available * target_mix.get("en", 0.40))
    ru_target = int(total_available * target_mix.get("ru", 0.35))
    kk_target = int(total_available * target_mix.get("kk", 0.25))

    # Sample (with replacement if insufficient data)
    def _sample(records: List[Dict], n: int) -> List[Dict]:
        if len(records) >= n:
            return random.sample(records, n)
        # Oversample if we don't have enough
        result = records.copy()
        while len(result) < n:
            result.extend(random.sample(records, min(n - len(result), len(records))))
        return result[:n]

    balanced_en = _sample(en_records, en_target)
    balanced_ru = _sample(ru_records, ru_target)
    balanced_kk = _sample(kk_records, kk_target)

    combined = balanced_en + balanced_ru + balanced_kk
    random.shuffle(combined)

    def _record_has_latin_leak(record: Dict) -> bool:
        messages = _messages_from_record(record)
        loc = _detect_locale(messages)
        if loc not in ("ru", "kk"):
            return False
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if content and _has_cyrillic(content) and _check_latin_leak(content):
                return True
        return False

    before_leak_filter = len(combined)
    combined = [r for r in combined if not _record_has_latin_leak(r)]
    if before_leak_filter != len(combined):
        logger.info("Dropped %d rows with Latin leaks", before_leak_filter - len(combined))

    # --- Write output ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in combined:
            # Strip internal locale tag before writing (not part of training format)
            clean_record = {k: v for k, v in record.items() if k != "locale"}
            # Ensure record has the expected ChatML format
            if "messages" not in clean_record:
                # Try to normalise
                if "conversations" in clean_record:
                    clean_record["messages"] = clean_record.pop("conversations")
                elif "chat" in clean_record:
                    clean_record["messages"] = clean_record.pop("chat")
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(combined)} records to {output_path}")

    # --- Compute report ---
    row_counts = {
        "en": len(balanced_en),
        "ru": len(balanced_ru),
        "kk": len(balanced_kk),
        "total": len(combined),
    }
    locale_mix = {
        "en": round(len(balanced_en) / len(combined), 4) if combined else 0,
        "ru": round(len(balanced_ru) / len(combined), 4) if combined else 0,
        "kk": round(len(balanced_kk) / len(combined), 4) if combined else 0,
    }

    report = {
        "row_counts": row_counts,
        "locale_mix": locale_mix,
        "output_path": str(output_path),
    }

    return report


def validate_dataset(path: str) -> Dict:
    """Validate a training dataset for quality constraints.

    Checks:
      - Locale mix meets minimums (≥20% RU, ≥15% KK, ≥30% EN)
      - 0 Latin leaks in Cyrillic assistant turns
      - No duplicate canned responses
      - All records have valid ChatML format
      - No structural leaks (role headers, punctuation loops)

    Args:
        path: Path to the JSONL dataset to validate.

    Returns:
        Validation report dict with passed/failed status and details.
    """
    path = Path(path)
    if not path.exists():
        return {"passed": False, "error": f"File not found: {path}"}

    total_records = 0
    locale_counts = Counter()
    latin_leak_count = 0
    latin_leak_samples = []
    canned_count = 0
    canned_samples = []
    structural_leak_count = 0
    structural_samples = []
    invalid_format_count = 0

    # Track seen assistant responses for duplication check
    seen_responses = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_format_count += 1
                continue

            messages = _messages_from_record(record)
            if not messages:
                invalid_format_count += 1
                continue

            # Detect locale
            locale = _detect_locale(messages)
            locale_counts[locale] += 1

            # Check assistant turns
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if not content:
                    continue

                # Check for Latin leaks in Cyrillic content
                if locale in ("ru", "kk") and _has_cyrillic(content):
                    leak = _check_latin_leak(content)
                    if leak:
                        latin_leak_count += 1
                        if len(latin_leak_samples) < 5:
                            latin_leak_samples.append({
                                "line": line_num,
                                "locale": locale,
                                "content_preview": content[:200],
                                "leak_type": leak,
                            })

                # Check for canned responses
                content_lower = content.strip().lower()
                for canned in _CANNED_GREETINGS:
                    if _jaccard_similarity(content_lower, canned) > 0.80:
                        canned_count += 1
                        if len(canned_samples) < 5:
                            canned_samples.append({
                                "line": line_num,
                                "matched_canned": canned,
                                "content_preview": content[:200],
                            })

                # Check for structural leaks
                if _has_structural_leak(content):
                    structural_leak_count += 1
                    if len(structural_samples) < 5:
                        structural_samples.append({
                            "line": line_num,
                            "content_preview": content[:200],
                        })

                # Track for duplication
                resp_key = content_lower[:100]
                seen_responses[resp_key] += 1

    # Compute fractions
    total_with_locale = sum(locale_counts.values())
    locale_mix = {
        loc: round(cnt / total_with_locale, 4) if total_with_locale else 0
        for loc, cnt in locale_counts.items()
    }

    # Check minimums
    minimum_ok = True
    min_failures = []
    for loc, min_frac in MINIMUM_LOCALE_FRACTIONS.items():
        actual = locale_mix.get(loc, 0)
        if actual < min_frac:
            minimum_ok = False
            min_failures.append(f"{loc}: {actual:.2%} < {min_frac:.0%}")

    # Check for duplicates (>50 identical responses — oversampled locales)
    duplicates = {k: v for k, v in seen_responses.items() if v > 50}

    passed = (
        minimum_ok
        and latin_leak_count == 0
        and invalid_format_count == 0
    )

    report = {
        "passed": passed,
        "total_records": total_records,
        "locale_mix": dict(locale_mix),
        "locale_counts": dict(locale_counts),
        "minimum_checks": {
            "passed": minimum_ok,
            "failures": min_failures,
        },
        "latin_leaks": {
            "count": latin_leak_count,
            "samples": latin_leak_samples,
        },
        "canned_responses": {
            "count": canned_count,
            "samples": canned_samples,
        },
        "structural_leaks": {
            "count": structural_leak_count,
            "samples": structural_samples,
        },
        "duplicate_responses": {
            "count": len(duplicates),
            "top_duplicates": sorted(duplicates.items(), key=lambda x: -x[1])[:5],
        },
        "invalid_format_count": invalid_format_count,
    }

    status = "PASSED" if passed else "FAILED"
    logger.info(f"Validation {status}: {total_records} records checked")
    if not passed:
        if not minimum_ok:
            logger.warning(f"  Locale minimums failed: {min_failures}")
        if latin_leak_count:
            logger.warning(f"  Latin leaks: {latin_leak_count}")
        if canned_count:
            logger.warning(f"  Canned responses: {canned_count}")
        if structural_leak_count:
            logger.warning(f"  Structural leaks: {structural_leak_count}")
        if invalid_format_count:
            logger.warning(f"  Invalid format: {invalid_format_count}")
        if duplicates:
            logger.warning(f"  Duplicates: {len(duplicates)}")

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r"[\u0400-\u04FF\u0500-\u052F]", text))


def _check_latin_leak(text: str) -> str:
    """Check for problematic Latin script in Cyrillic text.

    Returns empty string if clean, otherwise a description of the leak.
    """
    # Check for Polish diacritics
    if any(ch in text for ch in _POLISH_DIACRITICS):
        return "polish_diacritics"

    # Check for standalone English sentences (>5 Latin words)
    sentences = re.split(r"[.!?\n]+", text)
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) >= 5:
            latin_words = [
                w for w in words
                if re.match(r"^[a-zA-Z]+$", re.sub(r"[^\w\s]", "", w))
                and re.sub(r"[^\w\s]", "", w).upper() not in _ALLOWED_ACRONYMS
            ]
            if len(latin_words) >= 5:
                return f"standalone_english:{len(latin_words)}_words"

    # Check Latin ratio
    all_words = text.split()
    if all_words:
        latin_count = sum(
            1 for w in all_words
            if re.match(r"^[a-zA-Z]+$", re.sub(r"[^\w\s]", "", w))
            and re.sub(r"[^\w\s]", "", w).upper() not in _ALLOWED_ACRONYMS
        )
        if latin_count / len(all_words) > 0.10:
            return f"high_latin_ratio:{latin_count}/{len(all_words)}"

    return ""


def _has_structural_leak(text: str) -> bool:
    """Check for structural problems in generated text."""
    if re.search(r"^(Assistant|Question|User|Human):\s*", text, re.IGNORECASE):
        return True
    if re.search(r"\*\*Rubric\*\*|Score:\s*\d+|# (Excellent|Good|Fair|Poor)", text, re.IGNORECASE):
        return True
    if re.search(r"[.,;:!?]{3,}", text):
        return True
    return False


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity using character bigrams."""
    def _bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}
    bg_a = _bigrams(a)
    bg_b = _bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_sources(patterns: List[str]) -> List[str]:
    """Find source files matching given glob patterns or exact paths."""
    found = []
    for pat in patterns:
        p = Path(pat)
        if p.exists():
            found.append(str(p))
        else:
            # Try glob
            parent = p.parent if p.parent != Path(".") else Path(".")
            if parent.exists():
                matches = list(parent.glob(p.name))
                found.extend(str(m) for m in matches)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced EN/RU/KK training dataset for Daisy Qwen3 LoRA v15."
    )
    parser.add_argument(
        "--en-sources",
        nargs="+",
        default=[],
        help="English source JSONL files or globs.",
    )
    parser.add_argument(
        "--ru-sources",
        nargs="+",
        default=[],
        help="Russian source JSONL files or globs.",
    )
    parser.add_argument(
        "--kk-sources",
        nargs="+",
        default=[],
        help="Kazakh source JSONL files or globs.",
    )
    parser.add_argument(
        "--output",
        default="data/train_v15.jsonl",
        help="Output path for balanced dataset.",
    )
    parser.add_argument(
        "--target-mix",
        type=str,
        default=None,
        help='JSON string of target mix, e.g. \'{"en":0.40,"ru":0.35,"kk":0.25}\'',
    )
    parser.add_argument(
        "--find-sources",
        action="store_true",
        help="Auto-find sources in data/ and data/synthesized/",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Only validate an existing dataset (skip building).",
    )
    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        report = validate_dataset(args.validate_only)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0 if report["passed"] else 1)

    # Parse target mix
    target_mix = None
    if args.target_mix:
        target_mix = json.loads(args.target_mix)

    # Auto-find sources
    en_sources = args.en_sources
    ru_sources = args.ru_sources
    kk_sources = args.kk_sources

    if args.find_sources:
        data_dirs = ["data", "data/synthesized", "training"]
        for d in data_dirs:
            if not Path(d).exists():
                continue
            if not en_sources:
                en_sources = _find_sources([f"{d}/*en*.jsonl", f"{d}/*english*.jsonl"])
            if not ru_sources:
                ru_sources = _find_sources([f"{d}/*ru*.jsonl", f"{d}/*russian*.jsonl"])
            if not kk_sources:
                kk_sources = _find_sources([f"{d}/*kk*.jsonl", f"{d}/*kazakh*.jsonl"])

    logger.info(f"EN sources: {en_sources or '(none)'}")
    logger.info(f"RU sources: {ru_sources or '(none)'}")
    logger.info(f"KK sources: {kk_sources or '(none)'}")

    if not (en_sources or ru_sources or kk_sources):
        logger.error("No data sources found. Use --find-sources or specify explicitly.")
        sys.exit(1)

    # Build dataset
    report = build_dataset(
        en_sources=en_sources,
        ru_sources=ru_sources,
        kk_sources=kk_sources,
        target_mix=target_mix,
        output_path=args.output,
    )

    # Validate
    logger.info("Validating built dataset...")
    validation = validate_dataset(args.output)

    report["quality_metrics"] = validation

    # Print report
    print("\n" + "=" * 60)
    print("Dataset Build Report")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    status = "PASSED" if validation["passed"] else "FAILED"
    logger.info(f"Overall validation: {status}")
    sys.exit(0 if validation["passed"] else 1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run self-tests with synthetic data."""
    print("=" * 60)
    print("build_balanced_dataset.py self-test")
    print("=" * 60)

    import tempfile

    # Create synthetic data
    en_data = [
        {"messages": [
            {"role": "user", "content": f"I'm feeling issue {i}"},
            {"role": "assistant", "content": f"I hear that you're feeling issue {i}. Tell me more."},
        ]}
        for i in range(100)
    ]
    ru_data = [
        {"messages": [
            {"role": "user", "content": f"Я чувствую проблему {i}"},
            {"role": "assistant", "content": f"Мне жаль, что ты чувствуешь проблему {i}. Расскажи подробнее."},
        ]}
        for i in range(80)
    ]
    kk_data = [
        {"messages": [
            {"role": "user", "content": f"Мен мәселе {i} сезінемін"},
            {"role": "assistant", "content": f"Мен сіздің мәселеңізді түсінемін {i}. Толығырақ айтып бересіз бе?"},
        ]}
        for i in range(60)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        en_path = Path(tmp) / "en.jsonl"
        ru_path = Path(tmp) / "ru.jsonl"
        kk_path = Path(tmp) / "kk.jsonl"
        out_path = Path(tmp) / "train_v15.jsonl"

        for path, data in [(en_path, en_data), (ru_path, ru_data), (kk_path, kk_data)]:
            with path.open("w", encoding="utf-8") as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Test build
        report = build_dataset(
            en_sources=[str(en_path)],
            ru_sources=[str(ru_path)],
            kk_sources=[str(kk_path)],
            output_path=str(out_path),
        )

        assert report["row_counts"]["total"] > 0
        assert report["locale_mix"]["en"] > 0
        assert report["locale_mix"]["ru"] > 0
        assert report["locale_mix"]["kk"] > 0
        print(f"  [PASS] build_dataset: {report['row_counts']}")

        # Test validate
        validation = validate_dataset(str(out_path))
        assert validation["total_records"] == report["row_counts"]["total"]
        assert validation["latin_leaks"]["count"] == 0
        assert validation["canned_responses"]["count"] == 0
        print(f"  [PASS] validate_dataset: passed={validation['passed']}")

        # Test with target mix
        report2 = build_dataset(
            en_sources=[str(en_path)],
            ru_sources=[str(ru_path)],
            kk_sources=[str(kk_path)],
            target_mix={"en": 0.40, "ru": 0.35, "kk": 0.25},
            output_path=str(out_path) + "2",
        )
        mix = report2["locale_mix"]
        # Just verify all locales are present
        assert mix["en"] > 0 and mix["ru"] > 0 and mix["kk"] > 0
        print(f"  [PASS] target_mix: EN={mix['en']:.2%} RU={mix['ru']:.2%} KK={mix['kk']:.2%}")

    print("\n" + "=" * 60)
    print("All build_balanced_dataset self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
