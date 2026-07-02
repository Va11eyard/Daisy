#!/usr/bin/env python3
"""
Daisy Training Data Locale Audit + Fix Tool
============================================
Audits training JSONL files for locale mix and Latin leaks in Cyrillic
assistant turns. Optionally fixes leaks by stripping English meta-phrases.

Usage:
    python audit_training_locale.py [--fix] [--output-dir PATH] [--report PATH]

Example:
    python audit_training_locale.py --fix --output-dir data/cleaned \
        --report eval/results/training_locale_audit_fixed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("locale_audit")

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_GLOB = "data/train*.jsonl"
DEFAULT_OUTPUT_DIR = "data/cleaned"
DEFAULT_REPORT_PATH = "eval/results/training_locale_audit_fixed.json"

# Cyrillic character range
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
# Latin (ASCII) letters
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")
# Polish diacritics
_POLISH_RE = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")
# "Daisy noticed[,]? " pattern (anywhere in text)
_DAISY_NOTICED_RE = re.compile(r"\s*Daisy noticed,?\s*", re.IGNORECASE)
# English parentheticals like (trauma bonding), (compassion fatigue)
_EN_PARENTHETICAL_RE = re.compile(r"\s*\([a-zA-Z\s\-]+\)\s*")
# Formal Russian вы -> ты replacement rules (conservative)
_RU_FORMAL_RE = re.compile(r"\b(Вы\s+)([а-яё]+)\b", re.IGNORECASE)
_RU_FORMAL_BARE_RE = re.compile(r"\bВы\b")
# Lines to remove if too much Latin remains
_MAX_LATIN_RATIO = 0.30
# Minimum words for a line to be considered valid
_MIN_WORDS = 5


def _detect_locale(text: str) -> str:
    """
    Detect primary locale of text based on script.
    Returns 'ru', 'kk', 'en', or 'unknown'.
    """
    if not text or not isinstance(text, str):
        return "unknown"

    cyrillic = len(_CYRILLIC_RE.findall(text))
    latin = len(re.findall(r"[a-zA-Z]", text))

    if cyrillic > 0 and cyrillic >= latin:
        # Could be RU or KK — differentiate with KK-specific chars
        kk_chars = len(re.findall(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]", text))
        if kk_chars > cyrillic * 0.05:  # >5% Kazakh-specific chars
            return "kk"
        return "ru"
    elif latin > 0:
        return "en"
    return "unknown"


def _has_latin_leak(text: str) -> bool:
    """Check if a Cyrillic assistant turn has Latin word leakage."""
    if not text:
        return False
    latin_words = _LATIN_WORD_RE.findall(text)
    if not latin_words:
        return False
    # Count Latin alpha chars vs Cyrillic alpha chars
    latin_alpha = len(re.findall(r"[a-zA-Z]", text))
    cyrillic_alpha = len(_CYRILLIC_RE.findall(text))
    # Polish diacritics also count as a leak
    if _POLISH_RE.search(text):
        return True
    # If there are Latin words embedded in Cyrillic text, it's a leak
    if cyrillic_alpha > 0 and latin_words:
        return True
    return False


def _extract_latin_words(text: str) -> list[str]:
    """Extract Latin words from text, stripped of whitespace."""
    return [w.strip() for w in _LATIN_WORD_RE.findall(text) if w.strip()]


def _latin_alpha_ratio(text: str) -> float:
    """Calculate ratio of Latin alpha chars to total alpha chars."""
    latin = len(re.findall(r"[a-zA-Z]", text))
    cyrillic = len(_CYRILLIC_RE.findall(text))
    total = latin + cyrillic
    return latin / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# audit_file
# ---------------------------------------------------------------------------

def audit_file(path: str) -> dict[str, Any]:
    """
    Audit a JSONL training file for locale mix and Latin leaks.

    Args:
        path: Path to JSONL file.

    Returns:
        Dict with keys:
            - path: input path
            - n_examples: total rows processed
            - row_locale_mix: Counter dict of user message locales
            - assistant_locale_mix: Counter dict of assistant turn locales
            - cyrillic_assistant_latin_leak_count: rows with leaks
            - cyrillic_assistant_latin_leak_fraction: fraction
            - leak_samples: list of up to 20 sample leaks
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    n_examples = 0
    row_locale_counter: Counter = Counter()
    assistant_locale_counter: Counter = Counter()
    leak_count = 0
    leak_samples: list[dict[str, Any]] = []

    logger.info("Auditing %s ...", path)

    with filepath.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)
                continue

            n_examples += 1

            # Extract user message and assistant response
            messages = row.get("messages", [])
            if not messages:
                continue

            # Find user message (usually role='user')
            user_msg = ""
            assistant_msg = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content

            # Detect locales
            row_locale = _detect_locale(user_msg)
            assistant_locale = _detect_locale(assistant_msg)

            row_locale_counter[row_locale] += 1
            assistant_locale_counter[assistant_locale] += 1

            # Check for Latin leak in Cyrillic assistant turns
            if assistant_locale in ("ru", "kk") and assistant_msg:
                if _has_latin_leak(assistant_msg):
                    leak_count += 1
                    if len(leak_samples) < 20:
                        latin_words = _extract_latin_words(assistant_msg)
                        leak_samples.append({
                            "line": lineno,
                            "row_locale": row_locale,
                            "preview": assistant_msg[:200],
                            "leak": True,
                            "latin_word_count": len(latin_words),
                            "latin_alpha_ratio": round(_latin_alpha_ratio(assistant_msg), 4),
                            "polish_diacritics": bool(_POLISH_RE.search(assistant_msg)),
                            "sample_latin_words": latin_words[:10],
                        })

    cyrillic_count = sum(
        1 for loc, _ in assistant_locale_counter.items() if loc in ("ru", "kk")
    )
    # Properly compute: count of cyrillic assistant turns
    total_cyrillic_assistant = sum(
        cnt for loc, cnt in assistant_locale_counter.items() if loc in ("ru", "kk")
    )
    leak_fraction = (leak_count / total_cyrillic_assistant) if total_cyrillic_assistant > 0 else 0.0

    result: dict[str, Any] = {
        "path": str(path),
        "n_examples": n_examples,
        "row_locale_mix": dict(row_locale_counter),
        "assistant_locale_mix": dict(assistant_locale_counter),
        "cyrillic_assistant_latin_leak_count": leak_count,
        "cyrillic_assistant_latin_leak_fraction": round(leak_fraction, 4),
        "leak_samples": leak_samples,
    }

    logger.info(
        "  %d examples | row locales: %s | assistant locales: %s | leaks: %d/%d (%.1f%%)",
        n_examples,
        dict(row_locale_counter),
        dict(assistant_locale_counter),
        leak_count,
        total_cyrillic_assistant,
        leak_fraction * 100,
    )

    return result


# ---------------------------------------------------------------------------
# fix_latin_leaks
# ---------------------------------------------------------------------------

def fix_latin_leaks(input_path: str, output_path: str) -> dict[str, Any]:
    """
    Fix Latin leaks in a training JSONL file.

    Fixes applied:
    1. Strip "Daisy noticed[,]? " from Cyrillic assistant turns.
    2. Strip English parentheticals like (trauma bonding) from Cyrillic text.
    3. Replace formal "вы" with informal "ты" in RU assistant turns (conservative).
    4. Drop rows where Cyrillic assistant turn is >30% Latin after stripping.
    5. Keep Cyrillic text only for therapy terms that have Cyrillic equivalents.

    Args:
        input_path: Source JSONL file path.
        output_path: Destination JSONL file path (parent dirs created).

    Returns:
        Dict with:
            - rows_processed: int
            - rows_fixed: int
            - rows_dropped: int
            - rows_written: int
            - fix_samples: list of {line, before, after} dicts (max 20)
    """
    infile = Path(input_path)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    outfile = Path(output_path)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rows_processed = 0
    rows_fixed = 0
    rows_dropped = 0
    fix_samples: list[dict[str, Any]] = []

    logger.info("Fixing %s → %s ...", input_path, output_path)

    with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)
                continue

            rows_processed += 1
            was_fixed = False

            # Find assistant message
            messages = row.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    original = msg.get("content", "")
                    if not original:
                        continue

                    assistant_locale = _detect_locale(original)
                    if assistant_locale not in ("ru", "kk"):
                        continue

                    fixed = original

                    # Fix 1: Strip "Daisy noticed[,]? "
                    fixed_new = _DAISY_NOTICED_RE.sub(" ", fixed).strip()
                    if fixed_new != fixed:
                        fixed = fixed_new
                        was_fixed = True

                    # Fix 2: Strip English parentheticals
                    fixed_new = _EN_PARENTHETICAL_RE.sub(" ", fixed).strip()
                    if fixed_new != fixed:
                        fixed = fixed_new
                        was_fixed = True

                    # Fix 3: Replace formal вы → informal ты (conservative)
                    if assistant_locale == "ru":
                        # Replace "Вы " (capitalized, word boundary) with "Ты "
                        fixed_new = _RU_FORMAL_RE.sub(r"Ты \2", fixed)
                        fixed_new = _RU_FORMAL_BARE_RE.sub("Ты", fixed_new)
                        if fixed_new != fixed:
                            fixed = fixed_new
                            was_fixed = True

                    # Fix 4: Clean up multiple spaces from removals
                    fixed = re.sub(r"\s+", " ", fixed).strip()

                    # Check if we should drop this row
                    remaining_latin_ratio = _latin_alpha_ratio(fixed)
                    words = fixed.split()
                    if remaining_latin_ratio > _MAX_LATIN_RATIO and len(words) < _MIN_WORDS:
                        rows_dropped += 1
                        # Don't write this row
                        was_fixed = False
                        row = None
                        break

                    if was_fixed:
                        rows_fixed += 1
                        if len(fix_samples) < 20:
                            fix_samples.append({
                                "line": lineno,
                                "before": original[:200],
                                "after": fixed[:200],
                            })
                        msg["content"] = fixed

                    break  # Only fix first assistant message

            if row is not None:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                # Row was dropped — decrement isn't needed; we just don't write
                pass

    rows_written = rows_processed - rows_dropped

    result: dict[str, Any] = {
        "rows_processed": rows_processed,
        "rows_fixed": rows_fixed,
        "rows_dropped": rows_dropped,
        "rows_written": rows_written,
        "fix_samples": fix_samples,
    }

    logger.info(
        "  Processed: %d | Fixed: %d | Dropped: %d | Written: %d",
        rows_processed, rows_fixed, rows_dropped, rows_written,
    )

    return result


# ---------------------------------------------------------------------------
# find_data_files
# ---------------------------------------------------------------------------

def find_data_files(base_dir: str | None = None, pattern: str = DEFAULT_DATA_GLOB) -> list[Path]:
    """Find all JSONL files matching the pattern."""
    base = Path(base_dir) if base_dir else ROOT
    if "/" in pattern and not pattern.startswith("**"):
        files = sorted(base.glob(pattern))
    else:
        files = sorted(base.glob(pattern))
        if not files and pattern.startswith("data/"):
            files = sorted(ROOT.glob(pattern))
    return files


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(
    audit_before: list[dict[str, Any]],
    audit_after: list[dict[str, Any]] | None,
) -> None:
    """Print a formatted summary table to stderr."""
    logger.info("=" * 90)
    if audit_after:
        logger.info(
            "%-30s %10s %10s %10s %10s %10s %10s",
            "File", "Examples", "EN%", "RU%", "KK%", "Leaks%", "Leaks%After",
        )
    else:
        logger.info(
            "%-30s %10s %10s %10s %10s %10s",
            "File", "Examples", "EN%", "RU%", "KK%", "Leaks%",
        )
    logger.info("-" * 90)

    for before in audit_before:
        path = Path(before["path"]).name
        n = before["n_examples"]
        row_mix = before["row_locale_mix"]
        en_pct = (row_mix.get("en", 0) / n * 100) if n > 0 else 0
        ru_pct = (row_mix.get("ru", 0) / n * 100) if n > 0 else 0
        kk_pct = (row_mix.get("kk", 0) / n * 100) if n > 0 else 0
        leak_pct = before["cyrillic_assistant_latin_leak_fraction"] * 100

        if audit_after:
            # Find matching after audit
            after = next(
                (a for a in audit_after if Path(a["path"]).name == path),
                None,
            )
            leak_after = after["cyrillic_assistant_latin_leak_fraction"] * 100 if after else 0.0
            logger.info(
                "%-30s %10d %9.1f%% %9.1f%% %9.1f%% %9.1f%% %11.1f%%",
                path, n, en_pct, ru_pct, kk_pct, leak_pct, leak_after,
            )
        else:
            logger.info(
                "%-30s %10d %9.1f%% %9.1f%% %9.1f%% %9.1f%%",
                path, n, en_pct, ru_pct, kk_pct, leak_pct,
            )
    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and optionally fix training data locale mix + Latin leaks"
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT),
        help="Base directory to search for JSONL files",
    )
    parser.add_argument(
        "--train",
        dest="file",
        action="append",
        default=[],
        help="Single train file to audit (alias for --file)",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_DATA_GLOB,
        help="Glob pattern for finding JSONL files",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes and write cleaned files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write cleaned JSONL files",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="Path to write audit report JSON",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific file(s) to audit (can be used multiple times)",
    )
    args = parser.parse_args()

    # Discover files
    if args.file:
        data_files = [Path(f) for f in args.file]
        for f in data_files:
            if not f.exists():
                logger.error("Specified file not found: %s", f)
                sys.exit(1)
    else:
        data_files = find_data_files(args.data_dir, args.pattern)

    if not data_files:
        logger.warning("No JSONL files found matching '%s' in '%s'", args.pattern, args.data_dir)
        sys.exit(0)

    logger.info("Found %d JSONL file(s) to audit", len(data_files))

    # Audit (before)
    audit_results_before: list[dict[str, Any]] = []
    for filepath in data_files:
        try:
            result = audit_file(str(filepath))
            audit_results_before.append(result)
        except Exception as exc:
            logger.error("Audit failed for %s: %s", filepath, exc)
            audit_results_before.append({
                "path": str(filepath),
                "n_examples": 0,
                "row_locale_mix": {},
                "assistant_locale_mix": {},
                "cyrillic_assistant_latin_leak_count": 0,
                "cyrillic_assistant_latin_leak_fraction": 0.0,
                "leak_samples": [],
                "error": str(exc),
            })

    # Print before summary
    _print_summary_table(audit_results_before, None)

    # Fix (if requested)
    audit_results_after: list[dict[str, Any]] | None = None
    fix_results: list[dict[str, Any]] = []

    if args.fix:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_results_after = []

        for filepath in data_files:
            outname = filepath.name
            output_path = str(output_dir / outname)
            try:
                fix_result = fix_latin_leaks(str(filepath), output_path)
                fix_result["input_path"] = str(filepath)
                fix_result["output_path"] = output_path
                fix_results.append(fix_result)

                # Re-audit the fixed file
                after_audit = audit_file(output_path)
                after_audit["original_path"] = str(filepath)
                audit_results_after.append(after_audit)
            except Exception as exc:
                logger.error("Fix failed for %s: %s", filepath, exc)
                fix_results.append({
                    "input_path": str(filepath),
                    "output_path": output_path,
                    "rows_processed": 0,
                    "rows_fixed": 0,
                    "rows_dropped": 0,
                    "rows_written": 0,
                    "fix_samples": [],
                    "error": str(exc),
                })

        # Print after summary
        logger.info("")
        logger.info("--- AFTER FIX ---")
        _print_summary_table(audit_results_before, audit_results_after)

        # Print fix details
        logger.info("")
        logger.info("--- FIX DETAILS ---")
        for fr in fix_results:
            logger.info(
                "%s: processed=%d fixed=%d dropped=%d written=%d",
                Path(fr["input_path"]).name,
                fr.get("rows_processed", 0),
                fr.get("rows_fixed", 0),
                fr.get("rows_dropped", 0),
                fr.get("rows_written", 0),
            )

    # Save report
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_audited": len(data_files),
        "audit_before": audit_results_before,
    }
    if audit_results_after is not None:
        report["audit_after"] = audit_results_after
    if fix_results:
        report["fix_results"] = fix_results

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("Audit report saved to: %s", report_path)


if __name__ == "__main__":
    main()


