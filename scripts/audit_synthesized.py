"""
Audit all files in data/synthesized/ for voice-contract violations in
assistant turns: cliché openers, stacked questions, hollow warmth markers.

Read-only. Prints findings. Does not modify anything.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = REPO_ROOT / "data" / "synthesized"

CLICHE_OPENERS = (
    "It sounds like",
    "That sounds like",
    "That must",
    "I can hear",
    "I can imagine",
    "I understand",
    "I hear you",
    "Sounds like",
    "Похоже",
    "Звучит",
    "Это звучит",
)

HOLLOW_WARMTH_SUBSTRINGS = (
    "big accomplishment",
    "proud of you",
    "that's great",
    "great job",
    "well done",
)

SENTENCE_FINAL_EXCL = re.compile(r"!(?=\s|$|[\"'»)]|\n)")


def starts_with_cliche(text: str) -> str | None:
    stripped = text.lstrip()
    for op in CLICHE_OPENERS:
        if stripped.startswith(op):
            return op
    return None


def stacked_question_count(text: str) -> int:
    return text.count("?")


def find_hollow_warmth(text: str) -> list[str]:
    lower = text.lower()
    hits: list[str] = []
    for needle in HOLLOW_WARMTH_SUBSTRINGS:
        if needle in lower:
            hits.append(needle)
    return hits


def has_sentence_final_excl(text: str) -> bool:
    return bool(SENTENCE_FINAL_EXCL.search(text))


def main() -> None:
    files = sorted(SYN_DIR.glob("*.jsonl"))
    total_examples = 0
    findings: list[tuple[str, int, str, list[str], str]] = []

    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                total_examples += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[skip] invalid JSON in {f.name}:{idx}", file=sys.stderr)
                    continue

                assistant_turns = [
                    m["content"]
                    for m in obj.get("messages", [])
                    if m.get("role") == "assistant"
                ]
                meta = obj.get("meta", {})
                label = meta.get("state") or meta.get("source_folder") or ""

                for turn_idx, text in enumerate(assistant_turns):
                    hits: list[str] = []

                    op = starts_with_cliche(text)
                    if op is not None:
                        hits.append(f"cliche_opener: {op!r}")

                    qc = stacked_question_count(text)
                    if qc > 1:
                        hits.append(f"stacked_questions: {qc}")

                    if has_sentence_final_excl(text):
                        hits.append("sentence_final_exclamation")

                    warmth = find_hollow_warmth(text)
                    for w in warmth:
                        hits.append(f"hollow_warmth: {w!r}")

                    if hits:
                        findings.append((f.name, idx, label, hits, text))

    by_pattern: dict[str, int] = {}
    for _, _, _, hits, _ in findings:
        for h in hits:
            key = h.split(":", 1)[0]
            by_pattern[key] = by_pattern.get(key, 0) + 1

    print("=" * 72)
    print(f"Scanned {len(files)} files, {total_examples} examples")
    print(f"Examples with at least one hit: {len(findings)}")
    print("Hits by pattern:")
    for k in sorted(by_pattern):
        print(f"  {k}: {by_pattern[k]}")
    print("=" * 72)
    print()

    for fname, idx, label, hits, text in findings:
        print(f"--- {fname}:{idx}  [{label}]")
        for h in hits:
            print(f"    {h}")
        print(f"    TEXT: {text}")
        print()


if __name__ == "__main__":
    main()
