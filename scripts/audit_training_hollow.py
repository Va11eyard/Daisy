"""Audit training JSONL for hollow assistant openers (It sounds like / I'm sorry you're).

Usage:
  python scripts/audit_training_hollow.py
  python scripts/audit_training_hollow.py --data-dir E:/WebstormProjects/Daisy-Model/data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_ASSISTANT_SEGMENT = re.compile(
    r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
    re.DOTALL | re.IGNORECASE,
)

_HOLLOW_PATTERNS = (
    re.compile(r"^it sounds like you", re.I),
    re.compile(r"^i'?m sorry you'?re", re.I),
    re.compile(r"^that must be really", re.I),
    re.compile(r"^that must have felt", re.I),
    re.compile(r"^i hear that", re.I),
)


def _last_assistant(text: str) -> str:
    matches = _ASSISTANT_SEGMENT.findall(text)
    return matches[-1].strip() if matches else ""


def _hollow_kind(assistant: str) -> str | None:
    for pat in _HOLLOW_PATTERNS:
        if pat.search(assistant.strip()):
            return pat.pattern
    return None


def audit_file(path: Path) -> dict:
    total = 0
    hollow = 0
    samples: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("text") or ""
        assistant = _last_assistant(text)
        if not assistant:
            continue
        total += 1
        kind = _hollow_kind(assistant)
        if kind:
            hollow += 1
            if len(samples) < 5:
                samples.append(assistant[:120])
    return {
        "path": str(path),
        "rows": total,
        "hollow": hollow,
        "pct": round(100.0 * hollow / total, 1) if total else 0.0,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        alt = Path("E:/WebstormProjects/Daisy-Model/data")
        if alt.is_dir():
            data_dir = alt
        else:
            print(f"No data dir: {data_dir}", file=sys.stderr)
            return 1

    names = ("train_v3.jsonl", "train_v12.jsonl", "train_v13.jsonl", "val_v3.jsonl")
    any_file = False
    for name in names:
        path = data_dir / name
        if not path.is_file():
            continue
        any_file = True
        r = audit_file(path)
        print(f"\n{name}: {r['hollow']}/{r['rows']} hollow ({r['pct']}%)")
        for s in r["samples"]:
            print(f"  sample: {s!r}")

    if not any_file:
        print("No train_v3/v12/v13 JSONL found — copy dataset into data/ before retrain.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
