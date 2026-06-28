"""
Build train_v12.jsonl / val_v12.jsonl from GOOD training data only (train_v2 + train_v3 + RU seed).

Drops book-dump rows (In plain language:, source_md:, raw citations, hollow sympathy).
Uses DAISY_PROMPT_MODE=full when rendering new RU seed dialogues.

Usage:
  python scripts/build_v12_ru_seed.py
  python scripts/prepare_v12_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "inference"
SCRIPTS = ROOT / "scripts"
if str(INFERENCE) not in sys.path:
    sys.path.insert(0, str(INFERENCE))

from voice_qc import violates_voice_contract  # noqa: E402

_ASSISTANT_SEGMENT = re.compile(
    r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
    re.DOTALL | re.IGNORECASE,
)

_BOOK_DUMP_MARKERS = (
    "in plain language:",
    "here's a careful reading:",
    "один конкретный микро-шаг",
    "по смыслу терминов",
    "overview of the program",
    "table of contents",
    "chapter headings",
)

_HELD_OUT_USER_SNIPPETS = (
    "you are answering too short",
    "broke up w my bf",
    "hello still feeling sad",
    "мне тревожно сегодня",
    "я не знаю что делать",
)

_META_VAL_ROWS = [
    {
        "text": (
            "<|im_start|>system\nYou are Daisy.\n\n"
            "<|im_start|>user\nwork has been overwhelming lately\n"
            "<|im_start|>assistant\nThat must be incredibly stressful.\n"
            "<|im_start|>user\nYou are answering too short\n"
            "<|im_start|>assistant\n"
            "When work keeps piling up, it can feel like there is no room to breathe. "
            "What part of the overwhelm is hitting you hardest today?"
            ""
        )
    },
]


def _last_assistant_text(chat_text: str) -> str:
    matches = _ASSISTANT_SEGMENT.findall(chat_text)
    return matches[-1].strip() if matches else ""


def _is_book_dump(text: str) -> bool:
    if "source_md:" in text.lower():
        return True
    assistant = _last_assistant_text(text).lower()
    if not assistant:
        return True
    return any(m in assistant for m in _BOOK_DUMP_MARKERS)


def _is_quality_row(text: str) -> bool:
    if not text or not text.strip():
        return False
    if _is_book_dump(text):
        return False
    assistant = _last_assistant_text(text)
    if not assistant or len(assistant) < 25:
        return False
    if violates_voice_contract(assistant, "intake") and violates_voice_contract(assistant, "disclosure"):
        return False
    return True


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_jsonl_texts(path: Path) -> list[str]:
    rows: list[str] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or ""
            if t:
                rows.append(t)
    return rows


def _ensure_ru_seed() -> Path:
    seed_json = ROOT / "data" / "raw" / "v12_ru_dialogues.json"
    seed_script = SCRIPTS / "build_v12_ru_seed.py"
    if not seed_json.is_file() and seed_script.is_file():
        subprocess.run([sys.executable, str(seed_script)], check=True, cwd=ROOT)
    return seed_json


def _render_ru_seed_rows(seed_json: Path) -> list[str]:
    if not seed_json.is_file():
        return []
    rendered_path = ROOT / "data" / "raw" / "v12_ru_rendered.jsonl"
    env = os.environ.copy()
    env["DAISY_PROMPT_MODE"] = "full"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_dataset.py"),
            "--input",
            str(seed_json),
            "--output-dir",
            str(ROOT / "data" / "raw"),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    # prepare_dataset writes train.jsonl/val.jsonl under output-dir; use train only
    ru_train = ROOT / "data" / "raw" / "train.jsonl"
    if not ru_train.is_file():
        return []
    texts = _load_jsonl_texts(ru_train)
    with rendered_path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return texts


def _is_held_out(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in _HELD_OUT_USER_SNIPPETS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[
            ROOT / "data" / "train_v2.jsonl",
            ROOT / "data" / "train_v3.jsonl",
        ],
    )
    parser.add_argument("--ru-distilled", type=Path, default=ROOT / "data" / "raw" / "md_distilled_ru.jsonl")
    parser.add_argument("--output-train", type=Path, default=ROOT / "data" / "train_v12.jsonl")
    parser.add_argument("--output-val", type=Path, default=ROOT / "data" / "val_v12.jsonl")
    parser.add_argument("--val-split", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[str] = []
    dropped = 0
    seen: set[str] = set()

    for inp in args.inputs:
        for t in _load_jsonl_texts(inp):
            if not _is_quality_row(t):
                dropped += 1
                continue
            h = _row_hash(t)
            if h in seen:
                continue
            seen.add(h)
            rows.append(t)

    seed_json = _ensure_ru_seed()
    for t in _render_ru_seed_rows(seed_json):
        if not _is_quality_row(t):
            dropped += 1
            continue
        h = _row_hash(t)
        if h in seen:
            continue
        seen.add(h)
        rows.append(t)

    if args.ru_distilled.is_file():
        env = os.environ.copy()
        env["DAISY_PROMPT_MODE"] = "full"
        tmp_out = ROOT / "data" / "raw" / "_ru_distill_out"
        tmp_out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "prepare_dataset.py"),
                "--input",
                str(args.ru_distilled),
                "--output-dir",
                str(tmp_out),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
        for t in _load_jsonl_texts(tmp_out / "train.jsonl"):
            if not _is_quality_row(t):
                dropped += 1
                continue
            h = _row_hash(t)
            if h in seen:
                continue
            seen.add(h)
            rows.append(t)

    random.seed(args.seed)
    random.shuffle(rows)

    held_out = [t for t in rows if _is_held_out(t)]
    pool = [t for t in rows if not _is_held_out(t)]
    n_val = max(1, int(len(pool) * args.val_split))
    val_rows = pool[:n_val] + held_out[: min(20, len(held_out))]
    val_rows += [r["text"] for r in _META_VAL_ROWS]
    train_rows = pool[n_val:]

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    with args.output_train.open("w", encoding="utf-8") as f:
        for t in train_rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with args.output_val.open("w", encoding="utf-8") as f:
        for t in val_rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"Rows dropped (book-dump / hollow / dup): {dropped}")
    print(f"Wrote train={len(train_rows)} -> {args.output_train}")
    print(f"Wrote val={len(val_rows)} -> {args.output_val}")


if __name__ == "__main__":
    main()
