"""
Build train_v13.jsonl / val_v13.jsonl — curated sources + shape-balanced mix.

Extends v12 filters with shape bucketing and target composition:
  ~40% reflect+question, ~25% validation/explore, ~15% psychoeducation,
  ~10% action, ~10% other (greeting, crisis, etc.)

Usage:
  python scripts/build_v12_ru_seed.py
  python scripts/build_v13_shape_synth.py
  python scripts/prepare_v13_dataset.py
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
from collections import defaultdict
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

_SHAPE_TARGETS: dict[str, float] = {
    "reflect_plus_question": 0.40,
    "validation_no_question": 0.15,
    "question_other": 0.10,
    "psychoeducation": 0.15,
    "action_step": 0.10,
    "crisis_redirect": 0.03,
    "short_greeting": 0.04,
    "other": 0.03,
}

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
    if not assistant or len(assistant) < 15:
        return False
    if violates_voice_contract(assistant, "intake") and violates_voice_contract(
        assistant, "disclosure"
    ):
        return False
    return True


def _classify_shape(assistant: str) -> str:
    parts = [p for p in re.split(r"[.!?…]+", assistant) if p.strip()]
    has_q = "?" in assistant
    low = assistant.lower()
    if any(k in low for k in ("crisis", "988", "доверия", "кризис", "emergency")):
        return "crisis_redirect"
    if len(assistant) < 80 and has_q:
        return "short_greeting"
    if any(k in low for k in ("cognitive", "когнитив", "distortion", "искажен", "катастроф")):
        return "psychoeducation"
    if any(k in low for k in ("experiment", "step", "попробуй", "шаг", "звонок", "email")):
        return "action_step"
    if has_q and 2 <= len(parts) <= 4:
        return "reflect_plus_question"
    if has_q:
        return "question_other"
    if len(parts) >= 2:
        return "validation_no_question"
    return "other"


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


def _render_json_dialogues(json_path: Path) -> list[str]:
    if not json_path.is_file():
        return []
    env = os.environ.copy()
    env["DAISY_PROMPT_MODE"] = "full"
    out_dir = json_path.parent / f"_rendered_{json_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_dataset.py"),
            "--input",
            str(json_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    return _load_jsonl_texts(out_dir / "train.jsonl")


def _balance_by_shape(rows: list[str], target_n: int, seed: int) -> list[str]:
    """Subsample/oversample to approach _SHAPE_TARGETS."""
    by_shape: dict[str, list[str]] = defaultdict(list)
    for t in rows:
        assistant = _last_assistant_text(t)
        if not assistant:
            continue
        by_shape[_classify_shape(assistant)].append(t)

    rng = random.Random(seed)
    balanced: list[str] = []
    for shape, frac in _SHAPE_TARGETS.items():
        want = max(1, int(target_n * frac))
        pool = by_shape.get(shape, [])
        if not pool:
            continue
        if len(pool) >= want:
            balanced.extend(rng.sample(pool, want))
        else:
            balanced.extend(pool)
            extra = want - len(pool)
            balanced.extend(rng.choices(pool, k=extra))

    rng.shuffle(balanced)
    return balanced[:target_n] if len(balanced) > target_n else balanced


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
    parser.add_argument(
        "--shape-synth",
        type=Path,
        default=ROOT / "data" / "raw" / "v13_shape_synth.json",
    )
    parser.add_argument(
        "--ru-seed",
        type=Path,
        default=ROOT / "data" / "raw" / "v12_ru_dialogues.json",
    )
    parser.add_argument("--ru-distilled", type=Path, default=ROOT / "data" / "raw" / "md_distilled_ru.jsonl")
    parser.add_argument("--target-train", type=int, default=3200)
    parser.add_argument("--output-train", type=Path, default=ROOT / "data" / "train_v13.jsonl")
    parser.add_argument("--output-val", type=Path, default=ROOT / "data" / "val_v13.jsonl")
    parser.add_argument("--val-split", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    synth_script = SCRIPTS / "build_v13_shape_synth.py"
    if not args.shape_synth.is_file() and synth_script.is_file():
        subprocess.run([sys.executable, str(synth_script)], check=True, cwd=ROOT)

    rows: list[str] = []
    dropped = 0
    seen: set[str] = set()

    def _add(text: str) -> None:
        nonlocal dropped
        if not _is_quality_row(text):
            dropped += 1
            return
        h = _row_hash(text)
        if h in seen:
            return
        seen.add(h)
        rows.append(text)

    for inp in args.inputs:
        for t in _load_jsonl_texts(inp):
            _add(t)

    for t in _render_json_dialogues(args.ru_seed):
        _add(t)

    for t in _render_json_dialogues(args.shape_synth):
        _add(t)

    if args.ru_distilled.is_file():
        tmp_out = ROOT / "data" / "raw" / "_ru_distill_v13"
        tmp_out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DAISY_PROMPT_MODE"] = "full"
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
            _add(t)

    balanced = _balance_by_shape(rows, args.target_train, args.seed)

    held_out = [t for t in balanced if _is_held_out(t)]
    pool = [t for t in balanced if not _is_held_out(t)]
    random.seed(args.seed)
    random.shuffle(pool)
    n_val = max(1, int(len(pool) * args.val_split))
    val_rows = pool[:n_val] + held_out[: min(20, len(held_out))]
    val_rows += [r["text"] for r in _META_VAL_ROWS]
    train_rows = pool[n_val:]

    shape_mix: dict[str, int] = defaultdict(int)
    for t in train_rows:
        shape_mix[_classify_shape(_last_assistant_text(t))] += 1

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    with args.output_train.open("w", encoding="utf-8") as f:
        for t in train_rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with args.output_val.open("w", encoding="utf-8") as f:
        for t in val_rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    n_train = len(train_rows)
    reflect_q = shape_mix.get("reflect_plus_question", 0) / max(n_train, 1)
    print(f"Rows dropped (book-dump / hollow / dup): {dropped}")
    print(f"Pool before balance: {len(rows)} -> balanced train target: {len(balanced)}")
    print(f"reflect+question rate: {reflect_q:.3f} (target <=0.70)")
    print(f"shape_mix: {dict(shape_mix)}")
    print(f"Wrote train={n_train} -> {args.output_train}")
    print(f"Wrote val={len(val_rows)} -> {args.output_val}")


if __name__ == "__main__":
    main()
