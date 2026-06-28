"""
Build train.jsonl / val.jsonl using tokenizer.apply_chat_template (same format as inference).

Usage:
  python scripts/prepare_dataset.py --input conversations.json --output-dir ./data

Input JSON: list of objects with "messages" (OpenAI-style roles) and optional "meta".
Or JSONL with one object per line.

Environment:
  BASE_MODEL — tokenizer id (must match training and inference).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_prompts import build_system_from_meta, default_training_system  # noqa: E402

_INFERENCE = Path(__file__).resolve().parents[1] / "inference"
if str(_INFERENCE) not in sys.path:
    sys.path.insert(0, str(_INFERENCE))

from system_prompt import build_system_prompt  # noqa: E402
from state_detector import DaisyState  # noqa: E402


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("JSON root must be a list or use JSONL")


def messages_to_text(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Single training string: chat template without generation prompt for LM loss on full sequence."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


_VALID_STATES: tuple[str, ...] = ("intake", "disclosure", "psychoeducation", "action_planning", "crisis")


def _resolve_system_for_row(row: dict[str, Any], *, is_synthesized: bool) -> str:
    """Build the system prompt for a row, expanding DAISY_SYSTEM_PROMPT_PLACEHOLDER if present."""
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    meta = dict(meta or {})
    state_raw = meta.get("state") if isinstance(meta.get("state"), str) else "intake"
    state: DaisyState = state_raw if state_raw in _VALID_STATES else "intake"  # type: ignore[assignment]

    if is_synthesized:
        language = meta.get("language") or "en"
        locale = "ru" if language == "ru" else "en"
        meta.setdefault("locale", locale)
        meta.setdefault("persona", "flexible")

    locale = meta.get("locale") or meta.get("detected_lang") or "en"
    locale_s = locale.lower()[:8] if isinstance(locale, str) else "en"
    detected = locale_s[:2] if len(locale_s) >= 2 else "en"

    return build_system_prompt(
        locale=locale_s,
        detected_lang=detected,
        onboarding_summary=str(meta.get("onboarding_summary") or ""),
        user_context=str(meta.get("user_context") or ""),
        persona=str(meta.get("persona") or "flexible"),
        force_english=bool(meta.get("force_english", False)),
        user_gender=meta.get("user_gender"),
        psych_profile=meta.get("psych_profile") if isinstance(meta.get("psych_profile"), dict) else None,
        is_onboarding=bool(meta.get("is_onboarding", False)),
        onboarding_step=int(meta.get("onboarding_step", 0)),
        state=state,
    )


def _normalize_messages(row: dict[str, Any], *, is_synthesized: bool) -> list[dict[str, str]] | None:
    raw = row.get("messages")
    if not raw or not isinstance(raw, list):
        return None
    messages: list[dict[str, str]] = []
    has_system = False
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        if role == "system":
            has_system = True
            if content == "DAISY_SYSTEM_PROMPT_PLACEHOLDER":
                content = _resolve_system_for_row(row, is_synthesized=is_synthesized)
        messages.append({"role": role, "content": content})
    if not has_system:
        sys_content = _resolve_system_for_row(row, is_synthesized=is_synthesized)
        messages = [{"role": "system", "content": sys_content}] + messages
    return messages


def _lang_for_row(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if isinstance(meta, dict):
        lang = meta.get("language") or meta.get("locale")
        if isinstance(lang, str):
            return "ru" if lang.lower().startswith("ru") else "en"
    return "en"


def _state_for_row(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if isinstance(meta, dict):
        s = meta.get("state")
        if isinstance(s, str) and s in _VALID_STATES:
            return s
    return "unknown"


def run_merge(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"), trust_remote_code=True)

    synth_dir = Path(args.synthesized_dir)
    synth_files = sorted(p for p in synth_dir.glob("*.jsonl"))
    synth_rows: list[dict[str, Any]] = []
    for p in synth_files:
        synth_rows.extend(load_records(p))

    curated_rows = load_records(Path(args.curated))

    curated_weight = max(1, int(args.curated_weight))
    weighted_curated = curated_rows * curated_weight

    tagged: list[tuple[str, dict[str, Any], bool]] = []
    for row in synth_rows:
        tagged.append(("synth", row, True))
    for row in weighted_curated:
        tagged.append(("curated", row, False))

    stats_state: dict[str, int] = {}
    stats_lang: dict[str, int] = {}
    stats_origin: dict[str, int] = {"curated": 0, "synth": 0}

    texts: list[tuple[str, str, str, str]] = []
    for origin, row, is_synth in tagged:
        messages = _normalize_messages(row, is_synthesized=is_synth)
        if not messages:
            continue
        text = messages_to_text(tokenizer, messages, add_generation_prompt=False)
        state = _state_for_row(row)
        lang = _lang_for_row(row)
        texts.append((text, origin, state, lang))
        stats_state[state] = stats_state.get(state, 0) + 1
        stats_lang[lang] = stats_lang.get(lang, 0) + 1
        stats_origin[origin] = stats_origin.get(origin, 0) + 1

    if not texts:
        raise SystemExit("No examples collected from synthesized or curated inputs")

    random.shuffle(texts)
    n_total = len(texts)
    n_val = max(1, int(n_total * args.val_split))
    val = texts[:n_val]
    train = texts[n_val:]

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.output_train).open("w", encoding="utf-8") as f:
        for t, *_ in train:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with Path(args.output_val).open("w", encoding="utf-8") as f:
        for t, *_ in val:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    def _bucket(items: list[tuple[str, str, str, str]], key_idx: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for row_tuple in items:
            k = row_tuple[key_idx]
            out[k] = out.get(k, 0) + 1
        return out

    print(f"Synthesized files: {len(synth_files)}")
    print(f"Curated rows (raw): {len(curated_rows)} | weight={curated_weight} | expanded={len(weighted_curated)}")
    print(f"Total examples: {n_total} (train={len(train)}, val={len(val)})")
    print(f"Origin: curated={stats_origin['curated']} synth={stats_origin['synth']}")
    print(f"Language: {stats_lang}")
    print(f"State: {stats_state}")
    print("--- train ---")
    print(f"  origin: {_bucket(train, 1)}")
    print(f"  state:  {_bucket(train, 2)}")
    print(f"  lang:   {_bucket(train, 3)}")
    print("--- val ---")
    print(f"  origin: {_bucket(val, 1)}")
    print(f"  state:  {_bucket(val, 2)}")
    print(f"  lang:   {_bucket(val, 3)}")
    print(f"Wrote {len(train)} -> {args.output_train}")
    print(f"Wrote {len(val)} -> {args.output_val}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        parser = argparse.ArgumentParser(prog="prepare_dataset.py merge")
        parser.add_argument("--synthesized-dir", type=Path, required=True)
        parser.add_argument("--curated", type=Path, required=True)
        parser.add_argument("--output-train", type=Path, required=True)
        parser.add_argument("--output-val", type=Path, required=True)
        parser.add_argument("--val-split", type=float, default=0.1)
        parser.add_argument("--curated-weight", type=int, default=5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--base-model",
            default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        )
        args = parser.parse_args(sys.argv[2:])
        run_merge(args)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON array file or JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"), trust_remote_code=True)

    records = load_records(args.input)
    texts: list[str] = []

    for row in records:
        messages = row.get("messages")
        if not messages:
            continue
        messages = list(messages)
        # Ensure system message (aligned with inference when using meta — see docs/DATASET.md)
        if not any(m.get("role") == "system" for m in messages):
            if row.get("system") is not None:
                sys_content = str(row["system"])
            elif isinstance(row.get("meta"), dict):
                sys_content = build_system_from_meta(row["meta"])
            else:
                sys_content = default_training_system()
            messages = [{"role": "system", "content": sys_content}] + messages
        text = messages_to_text(tokenizer, messages, add_generation_prompt=False)
        texts.append(text)

    if not texts:
        raise SystemExit("No valid examples; expected messages[] in each record")

    random.shuffle(texts)
    n_val = max(1, int(len(texts) * args.val_ratio))
    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for t in train_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for t in val_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_texts)} train / {len(val_texts)} val -> {args.output_dir}")


if __name__ == "__main__":
    main()
