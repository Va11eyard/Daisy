"""CPU-only audits: RAG corpus genericness + LoRA training data shape."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent if (EVAL_DIR.parent / "inference" / "book_knowledge.py").exists() else EVAL_DIR
INFERENCE_DIR = ROOT / "inference" if (ROOT / "inference" / "book_knowledge.py").exists() else ROOT
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(EVAL_DIR))

from metrics import cosine_sim, distinct_n, templated_shape_rate  # noqa: E402

_BOOK_DUMP_MARKERS = (
    "in plain language:",
    "here's a careful reading:",
    "source_md:",
    "overview of the program",
    "table of contents",
    "chapter headings",
)

_GENERIC_TAIL_RE = re.compile(
    r"what feels most (important|pressing)|что сейчас беспокоит сильнее",
    re.IGNORECASE,
)

BLAND_REFERENCE = (
    "It sounds like things have been really overwhelming. "
    "What feels most important to talk about right now?"
)

META_INSTRUCTION_MARKERS = (
    "ask one open question",
    "reflect what you heard",
    "open question about what feels most pressing",
    "отрази услышанное",
    "один открытый вопрос",
    "avoid reassurance",
    "do not pivot",
)


def audit_rag(index_path: Path, embed_fn) -> dict:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    report: dict = {"n_chunks": len(chunks), "chunks": []}
    bland_vec = embed_fn(BLAND_REFERENCE)
    meta_hits = 0
    sims: list[float] = []

    for ch in chunks:
        text = (ch.get("text") or "").strip()
        low = text.lower()
        is_meta = any(m in low for m in META_INSTRUCTION_MARKERS)
        if is_meta:
            meta_hits += 1
        sim = cosine_sim(embed_fn(text), bland_vec)
        sims.append(sim)
        report["chunks"].append({
            "id": ch.get("id"),
            "state": ch.get("state"),
            "locale": ch.get("locale"),
            "is_meta_instruction": is_meta,
            "similarity_to_bland_output": round(sim, 4),
            "preview": text[:120],
        })

    report["meta_instruction_fraction"] = round(meta_hits / max(len(chunks), 1), 4)
    report["mean_similarity_to_bland_output"] = round(
        sum(sims) / max(len(sims), 1), 4
    )
    return report


def _extract_assistant_from_train_line(obj: dict) -> str | None:
    text = obj.get("text") or ""
    m = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    messages = obj.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "assistant":
                return (msg.get("content") or "").strip()
    return None


def _detect_locale(text: str) -> str:
    """Rough locale from rendered chat text (user + assistant)."""
    cyr = len(re.findall(r"[\u0400-\u04ff]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    kk = len(re.findall(r"[ӘәІіҢңҒғҮүҰұҚқӨөҺһ]", text))
    if kk >= 3:
        return "kk"
    if cyr > lat:
        return "ru"
    if lat > 0:
        return "en"
    return "unknown"


def _is_book_dump_line(obj: dict) -> bool:
    content = _extract_assistant_from_train_line(obj) or ""
    low = content.lower()
    if "source_md:" in low:
        return True
    return any(m in low for m in _BOOK_DUMP_MARKERS)


def _classify_shape(content: str) -> str:
    parts = [p for p in re.split(r"[.!?…]+", content) if p.strip()]
    has_q = "?" in content
    if has_q and 2 <= len(parts) <= 4:
        return "reflect_plus_question"
    if has_q:
        return "question_other"
    if len(parts) >= 2 and not has_q:
        return "validation_no_question"
    if any(k in content.lower() for k in ("cognitive", "когнитив", "distortion", "искажен")):
        return "psychoeducation"
    if any(k in content.lower() for k in ("crisis", "кризис", "988", "доверия", "emergency")):
        return "crisis_redirect"
    if len(content) < 80:
        return "short_greeting"
    return "other"


def audit_lora_data(train_path: Path) -> dict:
    assistants: list[str] = []
    sentence_counts: list[int] = []
    has_reflect_and_question = 0
    book_dump_rows = 0
    voice_contract_fail = 0
    locale_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    generic_tail_hits = 0

    try:
        from voice_qc import violates_voice_contract  # noqa: WPS433
    except ImportError:
        violates_voice_contract = None  # type: ignore[assignment,misc]

    with train_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if _is_book_dump_line(obj):
                book_dump_rows += 1
            row_text = obj.get("text") or ""
            loc = _detect_locale(row_text)
            locale_counts[loc] = locale_counts.get(loc, 0) + 1

            content = _extract_assistant_from_train_line(obj)
            if not content:
                continue
            assistants.append(content)
            parts = [p for p in re.split(r"[.!?…]+", content) if p.strip()]
            sentence_counts.append(len(parts))
            if "?" in content and 2 <= len(parts) <= 4:
                has_reflect_and_question += 1
            shape = _classify_shape(content)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            if _GENERIC_TAIL_RE.search(content):
                generic_tail_hits += 1
            if violates_voice_contract is not None:
                if violates_voice_contract(content, "intake") and violates_voice_contract(
                    content, "disclosure"
                ):
                    voice_contract_fail += 1

    n = len(assistants)
    n_lines = n + book_dump_rows
    return {
        "train_path": str(train_path),
        "n_lines": n_lines,
        "n_examples": n,
        "book_dump_row_fraction": round(book_dump_rows / max(n_lines, 1), 4),
        "locale_mix": locale_counts,
        "shape_mix": shape_counts,
        "generic_tail_fraction": round(generic_tail_hits / max(n, 1), 4),
        "voice_contract_fail_fraction": round(voice_contract_fail / max(n, 1), 4),
        "distinct_2": round(distinct_n(assistants, 2), 4),
        "mean_sentence_count": round(sum(sentence_counts) / max(n, 1), 2),
        "sentence_count_distribution": {
            "2": sum(1 for s in sentence_counts if s == 2),
            "3": sum(1 for s in sentence_counts if s == 3),
            "4": sum(1 for s in sentence_counts if s == 4),
            "other": sum(1 for s in sentence_counts if s not in (2, 3, 4)),
        },
        "reflect_plus_question_shape_rate": round(has_reflect_and_question / max(n, 1), 4),
        "templated_shape_rate": round(
            templated_shape_rate([{"response": a} for a in assistants]), 4
        ),
        "targets": {
            "reflect_plus_question_shape_rate_max": 0.70,
            "book_dump_row_fraction_max": 0.0,
        },
        "sample_targets": assistants[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Train JSONL to audit (default: train_v13 > train_v12 > train_v3)",
    )
    args = parser.parse_args()

    os.environ.setdefault("DAISY_BOOK_KNOWLEDGE", "true")
    from book_knowledge import embed_text  # noqa: WPS433

    index_path = INFERENCE_DIR / "knowledge" / "book_index.json"
    if not index_path.is_file():
        index_path = ROOT / "knowledge" / "book_index.json"

    if args.train is not None:
        train_path = args.train
    else:
        for candidate in (
            ROOT / "data" / "train_v13.jsonl",
            ROOT / "data" / "train_v12.jsonl",
            ROOT / "data" / "train_v3.jsonl",
        ):
            if candidate.is_file():
                train_path = candidate
                break
        else:
            train_path = ROOT / "data" / "archive" / "train.jsonl.retired"

    rag_report = audit_rag(index_path, embed_text)
    lora_report = audit_lora_data(train_path) if train_path.is_file() else {"error": f"{train_path} not found"}

    decoding_doc = {
        "production_defaults": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "voice_regen_temperature": 0.85,
            "voice_regen_repetition_penalty": 1.2,
            "other_regen_temperature_cap": 0.5,
        },
        "note": "Regen paths in score.py lower temperature, which can flatten diversity.",
    }

    out = {
        "rag_audit": rag_report,
        "lora_data_audit": lora_report,
        "decoding_params": decoding_doc,
    }

    out_path = EVAL_DIR / "results" / "offline_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== RAG corpus audit ===")
    print(f"  chunks: {rag_report['n_chunks']}")
    print(f"  meta-instruction fraction: {rag_report['meta_instruction_fraction']}")
    print(f"  mean sim to bland output: {rag_report['mean_similarity_to_bland_output']}")
    print("\n=== LoRA training data audit ===")
    print(f"  file: {train_path}")
    for k, v in lora_report.items():
        if k not in ("sample_targets", "targets"):
            print(f"  {k}: {v}")
    if "reflect_plus_question_shape_rate" in lora_report:
        rate = lora_report["reflect_plus_question_shape_rate"]
        ok = rate <= 0.70
        print(f"  shape_gate (<=0.70): {'PASS' if ok else 'FAIL'} ({rate})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
