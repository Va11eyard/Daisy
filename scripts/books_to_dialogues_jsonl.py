"""
Turn book excerpts (Markdown under data/md/) into high-quality synthetic therapy dialogues.

- Therapist-voice system strings (aligned with DAISY_VOICE=therapist at inference).
- Persona + modality inferred from folder path.
- Scenario-based client openers (psychoeducation, technique, self_apply, reflect).
- Optional two-turn dialogues (split at sentence boundaries) for more realistic rhythm.

Usage:
  python scripts/books_to_dialogues_jsonl.py --input-dir data/md --output data/book_dialogues.jsonl
  set DAISY_VOICE=therapist   # optional: match inference deployment
  python scripts/prepare_dataset.py --input data/book_dialogues.jsonl --output-dir data --base-model Qwen/Qwen2.5-7B-Instruct

Skips: *_conversion_*.txt, README, very small files.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from book_dialogue_templates import (  # noqa: E402
    infer_modality,
    persona_for_modality,
    pick_client_opener,
    pick_followup,
    pick_scenario,
    split_chunk_at_sentences,
)
from therapy_training_prompts import build_book_training_system  # noqa: E402

SKIP_NAMES = {"_conversion_log.txt", "_conversion_err.txt", "readme.md"}


def _cyrillic_ratio(s: str) -> float:
    cyr = sum(1 for ch in s if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in s if "a" <= ch.lower() <= "z")
    tot = cyr + lat
    return (cyr / tot) if tot else 0.0


def guess_locale(chunk: str) -> str:
    r = _cyrillic_ratio(chunk)
    if r >= 0.35:
        return "ru"
    return "en"


def guess_locale_with_path(chunk: str, path: Path) -> str:
    low = path.as_posix().lower()
    if "kk" in low or "қазақ" in low or "kazakh" in low:
        return "kk"
    return guess_locale(chunk)


def sanitize_chunk(text: str) -> str:
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_title(md: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", md, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def chunk_markdown(text: str, max_chars: int, min_chars: int) -> list[str]:
    title = extract_title(text)
    body = text
    if title:
        body = re.sub(r"^#\s+.+$", "", body, count=1, flags=re.MULTILINE)
    body = body.strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            s = "\n\n".join(buf)
            s = sanitize_chunk(s)
            if len(s) >= min_chars:
                chunks.append(s)
        buf = []
        size = 0

    for p in paras:
        plen = len(p)
        if plen > max_chars:
            flush()
            for i in range(0, plen, max_chars):
                piece = sanitize_chunk(p[i : i + max_chars])
                if len(piece) >= min_chars:
                    chunks.append(piece)
            continue
        if size + plen + 2 <= max_chars:
            buf.append(p)
            size += plen + 2
        else:
            flush()
            buf = [p]
            size = plen

    flush()
    return chunks


def iter_md_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.md"):
        if not p.is_file():
            continue
        if p.name.lower() in SKIP_NAMES or p.name.startswith("_"):
            continue
        out.append(p)
    return sorted(out)


def build_record(
    *,
    path: Path,
    root: Path,
    chunk: str,
    idx: int,
    locale: str,
    modality: str | None,
    persona: str,
    scenario: str,
    rng: random.Random,
    multi_turn: bool,
) -> dict:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    rel_s = rel.as_posix()
    system = build_book_training_system(
        locale=locale,
        persona=persona,
        modality=modality,
        source_book=rel_s,
    )
    user1 = pick_client_opener(locale, scenario, rng)

    meta = {
        "locale": locale,
        "persona": persona,
        "source_book": rel_s,
        "chunk_idx": idx,
        "scenario": scenario,
    }
    if modality:
        meta["modality"] = modality

    if multi_turn:
        parts = split_chunk_at_sentences(chunk)
        if parts:
            first, second = parts
            user2 = pick_followup(locale, rng)
            return {
                "system": system,
                "messages": [
                    {"role": "user", "content": user1},
                    {"role": "assistant", "content": first},
                    {"role": "user", "content": user2},
                    {"role": "assistant", "content": second},
                ],
                "meta": meta,
            }

    return {
        "system": system,
        "messages": [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": chunk},
        ],
        "meta": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "md")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "book_dialogues.jsonl")
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--min-chunk-chars", type=int, default=220)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--multi-turn-ratio",
        type=float,
        default=0.38,
        help="Fraction of rows that use two user/assistant turns when the chunk can be split cleanly.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = args.input_dir.resolve()
    files = iter_md_files(root)
    if not files:
        raise SystemExit(f"No .md files under {root}")
    rng.shuffle(files)

    records: list[dict] = []
    limit = args.max_samples if args.max_samples > 0 else None
    for path in files:
        if limit is not None and len(records) >= limit:
            break
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        modality = infer_modality(path)
        persona = persona_for_modality(modality)
        chunks = chunk_markdown(raw, args.chunk_chars, args.min_chunk_chars)
        rng.shuffle(chunks)
        for idx, chunk in enumerate(chunks):
            if limit is not None and len(records) >= limit:
                break
            loc = guess_locale_with_path(chunk, path)
            scenario = pick_scenario(chunk, rng)
            want_multi = rng.random() < args.multi_turn_ratio
            rec = build_record(
                path=path,
                root=root,
                chunk=chunk,
                idx=idx,
                locale=loc,
                modality=modality,
                persona=persona,
                scenario=scenario,
                rng=rng,
                multi_turn=want_multi,
            )
            records.append(rec)

    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} dialogue rows -> {args.output}")


if __name__ == "__main__":
    main()
