"""
Build a local book technique index for in-container RAG.

Chunks data/md, embeds with sentence-transformers, writes book_index.json.
Copy output to inference/knowledge/book_index.json for deployment.

  python scripts/build_book_index.py --output inference/knowledge/book_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from md_to_dialogues import _infer_locale, _iter_md_files, _title_from_md  # noqa: E402
from rich_md_content import _chunk_paragraphs, _paragraphs, _split_sections  # noqa: E402

STATE_HINTS = (
    ("intake", re.compile(r"\bintake|assessment|начал|знакомств", re.I)),
    ("disclosure", re.compile(r"\bdisclosure|emotion|grief|раскрыт|горе", re.I)),
    ("action_planning", re.compile(r"\bhomework|behavioral activation|план|шаг", re.I)),
    ("psychoeducation", re.compile(r"\bpsychoeducation|explain|объясн", re.I)),
    ("crisis", re.compile(r"\bcrisis|suicid|кризис", re.I)),
)
MODALITY_HINTS = (
    ("cbt", re.compile(r"\bCBT|cognitive|кпт", re.I)),
    ("dbt", re.compile(r"\bDBT|dialectical|дбт", re.I)),
    ("act", re.compile(r"\bACT|acceptance", re.I)),
    ("grief", re.compile(r"\bgrief|loss|горе", re.I)),
)


def _infer_state(path: str, section: str, text: str) -> str:
    blob = f"{path} {section} {text[:400]}"
    for name, pat in STATE_HINTS:
        if pat.search(blob):
            return name
    return "disclosure"


def _infer_modality(path: str) -> str:
    for name, pat in MODALITY_HINTS:
        if pat.search(path):
            return name
    return "general"


def _is_technique_chunk(text: str) -> bool:
    """Skip TOC, copyright, and raw bibliography noise."""
    low = text.lower()
    if len(text) < 120 or len(text) > 1200:
        return False
    if "table of contents" in low or "isbn" in low or "copyright" in low:
        return False
    if text.count("\n") > 8:
        return False
    technique_markers = (
        "reflect", "question", "validate", "client", "therapist",
        "отраз", "вопрос", "клиент", "техник", "skill", "intervention",
    )
    return any(m in low for m in technique_markers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build book RAG index")
    ap.add_argument("--md-root", type=Path, default=_ROOT / "data" / "md")
    ap.add_argument("--output", type=Path, default=_ROOT / "inference" / "knowledge" / "book_index.json")
    ap.add_argument("--max-chunks", type=int, default=500)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("pip install sentence-transformers") from exc

    chunks: list[dict] = []
    for md_path in _iter_md_files(args.md_root):
        rel = str(md_path.relative_to(args.md_root)).replace("\\", "/")
        locale = _infer_locale(rel)
        modality = _infer_modality(rel)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        title = _title_from_md(text)
        for section_title, section_body in _split_sections(text):
            for para_chunk in _chunk_paragraphs(_paragraphs(section_body)):
                if not _is_technique_chunk(para_chunk):
                    continue
                cid = hashlib.sha256(f"{rel}|{para_chunk[:120]}".encode()).hexdigest()[:12]
                state = _infer_state(rel, section_title or title, para_chunk)
                # Distill to a short technique line (first 2 sentences)
                sentences = re.split(r"(?<=[.!?])\s+", para_chunk.strip())
                short = " ".join(sentences[:2]).strip()
                if len(short) < 80:
                    continue
                chunks.append(
                    {
                        "id": cid,
                        "modality": modality,
                        "state": state,
                        "locale": locale,
                        "text": short[:900],
                    }
                )
                if len(chunks) >= args.max_chunks:
                    break
            if len(chunks) >= args.max_chunks:
                break
        if len(chunks) >= args.max_chunks:
            break

    print(f"Embedding {len(chunks)} chunks with {args.model}")
    model = SentenceTransformer(args.model)
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    vec_list = [[float(x) for x in v] for v in vectors]

    out = {
        "version": 1,
        "model": args.model,
        "chunks": chunks,
        "vectors": vec_list,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {args.output} ({len(chunks)} chunks, {len(vec_list)} vectors)")


if __name__ == "__main__":
    main()
