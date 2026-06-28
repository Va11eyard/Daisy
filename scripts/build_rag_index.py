"""
Build the Daisy Hybrid RAG indexes from the dialog corpus and therapy books.

Dense (dialog): walks data/synthesized/*/batch_*.jsonl, extracts (user -> assistant)
pairs, embeds the USER turn with a multilingual sentence-transformer.

Sparse (BM25): tokenizes the same dialog user turns, and chunks the therapy book
markdown (--books-dir, e.g. Daisy-Model/data/md) into 200-word windows for exact
terminology grounding.

Artifacts written into inference/knowledge/:
  rag_meta.json    - dialog records aligned row-for-row with the vectors (+ config)
  rag_vectors.npy  - float32, L2-normalized embedding matrix (faiss-free fallback)
  rag_index.faiss  - flat inner-product index (written only when faiss is available)
  bm25_dialog.pkl  - BM25Okapi over dialog user turns (parallel to rag_meta records)
  bm25_books.pkl   - BM25Okapi over book chunks (parallel to bm25_meta books)
  bm25_meta.json   - {dialog_count, books:[{text, lang, source}]}

  python scripts/build_rag_index.py
  python scripts/build_rag_index.py --books-dir E:\\WebstormProjects\\Daisy-Model\\data\\md
  python scripts/build_rag_index.py --no-bm25            # dense dialog only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE = _ROOT / "inference"
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _INFERENCE, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from state_detector import detect_state  # noqa: E402
from rag import _tokenize  # noqa: E402  (shared tokenizer -> corpus/query parity)
from md_to_dialogues import _infer_locale, _iter_md_files  # noqa: E402

_VALID_PHASES = ("intake", "disclosure", "psychoeducation", "action_planning", "crisis")
_DEFAULT_MODEL = os.environ.get(
    "DAISY_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Book cleaning patterns: strip residual headers, page numbers, footnotes, front-matter noise.
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_FOOTNOTE_RE = re.compile(r"^\s*\[\^?\d+\]")
_NOISE_RE = re.compile(
    r"(isbn|copyright|all rights reserved|table of contents|содержание|©)", re.I
)
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _iter_pairs(corpus_dir: Path):
    """Yield (user_text, assistant_text, meta) for each adjacent user->assistant turn."""
    for path in sorted(corpus_dir.rglob("batch_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = obj.get("messages") or []
                meta = {
                    "lang": (obj.get("lang") or "").lower()[:2] or "en",
                    "tone": obj.get("tone") or "",
                    "session_phase": obj.get("session_phase") or "",
                    "emotion_label": obj.get("emotion_label") or "",
                    "source": obj.get("source") or path.parent.name,
                }
                for i in range(len(msgs) - 1):
                    a, b = msgs[i], msgs[i + 1]
                    if a.get("role") == "user" and b.get("role") == "assistant":
                        u = (a.get("content") or "").strip()
                        r = (b.get("content") or "").strip()
                        if len(u) >= 8 and len(r) >= 16:
                            yield u, r, meta


def _derive_phase(user_text: str) -> str:
    try:
        phase = detect_state([{"role": "user", "content": user_text}], check_crisis=False)
    except Exception:
        phase = "intake"
    return phase if phase in _VALID_PHASES else "intake"


def _clean_book_text(raw: str) -> str:
    """Drop markdown headers, page numbers, footnotes, and front-matter noise lines."""
    out: list[str] = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            out.append("")
            continue
        if _HEADER_RE.match(line) or _PAGE_NUM_RE.match(s) or _FOOTNOTE_RE.match(s):
            continue
        if _NOISE_RE.search(s):
            continue
        s = _IMG_RE.sub(" ", s)
        s = _LINK_RE.sub(r"\1", s)
        out.append(s)
    text = "\n".join(out)
    return re.sub(r"[ \t]+", " ", text)


def _cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return cyr / len(letters)


def _detect_lang(text: str) -> str:
    """ru / en per chunk. Cyrillic ratio first; langdetect breaks the ambiguous middle."""
    ratio = _cyrillic_ratio(text)
    if ratio >= 0.30:
        return "ru"
    if ratio <= 0.05:
        return "en"
    try:
        from langdetect import detect

        d = detect(text)
        if d in ("ru", "en"):
            return d
    except Exception:
        pass
    return "ru" if ratio >= 0.15 else "en"


def _chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping fixed-size word windows; drops a too-short trailing window."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if len(piece) < 30 and start != 0:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def _interleave_by_locale(books_dir: Path) -> list[Path]:
    """Order files round-robin across inferred locales so a chunk cap does not
    starve one language (the RU book tree sorts after the EN one by path)."""
    from itertools import zip_longest

    buckets: dict[str, list[Path]] = {"ru": [], "en": [], "other": []}
    for p in _iter_md_files(books_dir):
        loc = _infer_locale(str(p.relative_to(books_dir)))
        buckets.get(loc, buckets["other"]).append(p)
    ordered: list[Path] = []
    for trio in zip_longest(buckets["ru"], buckets["en"], buckets["other"]):
        ordered.extend(p for p in trio if p is not None)
    return ordered


def _build_books(books_dir: Path, size: int, overlap: int, cap: int) -> list[dict]:
    """Chunk therapy book markdown into BM25 records (text, lang, source)."""
    records: list[dict] = []
    for path in _interleave_by_locale(books_dir):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        cleaned = _clean_book_text(raw)
        title = path.stem.replace("_", " ").strip()
        for chunk in _chunk_words(cleaned, size, overlap):
            if len(chunk) < 80:
                continue
            lang = _detect_lang(chunk)
            if lang not in ("en", "ru"):
                continue
            records.append({"text": chunk[:1200], "lang": lang, "source": title})
            if len(records) >= cap:
                print(f"Reached --max-book-chunks={cap}; stopping books ingest")
                return records
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Daisy Hybrid RAG indexes")
    ap.add_argument("--corpus", type=Path, default=_ROOT / "data" / "synthesized")
    ap.add_argument("--out-dir", type=Path, default=_INFERENCE / "knowledge")
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--books-dir", type=Path, default=_ROOT / "data" / "md")
    ap.add_argument("--books-chunk-size", type=int, default=200)
    ap.add_argument("--books-overlap", type=int, default=40)
    ap.add_argument("--max-book-chunks", type=int, default=15000)
    ap.add_argument("--no-bm25", action="store_true", help="dense dialog index only")
    ap.add_argument("--no-books", action="store_true", help="BM25 dialog only, skip books")
    args = ap.parse_args()

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install sentence-transformers numpy") from exc

    records: list[dict] = []
    queries: list[str] = []
    seen: set[str] = set()
    for user_text, assistant_text, meta in _iter_pairs(args.corpus):
        key = user_text.lower()[:160]
        if key in seen:
            continue
        seen.add(key)
        queries.append(user_text)
        records.append(
            {
                "text": assistant_text,          # what we inject as RETRIEVED CONTEXT
                "query": user_text,               # what we embed / match against
                "lang": meta["lang"],
                "phase": _derive_phase(user_text),
                "tone": meta["tone"],
                "emotion": meta["emotion_label"],
                "source": meta["source"],
            }
        )

    if not records:
        raise SystemExit(f"No dialog pairs found under {args.corpus}")

    print(f"Embedding {len(records)} dialog pairs with {args.model} ...")
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    dim = int(vectors.shape[1])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "rag_vectors.npy", vectors)
    meta_out = {
        "version": 1,
        "model": args.model,
        "dim": dim,
        "count": len(records),
        "records": records,
    }
    with (args.out_dir / "rag_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False)

    try:
        import faiss

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, str(args.out_dir / "rag_index.faiss"))
        print(f"Wrote rag_index.faiss ({index.ntotal} vectors, dim={dim})")
    except ImportError:
        print("faiss not installed; wrote rag_vectors.npy fallback only "
              "(inference/rag.py rebuilds the index at init)")

    langs: dict[str, int] = {}
    for r in records:
        langs[r["lang"]] = 1 + langs.get(r["lang"], 0)
    print(f"Wrote rag_meta.json ({len(records)} records, dim={dim})")
    print("Dialog language distribution:", dict(sorted(langs.items(), key=lambda x: -x[1])))

    if args.no_bm25:
        print("--no-bm25 set; skipping BM25 indexes (dense-only).")
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank-bm25 not installed; skipping BM25 indexes (pip install rank-bm25).")
        return

    import pickle

    dialog_tokens = [_tokenize(r["query"]) for r in records]
    bm25_dialog = BM25Okapi(dialog_tokens)
    with (args.out_dir / "bm25_dialog.pkl").open("wb") as f:
        pickle.dump(bm25_dialog, f)
    print(f"Wrote bm25_dialog.pkl ({len(dialog_tokens)} docs)")

    books_records: list[dict] = []
    if args.no_books:
        print("--no-books set; skipping book ingest.")
    elif args.books_dir and Path(args.books_dir).is_dir():
        print(f"Chunking books under {args.books_dir} ...")
        books_records = _build_books(
            Path(args.books_dir), args.books_chunk_size, args.books_overlap, args.max_book_chunks
        )
    else:
        print(f"books-dir {args.books_dir} not found; skipping books (dense + dialog BM25 only).")

    if books_records:
        books_tokens = [_tokenize(b["text"]) for b in books_records]
        bm25_books = BM25Okapi(books_tokens)
        with (args.out_dir / "bm25_books.pkl").open("wb") as f:
            pickle.dump(bm25_books, f)
        blangs: dict[str, int] = {}
        for b in books_records:
            blangs[b["lang"]] = 1 + blangs.get(b["lang"], 0)
        print(f"Wrote bm25_books.pkl ({len(books_records)} chunks) langs={blangs}")
    else:
        # Remove any stale books index so init won't mismatch against empty meta.
        stale = args.out_dir / "bm25_books.pkl"
        if stale.is_file():
            stale.unlink()
        print("No book chunks built; bm25_books.pkl not written.")

    bm25_meta = {
        "version": 1,
        "tokenizer": "unicode_word",
        "dialog_count": len(records),
        "books": books_records,
    }
    with (args.out_dir / "bm25_meta.json").open("w", encoding="utf-8") as f:
        json.dump(bm25_meta, f, ensure_ascii=False)
    print(f"Wrote bm25_meta.json (dialog_count={len(records)}, books={len(books_records)})")


if __name__ == "__main__":
    main()
