"""Layer 2 - Hybrid RAG context injection.

Replaces book_knowledge.py. Combines dense (FAISS) retrieval over the dialog
corpus with BM25 sparse retrieval over both the dialog user-turns and a separate
books corpus, fused with Reciprocal Rank Fusion (RRF). Returns top-k passages as
tone / vocabulary / terminology grounding.

Streams fused per query:
  - dense-dialog : FAISS cosine over dialog user-turn embeddings (tone/style)
  - bm25-dialog  : BM25Okapi over dialog user-turn tokens (lexical tone match)
  - bm25-books   : BM25Okapi over clinical book chunks (exact terminology)

Artifacts (under inference/knowledge/, override with DAISY_RAG_DIR):
  rag_meta.json    - dialog records (assistant text + lang/phase/tone) + build config
  rag_vectors.npy  - L2-normalized dialog embedding matrix (faiss-free fallback)
  rag_index.faiss  - optional flat IP index; rebuilt from vectors if absent
  bm25_dialog.pkl  - BM25Okapi over dialog user turns (parallel to rag_meta records)
  bm25_books.pkl   - BM25Okapi over book chunks (parallel to bm25_meta books)
  bm25_meta.json   - {dialog_count, books:[{text, lang, source}]}

Graceful degradation: full hybrid -> dense-only (BM25 artifacts/dep missing or
DAISY_BM25=false) -> [] (no stream available or empty query).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(os.environ.get("DAISY_RAG_DIR", str(Path(__file__).resolve().parent / "knowledge")))
_META_PATH = _KNOWLEDGE_DIR / "rag_meta.json"
_VECTORS_PATH = _KNOWLEDGE_DIR / "rag_vectors.npy"
_FAISS_PATH = _KNOWLEDGE_DIR / "rag_index.faiss"
_BM25_DIALOG_PATH = _KNOWLEDGE_DIR / "bm25_dialog.pkl"
_BM25_BOOKS_PATH = _KNOWLEDGE_DIR / "bm25_books.pkl"
_BM25_META_PATH = _KNOWLEDGE_DIR / "bm25_meta.json"

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Dense state (prewarmed in init_rag_index()).
_embedder: Any = None
_index: Any = None          # faiss index when available
_vectors: Any = None        # numpy fallback matrix
_records: list[dict[str, Any]] = []
_ready: bool = False
_model_id: str = _DEFAULT_MODEL

# BM25 state (prewarmed in init_bm25_index()).
_bm25_dialog: Any = None
_bm25_books: Any = None
_books_records: list[dict[str, Any]] = []
_bm25_dialog_ready: bool = False
_bm25_books_ready: bool = False

# Languages we will fall back to, in order, when the reply language has no corpus.
_LANG_FALLBACK: dict[str, tuple[str, ...]] = {
    "en": ("en",),
    "ru": ("ru", "en"),
    "kk": ("kk", "ru", "en"),  # no Kazakh corpus today -> lean on ru/en grounding
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def rag_enabled() -> bool:
    return os.environ.get("DAISY_RAG", "true").lower() in ("1", "true", "yes")


def bm25_enabled() -> bool:
    return os.environ.get("DAISY_BM25", "true").lower() in ("1", "true", "yes")


def rag_ready() -> bool:
    return _ready or _bm25_dialog_ready or _bm25_books_ready


def bm25_ready() -> bool:
    return _bm25_dialog_ready or _bm25_books_ready


def _min_score() -> float:
    try:
        return float(os.environ.get("DAISY_RAG_MIN_SCORE", "0.30"))
    except ValueError:
        return 0.30


def _rrf_k() -> int:
    try:
        return max(1, int(os.environ.get("DAISY_RRF_K", "60")))
    except ValueError:
        return 60


def _books_weight() -> float:
    try:
        return float(os.environ.get("DAISY_BOOKS_WEIGHT", "1.0"))
    except ValueError:
        return 1.0


def _tokenize(text: str) -> list[str]:
    """Unicode word tokenizer (Cyrillic-safe). Shared with the build script so
    corpus and query tokenization can never drift."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _load_embedder(model_id: str) -> Any:
    global _embedder
    if _embedder is not None:
        return _embedder
    from sentence_transformers import SentenceTransformer

    logger.info("Loading RAG embedder %s", model_id)
    _embedder = SentenceTransformer(model_id)
    return _embedder


def _init_dense_index() -> bool:
    """Load the dense dialog index + embedder. Returns True when ready."""
    global _index, _vectors, _records, _ready, _model_id
    if _ready:
        return True
    if not rag_enabled():
        logger.info("RAG disabled via DAISY_RAG")
        return False
    if not _META_PATH.is_file():
        logger.warning("RAG meta not found at %s; dense retrieval disabled", _META_PATH)
        return False

    try:
        with _META_PATH.open(encoding="utf-8") as f:
            meta = json.load(f)
        _records = meta.get("records") or []
        _model_id = meta.get("model") or _DEFAULT_MODEL
        if not _records:
            logger.warning("RAG meta has no records; dense retrieval disabled")
            return False

        import numpy as np

        if _VECTORS_PATH.is_file():
            _vectors = np.load(_VECTORS_PATH).astype("float32")
        else:
            logger.warning("RAG vectors not found at %s; dense retrieval disabled", _VECTORS_PATH)
            return False

        try:
            import faiss

            if _FAISS_PATH.is_file():
                _index = faiss.read_index(str(_FAISS_PATH))
            else:
                _index = faiss.IndexFlatIP(int(_vectors.shape[1]))
                _index.add(_vectors)
        except ImportError:
            _index = None  # numpy brute-force fallback in _search()

        _load_embedder(_model_id)
        _ready = True
        logger.info("Dense RAG ready: %s records, faiss=%s", len(_records), _index is not None)
        return True
    except Exception:
        logger.exception("Dense RAG init failed; continuing without dense retrieval")
        return False


def init_bm25_index() -> bool:
    """Load BM25 dialog + books indexes. Fully guarded; never raises."""
    global _bm25_dialog, _bm25_books, _books_records
    global _bm25_dialog_ready, _bm25_books_ready
    if not bm25_enabled():
        logger.info("BM25 disabled via DAISY_BM25")
        return False
    try:
        import pickle

        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 disabled (dense-only)")
            return False

        books_records: list[dict[str, Any]] = []
        if _BM25_META_PATH.is_file():
            with _BM25_META_PATH.open(encoding="utf-8") as f:
                bm_meta = json.load(f)
            books_records = bm_meta.get("books") or []

        if _BM25_DIALOG_PATH.is_file():
            with _BM25_DIALOG_PATH.open("rb") as f:
                _bm25_dialog = pickle.load(f)
            size = getattr(_bm25_dialog, "corpus_size", None)
            if _records and size == len(_records):
                _bm25_dialog_ready = True
            else:
                logger.warning(
                    "bm25_dialog size %s != dialog records %s; disabling bm25-dialog",
                    size, len(_records),
                )
                _bm25_dialog = None

        if _BM25_BOOKS_PATH.is_file() and books_records:
            with _BM25_BOOKS_PATH.open("rb") as f:
                _bm25_books = pickle.load(f)
            size = getattr(_bm25_books, "corpus_size", None)
            if size == len(books_records):
                _books_records = books_records
                _bm25_books_ready = True
            else:
                logger.warning(
                    "bm25_books size %s != books records %s; disabling bm25-books",
                    size, len(books_records),
                )
                _bm25_books = None

        logger.info("BM25 ready: dialog=%s books=%s", _bm25_dialog_ready, _bm25_books_ready)
        return _bm25_dialog_ready or _bm25_books_ready
    except Exception:
        logger.exception("BM25 init failed; continuing dense-only")
        return False


def init_rag_index() -> bool:
    """Warm dense + BM25 indexes at container startup. Returns dense readiness."""
    dense_ok = _init_dense_index()
    init_bm25_index()
    return dense_ok


def _search(query_vec: Any, k: int) -> list[tuple[float, int]]:
    """Return (score, row) pairs for the k nearest dialog entries, sorted desc."""
    import numpy as np

    if _index is not None:
        scores, idxs = _index.search(query_vec.reshape(1, -1), k)
        return [(float(s), int(i)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
    sims = (_vectors @ query_vec.reshape(-1, 1)).ravel()
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i)) for i in top]


def _dense_ranked(query_vec: Any, n: int) -> list[tuple[int, int, float]]:
    """(row, rank, cosine) for the top-n dense dialog matches."""
    return [(row, rank, score) for rank, (score, row) in enumerate(_search(query_vec, n), start=1)]


def _bm25_ranked(bm25: Any, tokens: list[str], n: int) -> list[tuple[int, int]]:
    """(row, rank) for the top-n BM25 matches with a positive score."""
    import numpy as np

    scores = bm25.get_scores(tokens)
    if len(scores) == 0:
        return []
    order = [int(r) for r in np.argsort(scores)[::-1][:n] if scores[r] > 0]
    return [(row, rank) for rank, row in enumerate(order, start=1)]


def retrieve(
    query: str,
    *,
    reply_lang: str,
    phase: str,
    top_k: int = 3,
    history_snippet: str = "",
) -> list[str]:
    """Top-k grounding passages via RRF over dense + BM25 streams.

    Prefers records matching reply_lang (with fallback chain) and the conversation
    phase. Dialog passages provide tone/style; book passages provide terminology.
    """
    if not query.strip():
        return []
    if not (_ready or _bm25_dialog_ready or _bm25_books_ready):
        return []

    text = query.strip()
    if history_snippet.strip():
        text = f"{history_snippet.strip()} {text}"

    k = _rrf_k()
    w_books = _books_weight()
    langs = _LANG_FALLBACK.get(reply_lang, (reply_lang, "en"))
    pool_n = max(top_k * 8, 20)

    # key -> {rrf, text, lang, phase, cosine, in_bm25}
    fused: dict[tuple[str, int], dict[str, Any]] = {}

    def _slot(key: tuple[str, int], rec: dict[str, Any], has_phase: bool) -> dict[str, Any]:
        e = fused.get(key)
        if e is None:
            e = fused[key] = {
                "rrf": 0.0,
                "text": rec.get("text", ""),
                "lang": rec.get("lang", "en"),
                "phase": rec.get("phase") if has_phase else None,
                "cosine": None,
                "in_bm25": False,
            }
        return e

    # Stream 1: dense dialog.
    if _ready:
        try:
            qvec = _load_embedder(_model_id).encode(
                text, normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")
            for row, rank, cosine in _dense_ranked(qvec, pool_n):
                if row >= len(_records):
                    continue
                e = _slot(("d", row), _records[row], has_phase=True)
                e["rrf"] += 1.0 / (k + rank)
                e["cosine"] = cosine if e["cosine"] is None else max(e["cosine"], cosine)
        except Exception:
            logger.exception("Dense retrieval failed; using BM25 streams only")

    tokens = _tokenize(text)

    # Stream 2: BM25 dialog.
    if _bm25_dialog_ready and tokens:
        for row, rank in _bm25_ranked(_bm25_dialog, tokens, pool_n):
            if row >= len(_records):
                continue
            e = _slot(("d", row), _records[row], has_phase=True)
            e["rrf"] += 1.0 / (k + rank)
            e["in_bm25"] = True

    # Stream 3: BM25 books (terminology grounding).
    if _bm25_books_ready and tokens:
        for row, rank in _bm25_ranked(_bm25_books, tokens, pool_n):
            if row >= len(_books_records):
                continue
            e = _slot(("b", row), _books_records[row], has_phase=False)
            e["rrf"] += w_books * (1.0 / (k + rank))
            e["in_bm25"] = True

    min_score = _min_score()
    phase_bonus = 1.0 / k
    lang_penalty = 0.5 / k

    scored: list[tuple[float, str]] = []
    for e in fused.values():
        if e["lang"] not in langs:
            continue
        # min_score gate: drop weak dense-only matches; BM25-supported ones bypass
        # (RRF scores are not comparable to cosine, so the gate stays on the dense side).
        if not e["in_bm25"] and (e["cosine"] is None or e["cosine"] < min_score):
            continue
        final = e["rrf"]
        if e["phase"] is not None and e["phase"] == phase:
            final += phase_bonus
        final -= langs.index(e["lang"]) * lang_penalty
        scored.append((final, e["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _f, txt in scored:
        t = (txt or "").strip()
        key = t.lower()[:80]
        if t and key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) >= top_k:
            break
    return out


def format_rag_block(passages: list[str]) -> str:
    """Render retrieved exemplars as a system-prompt grounding block."""
    if not passages:
        return ""
    lines = [
        "[RETRIEVED CONTEXT] Reference replies from past sessions for tone and vocabulary only.",
        "Apply their register and word choices in your own words; never quote, cite, or copy them verbatim.",
        *[f"- {p}" for p in passages],
    ]
    return "\n".join(lines)
