"""Layer 2 - RAG context injection.

Replaces book_knowledge.py. Loads the dialog-corpus index built by
scripts/build_rag_index.py, embeds the incoming user message (+ recent history),
and returns the top-k exemplary assistant replies as tone / vocabulary grounding.

Artifacts (under inference/knowledge/, override with DAISY_RAG_DIR):
  rag_meta.json    - records (assistant text + lang/phase/tone) and build config
  rag_vectors.npy  - L2-normalized embedding matrix (faiss-free fallback)
  rag_index.faiss  - optional flat IP index; rebuilt from vectors if absent

Everything degrades gracefully: if artifacts or dependencies are missing,
retrieve() returns [] and the pipeline proceeds without grounding.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(os.environ.get("DAISY_RAG_DIR", str(Path(__file__).resolve().parent / "knowledge")))
_META_PATH = _KNOWLEDGE_DIR / "rag_meta.json"
_VECTORS_PATH = _KNOWLEDGE_DIR / "rag_vectors.npy"
_FAISS_PATH = _KNOWLEDGE_DIR / "rag_index.faiss"

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Lazily initialized module state (prewarmed in init_rag_index()).
_embedder: Any = None
_index: Any = None          # faiss index when available
_vectors: Any = None        # numpy fallback matrix
_records: list[dict[str, Any]] = []
_ready: bool = False
_model_id: str = _DEFAULT_MODEL

# Languages we will fall back to, in order, when the reply language has no corpus.
_LANG_FALLBACK: dict[str, tuple[str, ...]] = {
    "en": ("en",),
    "ru": ("ru", "en"),
    "kk": ("kk", "ru", "en"),  # no Kazakh corpus today -> lean on ru/en grounding
}


def rag_enabled() -> bool:
    return os.environ.get("DAISY_RAG", "true").lower() in ("1", "true", "yes")


def rag_ready() -> bool:
    return _ready


def _min_score() -> float:
    try:
        return float(os.environ.get("DAISY_RAG_MIN_SCORE", "0.30"))
    except ValueError:
        return 0.30


def _load_embedder(model_id: str) -> Any:
    global _embedder
    if _embedder is not None:
        return _embedder
    from sentence_transformers import SentenceTransformer

    logger.info("Loading RAG embedder %s", model_id)
    _embedder = SentenceTransformer(model_id)
    return _embedder


def init_rag_index() -> bool:
    """Warm the embedder + index at container startup. Returns True when ready."""
    global _index, _vectors, _records, _ready, _model_id
    if _ready:
        return True
    if not rag_enabled():
        logger.info("RAG disabled via DAISY_RAG")
        return False
    if not _META_PATH.is_file():
        logger.warning("RAG meta not found at %s; retrieval disabled", _META_PATH)
        return False

    try:
        with _META_PATH.open(encoding="utf-8") as f:
            meta = json.load(f)
        _records = meta.get("records") or []
        _model_id = meta.get("model") or _DEFAULT_MODEL
        if not _records:
            logger.warning("RAG meta has no records; retrieval disabled")
            return False

        import numpy as np

        if _VECTORS_PATH.is_file():
            _vectors = np.load(_VECTORS_PATH).astype("float32")
        else:
            logger.warning("RAG vectors not found at %s; retrieval disabled", _VECTORS_PATH)
            return False

        try:
            import faiss

            if _FAISS_PATH.is_file():
                _index = faiss.read_index(str(_FAISS_PATH))
            else:
                _index = faiss.IndexFlatIP(int(_vectors.shape[1]))
                _index.add(_vectors)
        except ImportError:
            _index = None  # numpy brute-force fallback in retrieve()

        _load_embedder(_model_id)
        _ready = True
        logger.info("RAG ready: %s records, faiss=%s", len(_records), _index is not None)
        return True
    except Exception:
        logger.exception("RAG init failed; continuing without retrieval")
        return False


def _search(query_vec: Any, k: int) -> list[tuple[float, int]]:
    """Return (score, row) pairs for the k nearest corpus entries."""
    import numpy as np

    if _index is not None:
        scores, idxs = _index.search(query_vec.reshape(1, -1), k)
        return [(float(s), int(i)) for s, i in zip(scores[0], idxs[0]) if i >= 0]
    sims = (_vectors @ query_vec.reshape(-1, 1)).ravel()
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i)) for i in top]


def retrieve(
    query: str,
    *,
    reply_lang: str,
    phase: str,
    top_k: int = 3,
    history_snippet: str = "",
) -> list[str]:
    """Top-k exemplary assistant replies for tone/vocabulary grounding.

    Prefers records matching reply_lang (with fallback chain) and the conversation
    phase, but never hard-fails to empty when softer matches clear the score gate.
    """
    if not _ready or not query.strip():
        return []
    text = query.strip()
    if history_snippet.strip():
        text = f"{history_snippet.strip()} {text}"

    try:
        qvec = _load_embedder(_model_id).encode(text, normalize_embeddings=True, convert_to_numpy=True)
    except Exception:
        logger.exception("RAG query embedding failed")
        return []

    pool_scores = _search(qvec.astype("float32"), max(top_k * 8, 24))
    min_score = _min_score()
    langs = _LANG_FALLBACK.get(reply_lang, (reply_lang, "en"))

    scored: list[tuple[float, dict[str, Any]]] = []
    for score, row in pool_scores:
        if score < min_score or row >= len(_records):
            continue
        rec = _records[row]
        # Rank boost for language and phase agreement; lang mismatch outside the
        # fallback chain is dropped so we never inject an unrelated-language reply.
        rec_lang = rec.get("lang", "en")
        if rec_lang not in langs:
            continue
        lang_rank = langs.index(rec_lang)
        phase_bonus = 0.05 if rec.get("phase") == phase else 0.0
        adjusted = score + phase_bonus - (lang_rank * 0.02)
        scored.append((adjusted, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _adj, rec in scored:
        txt = (rec.get("text") or "").strip()
        key = txt.lower()[:80]
        if txt and key not in seen:
            seen.add(key)
            out.append(txt)
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
