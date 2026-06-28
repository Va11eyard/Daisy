"""
Build the Daisy RAG retrieval index from the synthesized dialog corpus.

Walks data/synthesized/*/batch_*.jsonl, extracts (user -> assistant) pairs with
their language / tone / phase metadata, embeds the USER turn with a multilingual
sentence-transformer, and writes three artifacts into inference/knowledge/:

  rag_meta.json    - records aligned row-for-row with the vectors (+ build config)
  rag_vectors.npy  - float32, L2-normalized embedding matrix (faiss-free fallback)
  rag_index.faiss  - flat inner-product index (written only when faiss is available)

Retrieval embeds the incoming user message and returns the top-k *assistant*
replies as tone / vocabulary grounding (see inference/rag.py).

  python scripts/build_rag_index.py
  python scripts/build_rag_index.py --model intfloat/multilingual-e5-base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE = _ROOT / "inference"
for p in (_ROOT, _INFERENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from state_detector import detect_state  # noqa: E402

_VALID_PHASES = ("intake", "disclosure", "psychoeducation", "action_planning", "crisis")
_DEFAULT_MODEL = os.environ.get(
    "DAISY_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Daisy RAG index from the dialog corpus")
    ap.add_argument("--corpus", type=Path, default=_ROOT / "data" / "synthesized")
    ap.add_argument("--out-dir", type=Path, default=_INFERENCE / "knowledge")
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
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
    print("Language distribution:", dict(sorted(langs.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
