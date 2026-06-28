"""Layer 4 - token confidence gate.

Computes the mean log-probability of the generated tokens. When it falls below a
threshold (default -2.5), the reply is treated as low-confidence garbage and the
pipeline skips straight to a fallback instead of regenerating repeatedly.

Kept dependency-light: mean_logprob() accepts any iterable of per-token log
probabilities (Python list, numpy array, or torch tensor), so the model/torch
machinery stays in generation.py.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = -2.5


def confidence_gate_enabled() -> bool:
    return os.environ.get("DAISY_CONFIDENCE_GATE", "true").lower() in ("1", "true", "yes")


def confidence_threshold() -> float:
    try:
        return float(os.environ.get("DAISY_CONFIDENCE_THRESHOLD", str(DEFAULT_THRESHOLD)))
    except ValueError:
        return DEFAULT_THRESHOLD


def mean_logprob(token_logprobs: Iterable[Any] | None) -> float | None:
    """Mean of finite per-token log-probabilities. None when nothing usable."""
    if token_logprobs is None:
        return None
    values: list[float] = []
    for v in token_logprobs:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            values.append(f)
    if not values:
        return None
    return sum(values) / len(values)


def passes_confidence_gate(score: float | None, *, threshold: float | None = None) -> bool:
    """True when the reply clears the gate (or when the score is unavailable)."""
    if not confidence_gate_enabled():
        return True
    if score is None:
        # No score (e.g. streaming without logprobs) -> do not block.
        return True
    thr = threshold if threshold is not None else confidence_threshold()
    ok = score >= thr
    if not ok:
        logger.warning("Confidence gate failed: mean_logprob=%.3f < %.3f", score, thr)
    return ok
