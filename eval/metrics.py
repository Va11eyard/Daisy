"""Genericness metrics for ablation evaluation."""

from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Any

# Known canned lines (must stay in sync with inference/generation.py pools)
_CANNED_EN = (
    "What feels most important to talk about right now?",
    "Where would you like to start?",
    "What's weighing on you most in this?",
    "Which part of this would help to dig into first?",
    "How are you holding up with all of it?",
    "What do you need most right now",
    "Hey — I'm glad you're here. What's been on your mind lately?",
    "I'm here. What feels strongest right now",
    "Breakups can leave everything feeling unsteady",
    "It sounds like dealing with an angry person",
    "It sounds like things have been really overwhelming",
)

_CANNED_RU = (
    "Что сейчас важнее всего обсудить?",
    "С чего бы тебе хотелось начать?",
    "Я рядом.",
    "Слышу, что тебе сейчас непросто",
)

_GENERIC_PATTERNS = (
    re.compile(r"\bit sounds like\b", re.I),
    re.compile(r"\bthat must be\b", re.I),
    re.compile(r"\bi'?m sorry to hear\b", re.I),
    re.compile(r"\bwhat feels most important to talk about right now\b", re.I),
    re.compile(r"\bhow can i best help you\b", re.I),
    re.compile(r"\bэто должно быть\b", re.I),
    re.compile(r"\bмне жаль слышать\b", re.I),
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct-n over concatenated corpus (higher = more diverse)."""
    grams: set[tuple[str, ...]] = set()
    total = 0
    for t in texts:
        toks = _tokenize(t)
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(grams) / total


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cross_scenario_genericness(
    records: list[dict[str, Any]],
    embed_fn: Any,
) -> float:
    """Mean pairwise cosine between one response per distinct cluster (higher = more generic)."""
    by_cluster: dict[str, str] = {}
    for r in records:
        cluster = r.get("cluster") or r["id"]
        if cluster not in by_cluster:
            by_cluster[cluster] = r["response"]
    responses = list(by_cluster.values())
    if len(responses) < 2:
        return 0.0
    vecs = [embed_fn(t) for t in responses]
    sims: list[float] = []
    for i, j in combinations(range(len(vecs)), 2):
        sims.append(cosine_sim(vecs[i], vecs[j]))
    return sum(sims) / len(sims)


def mean_specificity(
    records: list[dict[str, Any]],
    embed_fn: Any,
) -> float:
    """Mean cosine(response, user_message) — higher = more grounded in user words."""
    sims: list[float] = []
    for r in records:
        resp = (r.get("response") or "").strip()
        msg = (r.get("message") or "").strip()
        if not resp or not msg:
            continue
        sims.append(cosine_sim(embed_fn(resp), embed_fn(msg)))
    return sum(sims) / len(sims) if sims else 0.0


def canned_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of responses matching known canned/generic patterns."""
    if not records:
        return 0.0
    hits = 0
    for r in records:
        resp = (r.get("response") or "").strip()
        low = resp.lower()
        locale = r.get("locale", "en")
        pool = _CANNED_RU if locale == "ru" else _CANNED_EN
        if any(c.lower() in low for c in pool):
            hits += 1
            continue
        if any(p.search(resp) for p in _GENERIC_PATTERNS):
            hits += 1
    return hits / len(records)


def templated_shape_rate(records: list[dict[str, Any]]) -> float:
    """Fraction with 1-2 sentences + exactly one trailing question."""
    if not records:
        return 0.0
    hits = 0
    for r in records:
        resp = (r.get("response") or "").strip()
        if "?" not in resp:
            continue
        parts = [p.strip() for p in re.split(r"[.!?…]+", resp) if p.strip()]
        q_parts = resp.count("?")
        if 1 <= len(parts) <= 3 and q_parts == 1 and resp.strip().endswith("?"):
            hits += 1
    return hits / len(records)


def summarize_config(
    config_name: str,
    records: list[dict[str, Any]],
    embed_fn: Any,
) -> dict[str, Any]:
    responses = [(r.get("response") or "").strip() for r in records]
    return {
        "config": config_name,
        "n": len(records),
        "cross_scenario_genericness": round(cross_scenario_genericness(records, embed_fn), 4),
        "distinct_2": round(distinct_n(responses, 2), 4),
        "canned_rate": round(canned_rate(records), 4),
        "mean_specificity": round(mean_specificity(records, embed_fn), 4),
        "templated_shape_rate": round(templated_shape_rate(records), 4),
        "mean_response_len": round(sum(len(r) for r in responses) / max(len(responses), 1), 1),
    }


def rank_configs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by cross_scenario_genericness descending (worst first)."""
    return sorted(summaries, key=lambda s: s["cross_scenario_genericness"], reverse=True)


def adjacent_deltas(
    ordered_names: list[str],
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Effect sizes between adjacent cumulative configs."""
    out: list[dict[str, Any]] = []
    for a, b in zip(ordered_names, ordered_names[1:]):
        sa, sb = summaries[a], summaries[b]
        out.append({
            "from": a,
            "to": b,
            "delta_genericness": round(sb["cross_scenario_genericness"] - sa["cross_scenario_genericness"], 4),
            "delta_canned_rate": round(sb["canned_rate"] - sa["canned_rate"], 4),
            "delta_specificity": round(sb["mean_specificity"] - sa["mean_specificity"], 4),
            "delta_distinct_2": round(sb["distinct_2"] - sa["distinct_2"], 4),
        })
    return out
