"""
Compact «user image» for the therapist model — produced by the coordinator (tiny LM + DB)
or sent directly from the Daisy web app.

Recommended coordinator checkpoint (separate deployment / COORDINATOR_URL):
  Qwen/Qwen2.5-3B-Instruct
  — same family as the main Daisy model, strong multilingual + JSON, reasonable VRAM.

Schema version 1 aligns with Daisy web: ai_profile + psych_profile + preferences highlights.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

USER_IMAGE_SCHEMA_VERSION = "1"

# For HTTP coordinator services or documentation (not loaded by score.py by default).
RECOMMENDED_COORDINATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"

RISK_LEVELS = frozenset({"critical", "high", "medium", "low"})


def normalize_user_image(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Clamp lengths and enums; returns None if empty / invalid."""
    if not raw or not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}

    v = raw.get("version") or raw.get("schema_version")
    if v is not None:
        out["version"] = str(v)[:16]

    s = raw.get("summary")
    if isinstance(s, str) and s.strip():
        out["summary"] = s.strip()[:1200]

    for key, max_n, max_len in (
        ("goals", 6, 200),
        ("concerns", 6, 200),
        ("memory_highlights", 10, 280),
    ):
        arr = raw.get(key)
        if isinstance(arr, list):
            cleaned = []
            for x in arr[:max_n]:
                if isinstance(x, str) and x.strip():
                    cleaned.append(x.strip()[:max_len])
            if cleaned:
                out[key] = cleaned

    cs = raw.get("communication_style")
    if isinstance(cs, str) and cs.strip():
        out["communication_style"] = cs.strip()[:120]
    elif isinstance(cs, list):
        parts = [str(x).strip() for x in cs[:4] if str(x).strip()]
        if parts:
            out["communication_style"] = parts

    loc = raw.get("locale_hint") or raw.get("locale")
    if isinstance(loc, str) and loc.strip():
        out["locale_hint"] = loc.strip().lower()[:8]

    rl = raw.get("risk_level") or raw.get("riskLevel")
    if isinstance(rl, str) and rl.strip().lower() in RISK_LEVELS:
        out["risk_level"] = rl.strip().lower()

    idx = raw.get("indices")
    if isinstance(idx, dict):
        slim: dict[str, float] = {}
        for k in ("ESI", "BSI", "SSI", "PVI", "MRI"):
            if k in idx and isinstance(idx[k], (int, float)):
                slim[k] = float(idx[k])
        if slim:
            out["indices"] = slim

    b = raw.get("boundaries")
    if isinstance(b, dict):
        avoid = b.get("avoid_topics")
        sens = b.get("sensitive")
        bd: dict[str, list[str]] = {}
        if isinstance(avoid, list):
            bd["avoid_topics"] = [str(x).strip()[:120] for x in avoid[:12] if str(x).strip()]
        if isinstance(sens, list):
            bd["sensitive"] = [str(x).strip()[:120] for x in sens[:12] if str(x).strip()]
        if bd:
            out["boundaries"] = bd

    ph = raw.get("protocol_hint") or raw.get("protocol_directive")
    if isinstance(ph, str) and ph.strip():
        out["protocol_hint"] = ph.strip()[:400]

    if len(out) <= (1 if "version" in out else 0):
        return None
    out.setdefault("version", USER_IMAGE_SCHEMA_VERSION)
    return out


def format_user_image_for_prompt(
    img: dict[str, Any],
    *,
    force_english: bool,
    locale: str | None,
) -> str:
    """Single system-prompt block; language for labels follows force_english / locale."""
    ru = not force_english and (locale or "").lower().startswith("ru")
    kk = not force_english and (locale or "").lower().startswith("kk")

    def L(en: str, r: str, k: str = "") -> str:
        if kk and k:
            return k
        if ru:
            return r
        return en

    lines: list[str] = []
    title = L(
        "User image (coordinator summary — use to personalize tone and focus, not as facts to quote verbatim):",
        "Имидж пользователя (сводка координатора — для персонализации тона и фокуса, не цитируй дословно как факты):",
        "Пайдаланушы бейнесі (координатор қорытындысы — тон мен фокусты бейімдеу үшін):",
    )
    lines.append(title)

    if img.get("summary"):
        lines.append(L("Summary: ", "Кратко: ", "Қысқаша: ") + str(img["summary"]))

    if img.get("goals"):
        prefix = L("Goals: ", "Цели: ", "Мақсаттар: ")
        lines.append(prefix + "; ".join(img["goals"]))

    if img.get("concerns"):
        prefix = L("Concerns: ", "Тревоги/темы: ", "Мазмұны: ")
        lines.append(prefix + "; ".join(img["concerns"]))

    if img.get("communication_style"):
        cs = img["communication_style"]
        prefix = L("Preferred style: ", "Стиль общения: ", "Стиль: ")
        if isinstance(cs, list):
            lines.append(prefix + ", ".join(cs))
        else:
            lines.append(prefix + str(cs))

    if img.get("risk_level"):
        lines.append(L("Risk level: ", "Уровень риска: ", "Тәуекел: ") + str(img["risk_level"]))

    if img.get("indices"):
        parts = [f"{k}={v:.1f}" for k, v in sorted(img["indices"].items())]
        if parts:
            lines.append(L("Indices: ", "Индексы: ", "Индекстер: ") + ", ".join(parts))

    if img.get("memory_highlights"):
        prefix = L("Highlights: ", "Важное из памяти: ", "Есте сақтау: ")
        lines.append(prefix + " | ".join(img["memory_highlights"]))

    if img.get("boundaries"):
        b = img["boundaries"]
        if b.get("avoid_topics"):
            lines.append(
                L("Avoid topics: ", "Избегать тем: ", "Темалардан аулақ болу: ")
                + "; ".join(b["avoid_topics"])
            )
        if b.get("sensitive"):
            lines.append(
                L("Sensitive: ", "Деликатно: ", "Сезімтал: ") + "; ".join(b["sensitive"])
            )

    if img.get("protocol_hint"):
        lines.append(L("Protocol hint: ", "Протокол: ", "Протокол: ") + str(img["protocol_hint"]))

    return "\n".join(lines)


def parse_user_image_field(raw: Any) -> dict[str, Any] | None:
    """Accept JSON object or JSON string from request body."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return normalize_user_image(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return normalize_user_image(parsed)
        except json.JSONDecodeError:
            logger.info("user_image: invalid JSON string")
    return None
