"""Persona keys aligned with Daisy frontend (communication styles)."""

from __future__ import annotations

import json
from typing import Any

PERSONA_MAP: dict[str, dict[str, str]] = {
    "warm_friend": {
        "instruction": (
            "You are a warm, soulful friend. Be gentle, understanding, and emotionally supportive. "
            "Listen actively, validate feelings deeply."
        ),
        "example": "«Я вижу, как тебе сейчас непросто. Давай вместе разберёмся.»",
    },
    "practical_helper": {
        "instruction": (
            "You are a practical helper. Be specific, structured, and motivating. Focus on concrete steps."
        ),
        "example": "«Что конкретно ты можешь сделать сегодня?»",
    },
    "gentle_explorer": {
        "instruction": (
            "You are a curious, reflective explorer. Ask open-ended questions about feelings and meanings."
        ),
        "example": "«Откуда, как ты думаешь, может идти это чувство?»",
    },
    "soft_explorer": {
        "instruction": (
            "You are a curious, reflective explorer. Ask open-ended questions about feelings and meanings."
        ),
        "example": "«Откуда, как ты думаешь, может идти это чувство?»",
    },
    "calm_mentor": {
        "instruction": (
            "You are a calm mentor. Be balanced, accepting, patient. Give space for the person's conclusions."
        ),
        "example": "«Всё, что ты чувствуешь, имеет право быть.»",
    },
    "wise_teacher": {
        "instruction": (
            "You are an educational guide. Explain concepts clearly; name cognitive distortions when helpful."
        ),
        "example": "«Это похоже на когнитивное искажение — вот как оно работает…»",
    },
    "flexible": {
        "instruction": (
            "Adapt your style to what the person needs: sometimes warm, sometimes practical, sometimes exploratory."
        ),
        "example": "«Подстраивайся под текущую потребность человека.»",
    },
}

_PERSONA_ALIASES: dict[str, str] = {
    "тёплая подруга": "warm_friend",
    "теплая подруга": "warm_friend",
    "active_listener": "warm_friend",
    "практичный помощник": "practical_helper",
    "behavior_coach": "practical_helper",
    "мягкий исследователь": "gentle_explorer",
    "questioner": "gentle_explorer",
    "спокойный наставник": "calm_mentor",
    "emotion_control_provider": "calm_mentor",
    "мудрый учитель": "wise_teacher",
    "psychoeducator": "wise_teacher",
    "гибкая собеседница": "flexible",
    "flexible_companion": "flexible",
    "all_dynamic": "flexible",
}

# Canonical keys returned to site (CbtConversation.persona)
_DAISY_PERSONA_KEYS: dict[str, str] = {
    "soft_explorer": "gentle_explorer",
    "flexible": "flexible_companion",
}


def _canonical_one(token: str) -> str:
    raw_l = token.strip().lower()
    if raw_l in PERSONA_MAP:
        key = raw_l
    elif raw_l in _PERSONA_ALIASES:
        key = _PERSONA_ALIASES[raw_l]
    else:
        key = "flexible"
    return _DAISY_PERSONA_KEYS.get(key, key)


def get_canonical_persona_key(raw: str | None) -> str:
    """Normalize to site keys; supports up to two comma-separated styles (same as website)."""
    if not raw:
        return "flexible_companion"
    raw_l = raw.strip().lower()
    parts = [s.strip() for s in raw_l.replace("+", ",").split(",") if s.strip()]
    if not parts:
        return "flexible_companion"
    if len(parts) == 1:
        return _canonical_one(parts[0])
    return ",".join(_canonical_one(p) for p in parts[:2])


def _extract_communication_styles_from_payload(data: dict[str, Any]) -> list[str]:
    """Reads onboarding_summary.ai_profile.communication_style (1–2 entries) per Daisy web contract."""
    ob = data.get("onboarding_summary")
    if isinstance(ob, dict):
        ap = ob.get("ai_profile") or {}
        cs = ap.get("communication_style")
        if isinstance(cs, list):
            return [str(x).strip() for x in cs if x]
    if isinstance(ob, str):
        s = ob.strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
                if isinstance(d, dict):
                    ap = d.get("ai_profile") or {}
                    cs = ap.get("communication_style")
                    if isinstance(cs, list):
                        return [str(x).strip() for x in cs if x]
            except json.JSONDecodeError:
                pass
    return []


def effective_persona_for_prompt(data: dict[str, Any]) -> str:
    """
    Style(s) for resolve_persona / system prompt.

    Prefer `onboarding_summary.ai_profile.communication_style` (1–2 styles from the site).
    Otherwise use `persona` or `protocol` (may already be \"warm_friend,wise_teacher\").
    """
    styles = _extract_communication_styles_from_payload(data)
    if styles:
        seen: list[str] = []
        for s in styles:
            if s and s not in seen:
                seen.append(s)
            if len(seen) >= 2:
                break
        return ",".join(seen)
    return (data.get("persona") or data.get("protocol") or "").strip()


def resolve_persona(raw: str | None) -> tuple[str, str]:
    if not raw:
        p = PERSONA_MAP["flexible"]
        return p["instruction"], p["example"]
    raw_l = raw.strip().lower()
    if raw_l in PERSONA_MAP:
        p = PERSONA_MAP[raw_l]
        return p["instruction"], p["example"]
    if raw_l in _PERSONA_ALIASES:
        p = PERSONA_MAP[_PERSONA_ALIASES[raw_l]]
        return p["instruction"], p["example"]
    parts = [s.strip() for s in raw_l.replace("+", ",").split(",") if s.strip()]
    instrs: list[str] = []
    ex: str = ""
    for part in parts:
        k = _PERSONA_ALIASES.get(part, part)
        if k in PERSONA_MAP:
            instrs.append(PERSONA_MAP[k]["instruction"])
            ex = PERSONA_MAP[k]["example"]
    if instrs:
        return " ".join(instrs), ex
    return f"Match this communication style: {raw}.", ""
