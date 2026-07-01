"""Post-generation voice quality checks and regen hints."""

from __future__ import annotations

import re

from state_detector import DaisyState
from voice_contract import BANNED_PHRASES

_QC_STATES: frozenset[str] = frozenset(
    {"intake", "disclosure", "psychoeducation", "action_planning"}
)

_GENERIC_PATTERNS_EN: tuple[re.Pattern[str], ...] = (
    re.compile(r"^i'?m sorry to hear that\b", re.I),
    re.compile(r"\bi'?m (?:so )?sorry to hear\b", re.I),
    re.compile(r"\bi'?m sorry you'?re feeling\b", re.I),
    re.compile(r"\bit sounds like you'?re (?:dealing with|going through)\b", re.I),
    re.compile(r"\bit sounds like you (?:might be|are) carrying\b", re.I),
    re.compile(r"\bthat must be (?:incredibly )?(?:stressful|tough|hard|really tough)\b", re.I),
    re.compile(r"\bthat must feel very\b", re.I),
    re.compile(r"\bi understand (?:that )?this (?:is|must be)\b", re.I),
    re.compile(r"\bi understand how (?:hard|difficult|tough)\b", re.I),
    re.compile(r"\bhow can i best help you (?:right )?now\b", re.I),
    re.compile(r"\bi'?ll try to (?:answer|respond) (?:in a )?warmer\b", re.I),
)

_GENERIC_PATTERNS_RU: tuple[re.Pattern[str], ...] = (
    re.compile(r"^я понимаю,?\s*(?:что|как)\s+это\b", re.I),
    re.compile(r"\bмне жаль (?:это )?слышать\b", re.I),
    re.compile(r"\bэто (?:должно быть|звучит как)\b", re.I),
    re.compile(r"\bкак тебе лучше (?:всего )?помочь сейчас\b", re.I),
    re.compile(r"\bпрости,?\s*я попробую ответить\b", re.I),
    re.compile(r"\bэто может быть (?:очень )?трудно\b", re.I),
)

_GENERIC_PATTERNS_KK: tuple[re.Pattern[str], ...] = (
    re.compile(r"^мен түсінемін\b", re.I),
    re.compile(r"\bөкінішті естіу\b", re.I),
    re.compile(r"\bбұл (?:қиын|ауыр) болуы керек\b", re.I),
    re.compile(r"\bқалай көмектесуімді қалайсың\b", re.I),
    re.compile(r"\bкешір,?\s*жылырақ жауап\b", re.I),
)

_GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_GENERIC_PATTERNS_EN,
    *_GENERIC_PATTERNS_RU,
    *_GENERIC_PATTERNS_KK,
)

_EXEMPLAR_ECHO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"builds quietly before you name it", re.I),
    re.compile(r"hum you stop noticing", re.I),
    re.compile(r"stop noticing until it gets loud", re.I),
)

_THERAPIST_ROLE_BREAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bbracing myself\b", re.I),
    re.compile(r"\bhere'?s one concrete thing we can work on\b", re.I),
    re.compile(r"\bhere'?s one concrete way to\b", re.I),
    re.compile(r"\bhere'?s one way to phrase\b", re.I),
    re.compile(r"^in plain language:\s*", re.I),
    re.compile(r"\bone way to (?:think about|approach) this\b", re.I),
    re.compile(r"\bwhat the user wrote\b", re.I),
    re.compile(r"^I'?m feeling\b", re.I),
    re.compile(r"\bI feel anxious\b", re.I),
)

_META_QUALITY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btoo short\b", re.I),
    re.compile(r"\btoo plain\b", re.I),
    re.compile(r"\btoo generic\b", re.I),
    re.compile(r"\bnot helpful\b", re.I),
    re.compile(r"\banswer (?:longer|more)\b", re.I),
    re.compile(r"\b(?:you are|you're) answering too\b", re.I),
    re.compile(r"\b(?:more|longer) (?:answers|responses)\b", re.I),
    re.compile(r"\bслишком коротк", re.I),
    re.compile(r"\bслишком общ", re.I),
)


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?…]+", text.strip())
    return len([p for p in parts if p.strip()])


def is_therapist_role_break(text: str) -> bool:
    """True when Daisy speaks as the client or jumps to homework on a support turn."""
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _THERAPIST_ROLE_BREAK_PATTERNS)


def is_exemplar_echo(text: str) -> bool:
    """True when reply copies calibration exemplar phrasing instead of original dialogue."""
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _EXEMPLAR_ECHO_PATTERNS)


def is_too_brief(text: str, state: str = "intake") -> bool:
    """True when a non-crisis therapy turn is a single sentence or lacks an open question."""
    if state == "crisis":
        return False
    if state not in _QC_STATES:
        return False
    t = (text or "").strip()
    if not t:
        return True
    if _sentence_count(t) < 2:
        return True
    if "?" not in t:
        return True
    return False


def is_meta_quality_complaint(user_message: str) -> bool:
    t = (user_message or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _META_QUALITY_RES)


def violates_voice_contract(
    text: str, state: str, *, reply_lang: str = "en", user_message: str = ""
) -> bool:
    """True when reply matches hollow/generic patterns or misses structural rules."""
    qc_active = state in _QC_STATES or is_meta_quality_complaint(user_message)
    if not qc_active:
        return False
    t = (text or "").strip()
    if not t:
        return True

    if is_too_brief(t, state):
        return True

    if is_exemplar_echo(t):
        return True

    opening = re.split(r"[.!?…]+", t, maxsplit=1)[0].strip() or t[:120]
    opening_low = opening.lower()
    brief = is_too_brief(t, state)
    for pat in _GENERIC_PATTERNS:
        if pat.search(opening) and (brief or is_exemplar_echo(t)):
            return True

    for phrase in BANNED_PHRASES:
        if phrase.lower() in opening_low and brief:
            return True

    if "?" not in t:
        return True

    return False


def voice_regen_suffix(state: DaisyState, reply_lang: str) -> str:
    _ = state
    if reply_lang == "ru":
        return (
            "\n\nПоследний ответ был слишком коротким или только пересказал сказанное. "
            "Никогда не заканчивай без одного открытого вопроса об их переживании. "
            "3–5 предложений: короткое отражение → одно наблюдение или мягкий рефрейм → ОДИН открытый вопрос. "
            "НЕ начинай с «мне жаль слышать» или «это должно быть тяжело»."
        )
    if reply_lang == "kk":
        return (
            "\n\nСоңғы жауап тым қысқа немесе тек айтылғанды қайталады. "
            "Бір ашық сұрақсыз аяқтама. "
            "3–5 сөйлем: қысқа көрсету → бір бақылау немесе жұмсақ рефрейм → БІР ашық сұрақ. "
            "«I'm sorry to hear that» деп бастама."
        )
    return (
        "\n\nYour last reply was too brief or only restated what they said. "
        "Never end without one open question about their experience. "
        "Use 3–5 sentences: brief reflection → one specific observation or gentle reframe → ONE open question. "
        "Do NOT start with \"I'm sorry to hear that\" or \"It sounds like you're dealing with\"."
    )
