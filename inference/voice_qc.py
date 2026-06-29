"""Post-generation voice quality checks and regen hints."""

from __future__ import annotations

import re

from state_detector import DaisyState
from voice_contract import BANNED_PHRASES, FEW_SHOT_PAIRS, STRUCTURAL_RULES

_QC_STATES: frozenset[str] = frozenset({"intake", "disclosure", "action_planning"})

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
    # Detached psychoeducation one-liners (no "you", no question).
    re.compile(r"^(?:anxiety|worry|stress|grief|depression)\s+(?:can|often|tends|may)\b", re.I),
    re.compile(r"\bcan settle in quietly\b", re.I),
    re.compile(r"\bcreeping in when we\b", re.I),
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

    low = t.lower()
    for pat in _GENERIC_PATTERNS:
        if pat.search(t):
            return True

    for phrase in BANNED_PHRASES:
        if phrase.lower() in low:
            return True

    rules = STRUCTURAL_RULES.get(state)  # type: ignore[arg-type]
    if rules:
        min_s = rules.get("min_sentences")
        if min_s is not None and _sentence_count(t) < min_s:
            return True

    if "?" not in t:
        return True

    return False


def voice_regen_suffix(state: DaisyState, reply_lang: str) -> str:
    pair = FEW_SHOT_PAIRS.get(state) or FEW_SHOT_PAIRS.get("intake")
    good = pair["good"] if pair else ""
    if reply_lang == "ru":
        return (
            "\n\nОтвет слишком общий или шаблонный. НЕ начинай с «мне жаль слышать» или «это должно быть тяжело». "
            f"Отрази услышанное в 2–3 предложениях и задай один открытый вопрос. Пример тона:\n{good}"
        )
    if reply_lang == "kk":
        return (
            "\n\nЖауап тым жалпы. «I'm sorry to hear that» деп бастама. "
            f"2–3 сөйлеммен көрсетіп, бір ашық сұрақ қой. Мысал:\n{good}"
        )
    return (
        "\n\nReply is too generic. Do NOT start with \"I'm sorry to hear that\" or \"It sounds like you're dealing with\". "
        f"Reflect in 2–3 sentences, then one open question. Example tone:\n{good}"
    )
