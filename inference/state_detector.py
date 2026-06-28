"""State detector: classify the current conversation phase from recent messages.

Pure function (no I/O): deterministic phase classification used by Layer 1
(router.detect_phase) and the RAG retriever.
"""

from __future__ import annotations

from typing import Literal, TypedDict

DaisyState = Literal[
    "intake", "disclosure", "psychoeducation", "action_planning", "crisis"
]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


_CRISIS_PATTERNS: tuple[str, ...] = (
    "hurt myself",
    "end my life",
    "don't want to be here",
    "suicidal",
    "kill myself",
    "can't go on",
    "self-harm",
    "harm myself",
)

_PSYCHOED_PATTERNS: tuple[str, ...] = (
    "why do i",
    "what is",
    "what are",
    "how does",
    "is it normal",
    "explain",
    "help me understand",
    "what does it mean",
)

_ACTION_PATTERNS: tuple[str, ...] = (
    "what should i do",
    "how do i",
    "what can i do",
    "next steps",
    "how to handle",
    "what do i do",
    "advice",
    "suggest",
    "recommend",
)

_DISCLOSURE_CUES: tuple[str, ...] = (
    "i feel",
    "i felt",
    "feeling sad",
    "still feel",
    "still feeling",
    "feel sad",
    "feel weird",
    "i've been",
    "i can't",
    "i don't know",
    "i'm scared",
    "i'm angry",
    "i'm sad",
    "i'm tired",
    "i cried",
    "i broke down",
    "it hurts",
)

_DISCLOSURE_CUES_RU: tuple[str, ...] = (
    "я чувств",
    "мне тяж",
    "не знаю",
    "запутал",
    "тревож",
    "расстро",
    "устал",
    "устала",
    "боюсь",
    "плач",
    "больно",
    "напряж",
    "плохо",
    "тяжело",
    "одинок",
)

_DISCLOSURE_CUES_KK: tuple[str, ...] = (
    "мен сез",
    "білмей",
    "қиын",
    "шарш",
    "қорқ",
    "мазасыз",
)

_RELATIONSHIP_DISTRESS: tuple[str, ...] = (
    "broke up",
    "breakup",
    "break up",
    "dumped me",
    "left me",
    "расстал",
    "бросил",
    "бросила",
    "расставан",
    "разошл",
)

_DISCLOSURE_WORD_THRESHOLD: int = 80
_DISCLOSURE_WORD_THRESHOLD_CYRILLIC: int = 25

_VALID_STATES: frozenset[str] = frozenset(
    ("intake", "disclosure", "psychoeducation", "action_planning", "crisis")
)


def _last_user_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return (msg.get("content") or "").lower()
    return ""


def _cyrillic_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff" or c in "ёЁ")
    return cyr / len(letters)


def _disclosure_word_threshold(text: str) -> int:
    if _cyrillic_letter_ratio(text) > 0.5:
        return _DISCLOSURE_WORD_THRESHOLD_CYRILLIC
    return _DISCLOSURE_WORD_THRESHOLD


def _matches_disclosure(text: str) -> bool:
    all_cues = (
        _DISCLOSURE_CUES
        + _DISCLOSURE_CUES_RU
        + _DISCLOSURE_CUES_KK
        + _RELATIONSHIP_DISTRESS
    )
    if any(c in text for c in all_cues):
        return True
    return len(text.split()) > _disclosure_word_threshold(text)


def _recent_user_texts(messages: list[Message], *, limit: int = 3) -> list[str]:
    texts: list[str] = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            t = (msg.get("content") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= limit:
                break
    return list(reversed(texts))


def _thread_text(messages: list[Message], *, user_limit: int = 3) -> str:
    return " ".join(_recent_user_texts(messages, limit=user_limit)).lower()


def is_emotional_thread(messages: list[Message]) -> bool:
    """True when recent user turns carry disclosure/grief content (not just the last ack)."""
    return _matches_disclosure(_thread_text(messages))


def detect_state(
    messages: list[Message],
    *,
    check_crisis: bool = True,
) -> DaisyState:
    """Classify the current conversation phase from recent messages.

    Deconfliction with safety.py: in the live inference flow (score.py),
    safety.crisis_tier() is the authoritative crisis gate and short-circuits
    before this function is called. Callers in that flow should pass
    check_crisis=False so the crisis branch here cannot double-fire.
    The default (True) preserves standalone behavior for other callers / tests.
    """
    text = _last_user_text(messages)
    if not text:
        return "intake"

    if check_crisis and any(p in text for p in _CRISIS_PATTERNS):
        return "crisis"
    if any(p in text for p in _PSYCHOED_PATTERNS):
        return "psychoeducation"
    if any(p in text for p in _ACTION_PATTERNS):
        return "action_planning"

    if _matches_disclosure(text):
        return "disclosure"

    if is_emotional_thread(messages):
        return "disclosure"

    return "intake"
