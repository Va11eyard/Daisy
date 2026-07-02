"""Shared bare-minimum inference helpers for score_qwen3 and score.py."""

from __future__ import annotations

import os
import re
from typing import Dict, List

MINIMAL_SYSTEM_PROMPTS: Dict[str, str] = {
    "ru": (
        "Ты — Дэйзи, терапевт. Говори на 'ты', просто и тепло. "
        "Начинай ответ с конкретики — назови ситуацию человека. "
        "НЕ пиши 'Assistant:', 'Question:', '❮REFINE❯', 'ASURE'. "
        "НЕ повторяй одно слово много раз. "
        "Если нечего сказать — задай вопрос."
    ),
    "en": (
        "You are Daisy, a therapy chatbot. Speak simply and warmly. "
        "Start by naming the person's specific situation. "
        "NEVER write 'Assistant:', 'Question:', '❮REFINE❯', 'ASURE'. "
        "NEVER repeat the same word many times. "
        "If unsure — ask a question."
    ),
    "kk": (
        "Сіз Дэйзисіз, терапевт. Жай сөйлеңіз. "
        "Ешқашан 'Assistant:', 'Question:' жазбаңыз."
    ),
}

DEFAULT_STOP_STRINGS: List[str] = [
    "Assistant:",
    "Question:",
    "User:",
    "Human:",
    "\n\nUser",
    "\nUser:",
    "\nAssistant:",
    "\n\nAssistant",
    "assistant",
    "system prompt",
    "**Rubric**",
    "ASURE",
    "❮REFINE❯",
    "OffsetTable",
    "Daisy x",
    "👋ly",
    "\n\n",
]

_LEAK_STRINGS = [
    "Assistant:",
    "Question:",
    "User:",
    "Human:",
    "❮REFINE❯",
    "ASURE",
    "OffsetTable",
    "icaponecessario",
    "Valentino Respini Ricciotti",
]

_META_PHRASES = [
    "Подстраивайся под текущую потребность человека",
    "response as a person",
    "NEVER USE",
    "CRITICAL OUTPUT",
]

HISTORY_TURN_CAP = 6


def bare_minimum_enabled() -> bool:
    return os.environ.get("DAISY_BARE_MINIMUM", "true").lower() in ("1", "true", "yes")


def max_generation_tokens() -> int:
    raw = os.environ.get("DAISY_MAX_TOKENS", "").strip()
    if raw:
        return int(raw)
    return int(os.environ.get("DAISY_DEFAULT_MAX_TOKENS", "512"))


def generation_temperature() -> float:
    raw = os.environ.get("DAISY_TEMP", "").strip()
    if raw:
        return float(raw)
    return float(os.environ.get("DAISY_LORA_DEFAULT_TEMP", "0.7"))


def load_stop_strings() -> List[str]:
    raw = os.environ.get("DAISY_STOP_STRINGS", "").strip()
    if not raw:
        return list(DEFAULT_STOP_STRINGS)
    return [s.strip() for s in raw.split(",") if s.strip()]


def cap_history(history: List[dict], max_turns: int = HISTORY_TURN_CAP) -> List[dict]:
    if not history or len(history) <= max_turns:
        return list(history)
    return list(history[-max_turns:])


def minimal_clean(text: str) -> str:
    """One-pass cleanup for model output."""
    if not text:
        return ""

    for leak in _LEAK_STRINGS:
        text = text.replace(leak, "")

    for meta in _META_PHRASES:
        text = text.replace(meta, "")

    text = re.sub(r"(\b\w+\b)(\s+\1){2,}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"[.,\s]{3,}", "... ", text)
    text = text.strip(" ,.:;\n\t")

    if text and text[-1] not in ".!?":
        for sep in (". ", "? ", "! "):
            idx = text.rfind(sep)
            if idx > len(text) * 0.6:
                text = text[: idx + 1]
                break

    return text.strip()


def build_minimal_system_prompt(
    locale: str,
    *,
    history_summary: str = "",
    user_context: str = "",
    onboarding_summary: str = "",
) -> str:
    """Assemble bare-minimum system prompt with optional context blocks."""
    loc = locale if locale in MINIMAL_SYSTEM_PROMPTS else "en"
    parts = [MINIMAL_SYSTEM_PROMPTS[loc]]

    if history_summary.strip():
        parts.append(f"--- Prior conversation ---\n{history_summary.strip()}")

    if onboarding_summary.strip():
        trimmed = onboarding_summary.strip()[:400]
        parts.append(f"--- About this person ---\n{trimmed}")

    if user_context.strip():
        trimmed = user_context.strip()[:300]
        parts.append(f"--- What you remember ---\n{trimmed}")

    return "\n\n".join(parts)
