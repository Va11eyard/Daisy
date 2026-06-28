"""
Token budget for chat: truncate very long history lines, then drop oldest turns after system.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _max_history_message_chars() -> int:
    return max(2000, int(os.environ.get("MAX_HISTORY_MESSAGE_CHARS", "12000")))


def truncate_message_content(text: str, max_chars: int | None = None) -> str:
    """Middle-collapse oversized single messages so one turn cannot monopolize the context window."""
    mc = max_chars if max_chars is not None else _max_history_message_chars()
    if len(text) <= mc:
        return text
    gap = 12
    half = (mc - gap) // 2
    if half < 80:
        return text[:mc]
    return text[:half] + "\n…\n" + text[-half:]


def fit_messages_to_token_budget(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_input_tokens: int,
) -> list[dict[str, str]]:
    """
    messages[0] is system; last message is the current user turn.
    First shorten overlong contents, then remove messages[1], messages[2], … until the prompt fits.
    """
    if not messages:
        return messages

    max_chars = _max_history_message_chars()
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role") or "user"
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        if len(content) > max_chars:
            content = truncate_message_content(content, max_chars)
        out.append({"role": role, "content": content})

    while len(out) > 2:
        try:
            prompt = tokenizer.apply_chat_template(
                out,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning("context_policy apply_chat_template failed: %s", e)
            break
        n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tok <= max_input_tokens:
            break
        logger.info(
            "context trim: dropping oldest message after system (history lines left: %d)",
            len(out) - 2,
        )
        out.pop(1)

    return out
