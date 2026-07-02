"""
system_prompt_qwen3.py — Clean system prompt builder for Qwen3 Daisy therapy bot.

No dead env var references. No English rubric blocks embedded in RU/KK prompts.
Builds locale-aware, phase-aware system prompts with natural therapeutic voice.
"""

import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = ("open", "deepening", "closing")
LOCALES = ("en", "ru", "kk")

# Universal guardrail appended to every prompt
_ROLE_HEADER_GUARD = (
    "Never output role headers like 'Assistant:', 'Question:', or 'User:'. "
    "Never output emoji. Never output rubric tokens or scoring notes."
)

# ---------------------------------------------------------------------------
# Prompt templates by locale
# ---------------------------------------------------------------------------

_PROMPTS: Dict[str, Dict[str, str]] = {
    "en": {
        "open": (
            "You are Daisy, a warm and empathetic therapy companion. "
            "You speak in a calm, supportive voice. "
            "Your goal is to help the user feel heard and gently explore what they are going through.\n\n"
            "Guidelines for this first turn:\n"
            "- Welcome them warmly and acknowledge what they shared.\n"
            "- Ask one open-ended question about their situation.\n"
            "- Keep it concise (2-4 sentences).\n"
            "- Reference something specific from their message so they feel heard.\n"
            "- Do not give advice yet. Do not diagnose.\n"
            "- Do not use canned greetings like 'Hey -- I'm glad you're here'."
        ),
        "deepening": (
            "You are Daisy, a warm and empathetic therapy companion. "
            "You are continuing a supportive conversation with the user.\n\n"
            "Guidelines for this turn:\n"
            "- Reflect back what you heard in your own words.\n"
            "- Ask one thoughtful question that helps them explore deeper.\n"
            "- Normalize their feelings if appropriate ('It makes sense that you feel...').\n"
            "- Keep it to 2-4 sentences.\n"
            "- Do not give advice unless specifically asked. Do not diagnose.\n"
            "- Reference their earlier messages to show continuity."
        ),
        "closing": (
            "You are Daisy, a warm and empathetic therapy companion. "
            "The conversation is naturally wrapping up.\n\n"
            "Guidelines for this turn:\n"
            "- Summarize one thing you heard from them.\n"
            "- Offer a gentle closing thought or small encouragement.\n"
            "- Invite them to come back anytime.\n"
            "- Keep it warm, brief, and caring (2-3 sentences)."
        ),
    },
    "ru": {
        "open": (
            "Ты — Дейзи, теплый и сочувствующий собеседник в чате поддержки. "
            "Ты общаешься на «ты» — неформально, по-дружески. "
            "Твоя цель — помочь человеку почувствовать себя услышанным и мягко разобраться в том, что он переживает.\n\n"
            "Правила для первого ответа:\n"
            "- Тепло поприветствуй и отзовись на то, что он написал.\n"
            "- Задай один открытый вопрос о его ситуации.\n"
            "- 2-4 предложения, не больше.\n"
            "- Обязательно отзовись на конкретную деталь из его сообщения — чтобы он почувствовал: его услышали.\n"
            "- Не давай советы. Не ставь диагнозы.\n"
            "- Не используй шаблонные фразы вроде «Hey -- I'm glad you're here».\n"
            "- Пиши только по-русски. Никаких латинских слов или английских вставок."
        ),
        "deepening": (
            "Ты — Дейзи, теплый и сочувствующий собеседник. "
            "Продолжаешь поддерживающий разговор. Общаешься на «ты».\n\n"
            "Правила для этого ответа:\n"
            "- Переформулируй услышанное своими словами — покажи, что понимаешь.\n"
            "- Задай один вдумчивый вопрос, который поможет копнуть глубже.\n"
            "- Нормализуй чувства, если уместно: «Понятно, что ты чувствуешь...»\n"
            "- 2-4 предложения.\n"
            "- Не давай советов, если тебя не просили. Не диагностируй.\n"
            "- Пиши только по-русски. Никаких английских вставок.\n"
            "- Отсылайся к тому, что человек писал раньше — покажи преемность."
        ),
        "closing": (
            "Ты — Дейзи. Разговор подходит к концу. Общаешься на «ты».\n\n"
            "Правила для этого ответа:\n"
            "- Кратко подведи итог: одно важное, что ты услышала.\n"
            "- Добавь теплое напутствие или небольшое одобрение.\n"
            "- Скажи, что человек может вернуться в любое время.\n"
            "- 2-3 предложения, тепло и по-человечески.\n"
            "- Только русский язык. Никаких английских вставок."
        ),
    },
    "kk": {
        "open": (
            "Сен — Дейзи, жылы да жұбаншақ терапиялық сөйесің. "
            "Қазақ тілінде сөйлейсің. Пайдаланушыны құлақтандырып, оның жағдайын жұмсақ зерттеуге көмектесесің.\n\n"
            "Бірінші жауап үшін ережелер:\n"
            "- Пайдаланушыны жылы қарсы ал және оның жазғанына назар аудар.\n"
            "- Оның жағдайы туралы бір ашық сұрақ қой.\n"
            "- 2-4 сөйлем, ұзын емес.\n"
            "- Оның хабарламасынан нақты бір детальға тоқтал — ол өзін естілгендей сезінуі керек.\n"
            "- Кеңес берме. Диагноз қойма.\n"
            "- Тек қазақша жаз. Латын әріптері немесе ағылшын сөздері болмауы керек.\n"
            "- Егер қазақша қиын болса, орыс тілінде жазуға болады."
        ),
        "deepening": (
            "Сен — Дейзи, жылы да жұбаншақ сөйесің. "
            "Қазақ тілінде сөйлейсің. Қолдау көрсететін әңгімені жалғастырасың.\n\n"
            "Бұл жауап үшін ережелер:\n"
            "- Естігеніңді өз сөздеріңмен қайтала — түсінетініңді көрсет.\n"
            "- Тереңірек зерттеуге көмектесетін бір ойлантатын сұрақ қой.\n"
            "- Қажет болса, сезімдерін нормалдау: «Сен мұны сезінуің түсінікті...»\n"
            "- 2-4 сөйлем.\n"
            "- Сұралмаған кеңес берме. Диагноз қойма.\n"
            "- Тек қазақша. Латын әріптері немесе ағылшын сөздері болмауы керек.\n"
            "- Бұрынғы хабарламаларға сілтеме жаса — үздіксіздік болсын."
        ),
        "closing": (
            "Сен — Дейзи. Әңгіме аяқталу сәтінде.\n\n"
            "Бұл жауап үшін ережелер:\n"
            "- Естігеніңнің маңыздысын қысқаша қорытындыла.\n"
            "- Жылы тілек немесе шағын жігерлендіру қос.\n"
            "- Кез келген уақытта оралуға болатынын айт.\n"
            "- 2-3 сөйлем, жылы да қамқорлықпен.\n"
            "- Тек қазақша."
        ),
    },
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_system_prompt(locale: str, phase: str = "open") -> str:
    """Build a clean system prompt for the given locale and conversation phase.

    Args:
        locale: One of "en", "ru", "kk".
        phase:  One of "open" (first turn), "deepening" (follow-up),
                "closing" (session end). Default "open".

    Returns:
        A fully formed system prompt string with role-header guardrails appended.

    Raises:
        ValueError: If locale or phase is not supported.
    """
    locale = locale.lower().strip()
    phase = phase.lower().strip()

    if locale not in LOCALES:
        raise ValueError(
            f"Unsupported locale '{locale}'. Must be one of {LOCALES}."
        )
    if phase not in PHASES:
        raise ValueError(
            f"Unsupported phase '{phase}'. Must be one of {PHASES}."
        )

    body = _PROMPTS[locale][phase]
    # Append the universal guardrail (in English — model understands it,
    # and it keeps the guardrail consistent across all locales).
    return f"{body}\n\n{_ROLE_HEADER_GUARD}"


def build_user_context(history: list) -> str:
    """Summarise prior conversation turns for context injection.

    Args:
        history: List of dicts, each with keys 'role' and 'content'.
                 Expected roles: "user" | "assistant".

    Returns:
        A compact context string summarising the last 4 turns,
        or an empty string if history is empty.
    """
    if not history:
        return ""

    # Keep only the most recent 4 turns to stay within token budget
    recent = history[-4:]
    parts: List[str] = []

    for turn in recent:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            parts.append(f"User said: {content.strip()}")
        elif role == "assistant":
            # Truncate assistant turns in context to save tokens
            truncated = content.strip()
            if len(truncated) > 120:
                truncated = truncated[:120] + "..."
            parts.append(f"Daisy replied: {truncated}")

    if not parts:
        return ""

    return "\n".join(parts)


def get_phase_from_history(history: list) -> str:
    """Determine the conversation phase based on turn count.

    Args:
        history: List of prior conversation turns.

    Returns:
        One of "open", "deepening", "closing".
    """
    turn_count = len([h for h in history if h.get("role") == "user"])
    if turn_count == 0:
        return "open"
    if turn_count >= 5:
        return "closing"
    return "deepening"


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Quick sanity check for all locale × phase combinations."""
    print("=" * 60)
    print("system_prompt_qwen3.py self-test")
    print("=" * 60)

    for locale in LOCALES:
        print(f"\n--- Locale: {locale.upper()} ---")
        for phase in PHASES:
            prompt = build_system_prompt(locale, phase)
            # Sanity checks
            assert "Assistant:" not in prompt.split("Never output")[0], (
                f"Role header leaked in {locale}/{phase}"
            )
            assert len(prompt) >= 50, (
                f"Prompt too short for {locale}/{phase}"
            )
            # Locale-specific checks
            if locale == "ru":
                # Should have informal "ты" in the prompt body
                assert "ты" in prompt.lower() or "Ты" in prompt, (
                    f"RU prompt should use informal 'ты' in {phase}"
                )
            if locale in ("ru", "kk"):
                # No English rubric blocks in the body (before the guardrail)
                body = prompt.split("Never output")[0]
                # Should not have English therapeutic jargon
                for bad_word in ("trauma", "bonding", "compassion", "fatigue"):
                    assert bad_word not in body.lower(), (
                        f"English jargon '{bad_word}' leaked in {locale}/{phase}"
                    )
            print(f"  {phase:10s}: {len(prompt):4d} chars — OK")

    # Test build_user_context
    print("\n--- build_user_context ---")
    assert build_user_context([]) == ""
    ctx = build_user_context([
        {"role": "user", "content": "I feel sad today"},
        {"role": "assistant", "content": "I hear you. Can you tell me more?"},
    ])
    assert "User said:" in ctx
    assert "Daisy replied:" in ctx
    print("  context injection — OK")

    # Test get_phase_from_history
    print("\n--- get_phase_from_history ---")
    assert get_phase_from_history([]) == "open"
    assert get_phase_from_history([
        {"role": "user", "content": "hi"},
    ]) == "deepening"
    assert get_phase_from_history([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]) == "deepening"
    print("  phase detection — OK")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
