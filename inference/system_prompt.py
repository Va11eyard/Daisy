"""Build system prompt: language, persona, onboarding, memory, psych profile (lite),
and the voice contract overlay (banned phrases, precision vocab, state-specific tone,
few-shot register reference, global rules).

DAISY_PROMPT_MODE=aligned (default): training-like core + compact voice overlay.
DAISY_PROMPT_MODE=full: legacy full voice contract blocks.
"""

from __future__ import annotations

import os
from typing import Any

from personas import resolve_persona
from prompt_builder import STATE_TONE
from state_detector import DaisyState
from reply_language import language_lock_line, resolve_reply_language
from therapy_identity import get_persona_obedience_line, get_therapy_scope_guardrail, get_voice_lines
from voice_contract import (
    BANNED_PHRASES,
    GLOBAL_RULES,
    HOLLOW_CLOSINGS,
    PRECISION_VOCABULARY,
    STRUCTURAL_RULES,
)


def truncate_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def _bulleted(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _prompt_mode() -> str:
    return os.environ.get("DAISY_PROMPT_MODE", "aligned").lower().strip()


def _legacy_june_prompt() -> bool:
    return os.environ.get("DAISY_PROMPT_LEGACY_JUNE", "").lower() in ("1", "true", "yes")


def _therapy_turn_rules_block() -> str:
    """Highest-priority rules — must appear in the first ~200 tokens of the system prompt."""
    return (
        "THERAPY TURN RULES (follow on every reply):\n"
        "Never end a response without asking the user one open question about their experience.\n"
        "A response that only restates or validates what the user said, then stops, is a failure.\n"
        "Structure: brief reflection (1 sentence max) → one specific observation or gentle reframe "
        "(1–2 sentences) → ONE open question that invites them to elaborate.\n"
        "Aim for 3–5 sentences on ordinary therapy turns; never a single sentence.\n"
        "Do not open with generalized metaphors about anxiety or stress building quietly — "
        "speak to this person directly about what they just said.\n"
        "You are Daisy speaking TO the user: use second person (you/your). "
        "Never speak as the client (no 'myself', 'I feel anxious', 'bracing myself'). "
        "On intake, do not assign homework or 'one concrete thing to work on' — invite them to say more."
    )


def _optional_therapy_turn_rules() -> str:
    """June inference overlay — off by default to match v11 training prompts."""
    if not _legacy_june_prompt():
        return ""
    return _therapy_turn_rules_block() + "\n\n"


def _training_core_block(reply_lang: str) -> str:
    """Matches training JSONL opener — without 'validate feelings first' sympathy default."""
    lang_note = ""
    if reply_lang == "ru":
        lang_note = "Отвечай на русском языке."
    elif reply_lang == "kk":
        lang_note = "Қазақ тілінде жауап бер."
    else:
        lang_note = "Respond in the same language the user writes in."
    critical = (
        _slim_critical_block(reply_lang)
        if reply_lang in ("ru", "kk")
        else _critical_override_block()
    )
    return (
        critical
        + "\n\n"
        + _optional_therapy_turn_rules()
        + "You are Daisy, a warm, perceptive companion for emotional support. "
        "You think like a skilled clinician and speak like a trusted friend who has read the research.\n"
        "Do real therapeutic work: engage with this specific person, help them open up or sort through "
        "what's on their mind, and move the conversation somewhere useful.\n"
        "Do not reply with a single poetic generalization or metaphor that only names the feeling and "
        "then stops — that isn't therapy. Therapy is a dialogue.\n"
        "You are not a substitute for emergency or professional care.\n"
        f"{lang_note}\n"
        + get_therapy_scope_guardrail()
    )


def _compact_voice_overlay(state: DaisyState) -> str:
    top_banned = BANNED_PHRASES[:3]
    lines = [
        f"Where things are right now: {STATE_TONE[state]}",
        "Avoid hollow therapy clichés — don't sound like: "
        + "; ".join(f'"{p}"' for p in top_banned)
        + ".",
        "Vary your length and shape to fit the moment; sound like a real person, not a template.",
    ]
    if _legacy_june_prompt():
        lines.append(
            "Every turn must end with one open question — reflection alone is not enough."
        )
    return "\n".join(lines)


def _critical_override_block() -> str:
    rules = (
        "You are in a live conversation. Respond as a person, not as a textbook.",
        'NEVER begin a response with "Here\'s a careful reading:"',
        'NEVER begin a response with "In plain language:"',
        "NEVER quote from books, research papers, or clinical literature verbatim",
        "NEVER reproduce table of contents, chapter headings, footnotes, or citations",
        'NEVER mention "Who wrote this book", authors, publishers, or page numbers',
        "NEVER produce responses longer than 6 sentences",
        "NEVER start with meta-headers that describe your reply (e.g. strategy labels, "
        "\"One question that…\", or rubric text meant for you only)",
        "NEVER use academic or psychoanalytic terminology without immediately translating it into plain spoken language",
        "Your response must sound like something a human would say in conversation, not something printed in a textbook",
        "If you feel an urge to quote a book passage, STOP and instead write one original sentence that reflects what the user said",
    )
    return "CRITICAL OUTPUT RULES — OVERRIDE ALL OTHER BEHAVIOR:\n" + _bulleted(rules)


def _slim_critical_block(reply_lang: str) -> str:
    """Short critical rules for RU/KK — avoids dumping English rubric into Cyrillic replies."""
    if reply_lang == "ru":
        rules = (
            "Ты в живом разговоре — отвечай как человек, не как учебник.",
            "Никогда не цитируй книги, исследования или рубрики дословно.",
            "Не вставляй мета-заголовки и инструкции в ответ.",
            "Не больше 6 предложений за раз.",
            "Пиши только по-русски — без английских фраз и без латиницы.",
        )
        header = "КРИТИЧЕСКИЕ ПРАВИЛА:"
    elif reply_lang == "kk":
        rules = (
            "Сен тірі сөйлесудесің — оқулық емес, адамша жауап бер.",
            "Кітаптар мен рубрикаларды досын айтпа.",
            "Жауапқа мета-тақырыптар мен нұсқауларды қоспа.",
            "Бір жауапта 6 сөйлемнен артық болма.",
            "Тек қазақша жаз — ағылшын немесе латын әріптерін қолданба.",
        )
        header = "МАҢЫЗДЫ ЕРЕЖЕЛЕР:"
    else:
        return _critical_override_block()
    return header + "\n" + _bulleted(rules)


def _banned_phrases_block() -> str:
    return "NEVER USE:\n" + _bulleted(BANNED_PHRASES)


def _hollow_closings_block() -> str:
    return "NEVER CLOSE WITH:\n" + _bulleted(HOLLOW_CLOSINGS)


def _precision_vocab_block() -> str:
    rows = [
        f"Instead of '{key}' → use: {' / '.join(alts)}"
        for key, alts in PRECISION_VOCABULARY.items()
    ]
    return "PREFER PRECISE LANGUAGE:\n" + "\n".join(rows)


def _interaction_mode_block(state: DaisyState) -> str:
    lines: list[str] = [
        f"CURRENT INTERACTION MODE: {state.upper()}",
        STATE_TONE[state],
    ]
    rules = STRUCTURAL_RULES[state]
    min_s, max_s = rules["min_sentences"], rules["max_sentences"]
    if min_s is None and max_s is None:
        lines.append("Response length: As short as needed.")
    else:
        lines.append(f"Response length: {min_s}–{max_s} sentences.")
    if state == "action_planning" and rules["max_steps"] is not None:
        lines.append(f"Max steps: {rules['max_steps']}.")
    return "\n".join(lines)


def _few_shot_block(state: DaisyState) -> str | None:
    """Few-shot exemplars are for training data only — never inject into live prompts."""
    _ = state
    return None


def _global_rules_block() -> str:
    return "ALWAYS:\n" + _bulleted(GLOBAL_RULES)


def build_minimal_system_prompt(
    *,
    locale: str | None,
    detected_lang: str,
    onboarding_summary: str,
    user_context: str,
    persona: str,
    force_english: bool,
    user_gender: str | None,
    psych_profile: dict[str, Any] | None,
    is_onboarding: bool,
    onboarding_step: int,
    user_image_block: str | None = None,
    state: DaisyState = "intake",
) -> str:
    """Short therapist prompt for base-model simple inference (no voice-contract stack)."""
    reply_lang = resolve_reply_language(detected_lang, locale)
    lang_line = language_lock_line(reply_lang, force_english=force_english)
    core1, core2 = get_voice_lines(os.environ.get("DAISY_VOICE", "therapist"))
    lines = [
        core1,
        core2,
        lang_line,
        get_therapy_scope_guardrail(),
        "Respond in a warm, conversational tone. Ask one open question when appropriate.",
        "You are not a substitute for emergency or professional care.",
    ]

    gender_line = ""
    if user_gender == "female" and not force_english:
        if reply_lang == "ru":
            gender_line = "Пользователь — женщина. Используй женский род в обращении."
        elif reply_lang == "kk":
            gender_line = "Пайдаланушы — әйел."
    elif user_gender == "male" and not force_english:
        if reply_lang == "ru":
            gender_line = "Пользователь — мужчина. Используй мужской род в обращении."
    if gender_line:
        lines.append(gender_line)

    if user_image_block and user_image_block.strip():
        lines.append(user_image_block.strip())
    if onboarding_summary:
        lines.append("About this person:\n" + truncate_chars(onboarding_summary, 4000))
    if user_context:
        lines.append("Remember from past conversations:\n" + truncate_chars(user_context, 2000))

    if psych_profile and isinstance(psych_profile, dict):
        risk = psych_profile.get("riskLevel") or psych_profile.get("risk_level")
        parts = [f"{k}={psych_profile.get(k)}" for k in ("ESI", "BSI", "SSI", "MRI") if psych_profile.get(k) is not None]
        if parts or risk:
            lines.append("Psych profile: " + ", ".join(parts) + (f", risk={risk}" if risk else ""))

    pi, pe = resolve_persona(persona)
    lines.append(get_persona_obedience_line())
    lines.append(pi)
    if pe:
        lines.append("Example tone: " + pe)

    if is_onboarding:
        if onboarding_step <= 0:
            lines.append(
                "First message after onboarding: greet warmly as Daisy, acknowledge what you learned, "
                "invite them to share how they feel."
            )
        elif onboarding_step == 1:
            lines.append("Ask ONE short question about their main goal.")
        else:
            lines.append("Warm closing: summarize and invite them to continue anytime.")

    _ = state
    return "\n".join(lines)


def build_system_prompt(
    *,
    locale: str | None,
    detected_lang: str,
    onboarding_summary: str,
    user_context: str,
    persona: str,
    force_english: bool,
    user_gender: str | None,
    psych_profile: dict[str, Any] | None,
    is_onboarding: bool,
    onboarding_step: int,
    user_image_block: str | None = None,
    state: DaisyState = "intake",
    rag_block: str | None = None,
) -> str:
    reply_lang = resolve_reply_language(detected_lang, locale)
    lang_line = language_lock_line(reply_lang, force_english=force_english)
    aligned = _prompt_mode() != "full"
    use_slim_cyrillic = reply_lang in ("ru", "kk") and not force_english
    slim_prompt = aligned or use_slim_cyrillic

    gender_line = ""
    if user_gender == "female" and not force_english:
        loc = reply_lang
        if loc == "ru":
            gender_line = "Пользователь — женщина. Используй женский род в обращении."
        elif loc == "kk":
            gender_line = "Пайдаланушы — әйел."
    elif user_gender == "male" and not force_english:
        loc = reply_lang
        if loc == "ru":
            gender_line = "Пользователь — мужчина. Используй мужской род в обращении."

    if slim_prompt:
        lines = [_training_core_block(reply_lang), lang_line]
    else:
        voice = os.environ.get("DAISY_VOICE", "companion")
        core1, core2 = get_voice_lines(voice)
        june_rules = _optional_therapy_turn_rules()
        lines = []
        if june_rules:
            lines.append(june_rules.rstrip())
            lines.append("")
        lines.extend(
            [
                _critical_override_block(),
                "",
                core1,
                core2,
                lang_line,
                get_therapy_scope_guardrail(),
            ]
        )

    if gender_line:
        lines.append(gender_line)

    if user_image_block and user_image_block.strip():
        lines.append("\n" + user_image_block.strip())

    if onboarding_summary:
        lines.append("\nAbout this person:\n" + truncate_chars(onboarding_summary, 4000))
    if user_context:
        lines.append("\nRemember from past conversations:\n" + truncate_chars(user_context, 2000))

    if psych_profile and isinstance(psych_profile, dict):
        risk = psych_profile.get("riskLevel") or psych_profile.get("risk_level")
        parts = [f"{k}={psych_profile.get(k)}" for k in ("ESI", "BSI", "SSI", "MRI") if psych_profile.get(k) is not None]
        if parts or risk:
            lines.append("Psych profile: " + ", ".join(parts) + (f", risk={risk}" if risk else ""))

    lines.append("\n" + get_persona_obedience_line())
    pi, pe = resolve_persona(persona)
    lines.append("\n" + pi)
    if pe:
        lines.append("Example tone: " + pe)

    if slim_prompt:
        lines.append("\n" + _compact_voice_overlay(state))
        lines.append("\n" + _global_rules_block())
    else:
        lines.append("\n" + _banned_phrases_block())
        lines.append("\n" + _hollow_closings_block())
        lines.append("\n" + _precision_vocab_block())
        lines.append("\n" + _interaction_mode_block(state))
        fs = _few_shot_block(state)
        if fs is not None:
            lines.append("\n" + fs)
        lines.append("\n" + _global_rules_block())

    if rag_block and rag_block.strip():
        lines.append("\n" + rag_block.strip())

    if is_onboarding:
        if onboarding_step <= 0:
            lines.append(
                "\nFirst message after onboarding: greet warmly as Daisy, acknowledge what you learned, "
                "invite them to share how they feel. 3-4 sentences."
            )
        elif onboarding_step == 1:
            lines.append("\nAsk ONE short question about their main goal.")
        else:
            lines.append("\nWarm closing: summarize and invite them to continue anytime.")

    return "\n".join(lines)
