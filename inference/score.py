"""
Azure ML managed online endpoint entrypoint.

Implements init() / run(raw_data) expected by azureml-inference-server-http.

run() executes a single linear pipeline that trusts the model to choose its own
response shape, with safety as the only hard enforcement:

  Layer 5 - Crisis hard override   (safety.crisis_tier, fires first, bypasses all)
  Layer 0 - Off-topic / meta short-circuits (safety)
  Layer 1 - Phase hint             (router.detect_phase -> soft tone/RAG hint only)
  Layer 2 - RAG context injection  (rag.retrieve -> [RETRIEVED CONTEXT] block)
            single reasoning pass   (Qwen3 thinks in <think>...</think>, then answers)
  Layer 3 - Answer floor + voice QC  (regen if empty/degenerate; QC hard gate up to 3 regens;
            only ship QC-passing candidates — no template fallbacks)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, cast

from transformers import PreTrainedTokenizer

import rag
from context_policy import fit_messages_to_token_budget
from crisis_resources import get_crisis_resources
from generation import (
    clean_model_text,
    generate_reply,
    postprocess_model_response,
)
from model_loader import adapter_loaded, load_model_and_tokenizer
from router import detect_phase
from onboarding import parse_onboarding
from personas import effective_persona_for_prompt, get_canonical_persona_key
from report_handlers import run_dynamics_insights, run_weekly_report
from safety import (
    crisis_tier,
    is_meta_question,
    is_off_topic,
    normalize_history,
    sanitize_prompt_field,
    sanitize_user_input,
)
from state_detector import Message as DetectMessage
from reply_language import (
    detect_intent_language,
    generation_has_script_leak,
    generation_used_wrong_script,
    language_retry_suffix,
    resolve_reply_language,
    strip_cjk_from_response,
)
from system_prompt import build_minimal_system_prompt, build_system_prompt
from translator import Translator
from user_image import format_user_image_for_prompt, parse_user_image_field
from voice_qc import (
    is_exemplar_echo,
    is_therapist_role_break,
    is_too_brief,
    violates_voice_contract,
    voice_regen_suffix,
)
from minimal_inference import (
    bare_minimum_enabled,
    build_minimal_system_prompt as build_bare_system_prompt,
    cap_history,
    generation_temperature,
    max_generation_tokens,
    minimal_clean,
)
from system_prompt_qwen3 import summarize_history_for_prompt

DEBUG_MODE = os.environ.get("DEBUG_MODE", "").lower() in ("1", "true", "yes")
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", "262144"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "50"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: Any = None
tokenizer: PreTrainedTokenizer | None = None
translator: Translator | None = None

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_LABEL = os.environ.get("MODEL_DISPLAY_NAME", "daisy-model")
INFERENCE_BUILD = os.environ.get("INFERENCE_BUILD", "2026-06-phi-rebuild")
INFERENCE_MODE = os.environ.get("DAISY_INFERENCE_MODE", "").lower().strip()
SIMPLE_INFERENCE = INFERENCE_MODE == "simple"

# Qwen3 reasoning needs room for the <think> block plus the visible answer.
MAX_CONTEXT_TOKENS = 8192
RESERVED_FOR_RESPONSE_TOKENS = 2048
DEFAULT_MAX_TOKENS = int(os.environ.get("DAISY_DEFAULT_MAX_TOKENS", "384"))
MAX_TOKENS_CAP = int(os.environ.get("DAISY_MAX_TOKENS_CAP", "768"))
LORA_THERAPY_MIN_NEW = int(os.environ.get("DAISY_LORA_THERAPY_MIN_NEW", "56"))
THERAPY_MIN_NEW = int(os.environ.get("DAISY_THERAPY_MIN_NEW", "48"))
LORA_DEFAULT_TEMP = float(os.environ.get("DAISY_LORA_DEFAULT_TEMP", "0.75"))
# Qwen3 thinking can consume hundreds of tokens before the visible answer.
MIN_THINKING_MAX_TOKENS = 768
THINKING_RETRY_MAX_TOKENS = 1536
RAG_TOP_K = int(os.environ.get("DAISY_RAG_TOP_K", "3"))

DISCLAIMER_RU = (
    "Внимание: Daisy — AI-помощник для эмоциональной поддержки, не замена профессиональной помощи; "
    "в кризисе обратитесь к специалисту или на линию доверия."
)
DISCLAIMER_EN = (
    "Note: Daisy is an AI support companion, not a substitute for professional care; "
    "in a crisis, contact a qualified professional or crisis line."
)

META_RESPONSES = {
    "who_created": {
        "en": "I was built by a team who wanted Daisy to be there for people when things feel heavy. I'm not perfect, but I'm here to listen.",
        "ru": "Меня создала команда, которая хотела, чтобы Daisy могла быть рядом, когда тяжело. Я не идеальна, но я здесь, чтобы выслушать.",
        "kk": "Мені Daisy командасы жасады. Daisy адамдарға қиын сәтте қолдау болуы үшін жасалған ЖИ. Мен мінсіз емеспін, бірақ тыңдауға дайынмын.",
    }
}


_DEGENERATE_LAST_RESORT: dict[str, str] = {
    "en": "I'm here — could you say a bit more?",
    "ru": "Я здесь — не мог(ла) бы ты рассказать чуть подробнее?",
    "kk": "Мен мұндамын — сәл толығырақ айта аласыз ба?",
}


def _degenerate_last_resort(reply_lang: str) -> str:
    """Minimal non-therapeutic line when every model candidate is empty — never topic-specific."""
    return _DEGENERATE_LAST_RESORT.get(reply_lang) or _DEGENERATE_LAST_RESORT["en"]


def _count_script_chars_for_qc(text: str) -> tuple[int, int, bool]:
    latin = 0
    cyr = 0
    has_kk = False
    for c in text:
        if not c.isalpha():
            continue
        if "\u0400" <= c <= "\u04ff":
            cyr += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    return latin, cyr, has_kk


def _apply_post_translate_qc(response: str, reply_lang: str) -> str:
    """Optional QC on back-translated RU/KK replies (translate deployment v2)."""
    if reply_lang not in ("ru", "kk"):
        return response
    if os.environ.get("DAISY_POST_TRANSLATE_QC", "false").lower() not in ("1", "true", "yes"):
        return response
    text = (response or "").strip()
    min_len = int(os.environ.get("DAISY_TRANSLATE_MIN_LENGTH", "40"))
    if len(text) < min_len:
        return _degenerate_last_resort(reply_lang)
    if os.environ.get("DAISY_TRANSLATE_SCRIPT_GUARD", "true").lower() in ("1", "true", "yes"):
        if generation_has_script_leak(text, reply_lang):
            return _degenerate_last_resort(reply_lang)
        latin, cyr, _ = _count_script_chars_for_qc(text)
        alpha = latin + cyr
        if alpha > 0:
            max_ratio = float(os.environ.get("DAISY_TRANSLATE_MAX_LATIN_RATIO", "0.08"))
            if latin / alpha > max_ratio:
                return _degenerate_last_resort(reply_lang)
    return text


def _last_assistant_content(history: list[dict[str, str]]) -> str | None:
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return None


def _user_history_snippet(history: list[dict[str, str]], *, limit: int = 3) -> str:
    turns = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    return " ".join(turns[-limit:])


def _recent_assistant_contents(history: list[dict[str, str]], *, limit: int = 3) -> list[str]:
    turns = [m["content"] for m in history if m.get("role") == "assistant" and m.get("content")]
    return turns[-limit:]


def _apply_chat_template(
    tok: PreTrainedTokenizer, messages: list[dict[str, str]], *, enable_thinking: bool = True
) -> str:
    """Render the chat template. Qwen3 'thinking' is on by default so the model can
    reason before replying (the <think> block is stripped downstream); the quality
    floor renders with thinking off to guarantee a terminated, leak-free answer."""
    try:
        return cast(
            str,
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            ),
        )
    except TypeError:
        return cast(
            str,
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        )


def _process_generated(
    raw: str,
    reply_lang: str,
    *,
    aggressive_trim: bool | None = None,
) -> tuple[str, bool, str]:
    if generation_used_wrong_script(raw, reply_lang):
        raw = strip_cjk_from_response(raw)
    processed, sanitized = postprocess_model_response(
        raw, reply_lang, aggressive_trim=aggressive_trim
    )
    return processed, sanitized, raw.strip()


def _answer_is_degenerate(text: str) -> bool:
    """Quality floor only: empty or content-free output (no template enforcement)."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    return not any(ch.isalnum() for ch in t)


def _effective_max_tokens(client_max: int, *, thinking: bool = True) -> int:
    """Floor token budget so Qwen3 thinking + answer both fit.

    Simple mode disables thinking (enable_thinking=False), so the 768-token
    thinking floor doesn't apply there — enforcing it anyway silently
    overrides DAISY_DEFAULT_MAX_TOKENS and gives generation far more runway
    than intended to drift into post-reply corruption/hallucinated turns.
    """
    capped = min(client_max, MAX_TOKENS_CAP)
    if not thinking:
        return capped
    return max(capped, MIN_THINKING_MAX_TOKENS)


QC_MAX_REGENS = 1
_FINAL_QC_SUFFIX = (
    "\n\nYour reply must be 3 sentences and end with one open question about their experience."
)
_THERAPY_QC_STATES = frozenset({"intake", "disclosure", "psychoeducation", "action_planning"})


def _pick_best_structural(
    *candidates: str,
    state: str,
) -> str:
    """Prefer model output with dialogue shape when full QC fails (no echo / one-liners)."""
    best = ""
    best_score = -1
    for candidate in candidates:
        t = (candidate or "").strip()
        if _answer_is_degenerate(t) or is_exemplar_echo(t) or is_too_brief(t, state):
            continue
        sents = len([p for p in re.split(r"[.!?…]+", t) if p.strip()])
        score = sents * 100 + (50 if "?" in t else 0) + len(t)
        if score > best_score:
            best_score = score
            best = t
    return best


def _pick_best_loose(*candidates: str, state: str = "intake") -> str:
    """Best non-degenerate model output when QC fails — never ship hollow one-liners."""
    best = ""
    best_score = -1
    therapy = state in _THERAPY_QC_STATES
    for candidate in candidates:
        t = (candidate or "").strip()
        if _answer_is_degenerate(t):
            continue
        if therapy and is_too_brief(t, state):
            continue
        sents = len([p for p in re.split(r"[.!?…]+", t) if p.strip()])
        score = sents * 100 + (50 if "?" in t else 0) + len(t)
        if score > best_score:
            best_score = score
            best = t
    return best


def _pick_best_any_non_empty(*candidates: str) -> str:
    """Longest non-degenerate model text — prefer model output over static fallback."""
    best = ""
    best_score = -1
    for candidate in candidates:
        t = (candidate or "").strip()
        if _answer_is_degenerate(t):
            continue
        if len(t) > best_score:
            best_score = len(t)
            best = t
    return best


def _pick_best_qc_passing(
    *candidates: str,
    state: str,
    reply_lang: str,
    user_message: str,
) -> str:
    """Return the best candidate that passes voice QC; empty string if none pass."""
    best = ""
    best_score = -1
    for candidate in candidates:
        t = (candidate or "").strip()
        if _answer_is_degenerate(t):
            continue
        if violates_voice_contract(
            t, state, reply_lang=reply_lang, user_message=user_message
        ):
            continue
        sents = len([p for p in re.split(r"[.!?…]+", t) if p.strip()])
        score = sents * 100 + (50 if "?" in t else 0) + len(t)
        if score > best_score:
            best_score = score
            best = t
    return best


def _pick_best_with_question(*candidates: str) -> str:
    """Prefer model text that includes an open question and sane therapist voice."""
    best = ""
    best_score = -1
    for candidate in candidates:
        t = (candidate or "").strip()
        if _answer_is_degenerate(t) or is_therapist_role_break(t) or "?" not in t:
            continue
        sents = len([p for p in re.split(r"[.!?…]+", t) if p.strip()])
        score = sents * 100 + len(t)
        if score > best_score:
            best_score = score
            best = t
    return best


def init() -> None:
    global model, tokenizer, translator
    logger.info("Initializing Daisy scoring script (BASE_MODEL=%s)", BASE_MODEL)
    translator = Translator()
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, HF_TOKEN)
    try:
        rag.init_rag_index()  # Layer 2 prewarm — embedder + index loaded once.
    except Exception:
        logger.exception("RAG init failed; continuing without retrieval grounding")
    logger.info("Scoring init complete.")


def _extract_gender(data: dict) -> str | None:
    g = (data.get("gender") or "").strip().lower()
    if g in ("female", "f", "женский", "ж", "әйел"):
        return "female"
    if g in ("male", "m", "мужской", "м", "ер"):
        return "male"
    return None


def _bare_minimum_json_response(
    *,
    mdl: Any,
    tok: PreTrainedTokenizer,
    data: dict[str, Any],
    user_message_en: str,
    history_for_prompt: list[dict[str, str]],
    reply_lang: str,
    locale: str | None,
    onboarding_summary: str,
    user_context: str,
    persona: str,
    use_translation: bool,
    user_gender: str | None,
    layer_trace: list[dict[str, Any]],
) -> str:
    """Single-pass generation with minimal prompt and clean — no QC/RAG/regen."""
    history_capped = cap_history(history_for_prompt)
    history_summary = summarize_history_for_prompt(history_capped, reply_lang)
    system_content = build_bare_system_prompt(
        reply_lang,
        history_summary=history_summary,
        user_context=user_context,
        onboarding_summary=onboarding_summary,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for msg in history_capped:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message_en})

    prompt = _apply_chat_template(tok, messages, enable_thinking=False)
    max_tokens = max_generation_tokens()
    temperature = generation_temperature()
    raw_response = generate_reply(
        mdl,
        tok,
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=None,
        min_new_tokens=None,
    )
    if generation_used_wrong_script(raw_response, reply_lang):
        raw_response = strip_cjk_from_response(raw_response)
    response = minimal_clean(raw_response.strip())
    if not response:
        response = _degenerate_last_resort(reply_lang)

    layer_trace.append(
        {
            "layer": "bare",
            "name": "single_pass_generate",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "raw_len": len(raw_response),
        }
    )

    if use_translation and translator:
        response = translator.translate(response, "en", reply_lang, user_gender=user_gender)
        response = minimal_clean(_apply_post_translate_qc(response, reply_lang))

    out: dict[str, Any] = {
        "response": response.encode("utf-8", errors="replace").decode("utf-8").replace("\u0000", ""),
        "persona_used": get_canonical_persona_key(persona),
        "protocol_used": "cbt",
        "language": reply_lang,
        "model": MODEL_LABEL,
        "inference_build": INFERENCE_BUILD,
        "inference_mode": "bare_minimum",
        "adapter_loaded": adapter_loaded(),
        "translation_enabled": bool(translator and translator.openai_client),
        "crisis_detected": False,
        "disclaimer_ru": DISCLAIMER_RU,
        "disclaimer_en": DISCLAIMER_EN,
    }
    if DEBUG_MODE or bool(data.get("debug")):
        out["debug_context"] = {"layer_trace": layer_trace, "inference_build": INFERENCE_BUILD}
    return json.dumps(out, ensure_ascii=False)


def run(raw_data: str | bytes) -> str:
    global model, tokenizer, translator
    try:
        if isinstance(raw_data, bytes):
            if len(raw_data) > MAX_REQUEST_BYTES:
                return json.dumps({"error": "payload_too_large", "model": MODEL_LABEL}, ensure_ascii=False)
            raw_data = raw_data.decode("utf-8")
        elif len(raw_data.encode("utf-8")) > MAX_REQUEST_BYTES:
            return json.dumps({"error": "payload_too_large", "model": MODEL_LABEL}, ensure_ascii=False)

        data = json.loads(raw_data)
        req_type = data.get("request_type") or data.get("action")
        mdl, tok = model, tokenizer

        if req_type == "dynamics_insights":
            if mdl is None or tok is None:
                return json.dumps({"error": "model not initialized", "model": MODEL_LABEL}, ensure_ascii=False)
            return run_dynamics_insights(mdl, tok, data)
        if req_type == "weekly_report":
            if mdl is None or tok is None:
                return json.dumps({"error": "model not initialized", "model": MODEL_LABEL}, ensure_ascii=False)
            return run_weekly_report(mdl, tok, data)

        raw_message = (data.get("message") or data.get("text") or "").strip()
        layer_trace: list[dict[str, Any]] = []

        # ---- Layer 5: Crisis hard override (fires first, bypasses Layers 1-4) ----
        tier = crisis_tier(raw_message)
        if tier:
            layer_trace.append({"layer": 5, "name": "crisis_override", "fired": True, "tier": tier})
            reply_lang = resolve_reply_language(
                detect_intent_language(raw_message), (data.get("locale") or "")[:2] or None
            )
            resources = get_crisis_resources(reply_lang)
            if reply_lang == "ru":
                short = "Я слышу, что вам очень тяжело. Пожалуйста, обратитесь к специалисту или на линию доверия."
            elif reply_lang == "kk":
                short = "Сіздің қиындығыңызды естідім. Маманға немесе сенім сызығына хабарласыңыз."
            else:
                short = "I hear you're in pain. Please reach out to a crisis line or a mental health professional."
            lines = [short]
            for r in resources[:3]:
                lines.append(f"{r.get('name', '')}: {r.get('contact', '')}")
            out = {
                "response": "\n".join(lines),
                "persona_used": "crisis",
                "protocol_used": "crisis",
                "language": reply_lang,
                "model": MODEL_LABEL,
                "crisis_detected": True,
                "crisis_tier": tier,
                "disclaimer_ru": DISCLAIMER_RU,
                "disclaimer_en": DISCLAIMER_EN,
            }
            if DEBUG_MODE:
                out["debug_context"] = {"layer_trace": layer_trace, "inference_build": INFERENCE_BUILD}
            return json.dumps(out, ensure_ascii=False)

        layer_trace.append({"layer": 5, "name": "crisis_override", "fired": False})

        user_message = sanitize_user_input(raw_message)
        conversation_history = normalize_history(data.get("history") or [], max_turns=MAX_HISTORY_TURNS)
        try:
            max_tokens = min(int(data.get("max_tokens", DEFAULT_MAX_TOKENS)), MAX_TOKENS_CAP)
        except (TypeError, ValueError):
            max_tokens = DEFAULT_MAX_TOKENS
        max_tokens = _effective_max_tokens(max_tokens, thinking=not SIMPLE_INFERENCE)
        try:
            if "temperature" in data:
                temperature = float(data.get("temperature", 0.6))
            elif adapter_loaded():
                temperature = LORA_DEFAULT_TEMP
            else:
                temperature = 0.6
        except (TypeError, ValueError):
            temperature = LORA_DEFAULT_TEMP if adapter_loaded() else 0.6
        temperature = max(0.0, min(1.0, temperature))
        locale = (data.get("locale") or data.get("language") or "").lower()[:2] or None

        raw_onboarding = sanitize_prompt_field(data.get("onboarding_summary") or "")
        user_context_raw = data.get("user_context") or ""
        if isinstance(user_context_raw, (dict, list)):
            user_context_raw = json.dumps(user_context_raw, ensure_ascii=False)
        user_context_raw = sanitize_prompt_field(str(user_context_raw))
        onboarding_summary, og_gender = parse_onboarding(raw_onboarding)
        user_context = user_context_raw
        if onboarding_summary and user_context and onboarding_summary.strip() == user_context.strip():
            user_context = ""
        if user_context and len(user_context) > 1500 and translator:
            user_context = translator.summarize_memory(user_context)

        persona = effective_persona_for_prompt(data)
        is_onboarding = bool(data.get("is_onboarding_session"))
        onboarding_step = int(data.get("onboarding_step", 0))
        psych_profile = data.get("psych_profile")

        intent_lang = detect_intent_language(user_message)
        reply_lang = resolve_reply_language(intent_lang, locale)
        if is_onboarding and locale in ("en", "ru", "kk"):
            reply_lang = locale

        if user_context:
            user_context = clean_model_text(user_context, lang=reply_lang)
        if onboarding_summary:
            onboarding_summary = clean_model_text(onboarding_summary, lang=reply_lang)

        if is_meta_question(user_message) and "who_created" in META_RESPONSES:
            layer_trace.append({"layer": 0, "name": "meta_short_circuit", "fired": True})
            lang = reply_lang if reply_lang in META_RESPONSES["who_created"] else "en"
            out = {
                "response": META_RESPONSES["who_created"][lang],
                "persona_used": get_canonical_persona_key(persona),
                "protocol_used": "cbt",
                "language": reply_lang,
                "model": MODEL_LABEL,
                "translation_enabled": bool(translator and translator.openai_client),
                "meta_response": "who_created",
                "disclaimer_ru": DISCLAIMER_RU,
                "disclaimer_en": DISCLAIMER_EN,
            }
            if DEBUG_MODE:
                out["debug_context"] = {"layer_trace": layer_trace, "inference_build": INFERENCE_BUILD}
            return json.dumps(out, ensure_ascii=False)

        if is_off_topic(user_message):
            layer_trace.append({"layer": 0, "name": "off_topic_short_circuit", "fired": True})
            off = {
                "en": "I’m here for emotional support and mental wellbeing — not recipes, games, or general topics. "
                "What’s weighing on you or how are you feeling today?",
                "ru": "Я могу помогать с эмоциями, переживаниями и психологическим благополучием — не с рецептами, играми "
                "и прочими темами вне этого. Расскажешь, что сейчас тяжелее всего или что на душе?",
                "kk": "Мен эмоционалдық қолдау және психологиялық амандық туралы ғана сөйлесемін — рецепт, ойын сияқты "
                "басқа тақырыптар емес. Қазіргі сәтте не қиын немесе қалай сезінесіз?",
            }
            lang = reply_lang if reply_lang in off else "en"
            out = {
                "response": off[lang],
                "persona_used": get_canonical_persona_key(persona),
                "protocol_used": "cbt",
                "language": reply_lang,
                "model": MODEL_LABEL,
                "translation_enabled": False,
                "off_topic": True,
                "disclaimer_ru": DISCLAIMER_RU,
                "disclaimer_en": DISCLAIMER_EN,
            }
            if DEBUG_MODE:
                out["debug_context"] = {"layer_trace": layer_trace, "inference_build": INFERENCE_BUILD}
            return json.dumps(out, ensure_ascii=False)

        if mdl is None or tok is None:
            return json.dumps({"error": "model not initialized", "model": MODEL_LABEL}, ensure_ascii=False)

        translation_available = bool(
            translator
            and (
                (translator.use_azure_translator and translator.azure_key)
                or translator.openai_client
            )
        )
        direct_multilingual = os.environ.get("DAISY_DIRECT_MULTILINGUAL", "true").lower() == "true"
        use_translation = reply_lang != "en" and translation_available and not direct_multilingual
        translation_path = "en_round_trip" if use_translation else "direct"

        history_original = [
            {
                "role": m["role"],
                "content": clean_model_text((m.get("content") or "").strip(), lang=reply_lang),
            }
            for m in conversation_history
            if isinstance(m, dict) and m.get("role") and (m.get("content") or "").strip()
        ]
        last_assistant = _last_assistant_content(history_original)

        if use_translation and translator:
            user_message_en = translator.translate(user_message, reply_lang, "en")
            history_for_prompt = []
            for msg in history_original:
                content = msg["content"]
                ml = detect_intent_language(content) or "en"
                if ml == "en":
                    history_for_prompt.append({"role": msg["role"], "content": content})
                else:
                    history_for_prompt.append(
                        {"role": msg["role"], "content": translator.translate(content, ml, "en")}
                    )
        else:
            user_message_en = user_message
            history_for_prompt = list(history_original)

        if bare_minimum_enabled():
            return _bare_minimum_json_response(
                mdl=mdl,
                tok=tok,
                data=data,
                user_message_en=user_message_en,
                history_for_prompt=history_for_prompt,
                reply_lang=reply_lang,
                locale=locale,
                onboarding_summary=onboarding_summary,
                user_context=user_context,
                persona=persona,
                use_translation=use_translation,
                user_gender=_extract_gender(data) or og_gender,
                layer_trace=layer_trace,
            )

        user_gender = _extract_gender(data) or og_gender

        user_image_obj = parse_user_image_field(data.get("user_image"))
        user_image_block = None
        if user_image_obj:
            user_image_block = format_user_image_for_prompt(
                user_image_obj, force_english=use_translation, locale=locale or reply_lang
            )

        # ---- Layer 1: Input classification (single routing decision) ----
        state_msgs: list[DetectMessage] = [
            cast(DetectMessage, {"role": m["role"], "content": m["content"]})
            for m in history_for_prompt
        ]
        history_snippet = _user_history_snippet(history_for_prompt)
        state_msgs.append({"role": "user", "content": user_message_en})
        state = detect_phase(state_msgs)
        layer_trace.append({"layer": 1, "name": "input_classification", "phase": state})

        lora_adapter = adapter_loaded()

        # ---- Layer 2: RAG context injection ----
        if SIMPLE_INFERENCE:
            rag_passages: list = []
            rag_block = None
            layer_trace.append(
                {
                    "layer": 2,
                    "name": "rag_injection",
                    "retrieved": 0,
                    "ready": rag.rag_ready(),
                    "skipped": "simple",
                }
            )
        else:
            rag_passages = rag.retrieve(
                user_message_en,
                reply_lang=reply_lang,
                phase=state,
                top_k=RAG_TOP_K,
                history_snippet=history_snippet,
            )
            rag_block = rag.format_rag_block(rag_passages)
            layer_trace.append(
                {
                    "layer": 2,
                    "name": "rag_injection",
                    "retrieved": len(rag_passages),
                    "ready": rag.rag_ready(),
                }
            )

        if SIMPLE_INFERENCE and not lora_adapter:
            system_content = build_minimal_system_prompt(
                locale=locale,
                detected_lang=reply_lang,
                onboarding_summary=onboarding_summary,
                user_context=user_context,
                persona=persona,
                force_english=use_translation,
                user_gender=user_gender,
                psych_profile=psych_profile if isinstance(psych_profile, dict) else None,
                is_onboarding=is_onboarding,
                onboarding_step=onboarding_step,
                user_image_block=user_image_block,
                state=state,
            )
        else:
            system_content = build_system_prompt(
                locale=locale,
                detected_lang=reply_lang,
                onboarding_summary=onboarding_summary,
                user_context=user_context,
                persona=persona,
                force_english=use_translation,
                user_gender=user_gender,
                psych_profile=psych_profile if isinstance(psych_profile, dict) else None,
                is_onboarding=is_onboarding,
                onboarding_step=onboarding_step,
                user_image_block=user_image_block,
                state=state,
                rag_block=None if SIMPLE_INFERENCE else rag_block,
            )

        messages = [{"role": "system", "content": system_content}]
        for msg in history_for_prompt:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message_en})

        max_input_tokens = max(512, MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE_TOKENS)
        messages = fit_messages_to_token_budget(messages, tok, max_input_tokens)

        thinking_stripped_empty = False
        regenerated = False
        voice_qc_regenerated = False
        voice_qc_regen_count = 0
        voice_qc_best_of_model = False
        output_sanitized = False
        model_raw_text = ""
        trim_meta: dict[str, Any] = {}
        request_debug = DEBUG_MODE or bool(data.get("debug"))

        if SIMPLE_INFERENCE:
            prompt = _apply_chat_template(tok, messages, enable_thinking=False)
            final_tokens = len(tok.encode(prompt, add_special_tokens=False))
            logger.info("Prompt tokens ~%s (simple mode)", final_tokens)

            first_min_tokens = (
                THERAPY_MIN_NEW if state in _THERAPY_QC_STATES else None
            )
            raw_response = generate_reply(
                mdl,
                tok,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_new_tokens=first_min_tokens,
            )
            response, output_sanitized, model_raw_text = _process_generated(
                raw_response, reply_lang, aggressive_trim=False
            )
            trim_meta = {
                "model_raw_text": model_raw_text[:2000],
                "shipped_len": len(response),
                "raw_len": len(model_raw_text),
                "trim_amputated": len(model_raw_text) > len(response) + 20,
                "min_new_tokens": first_min_tokens,
                "aggressive_trim": False,
            }
            if generation_has_script_leak(response, reply_lang):
                regen_messages = list(messages)
                regen_messages[0] = {
                    "role": "system",
                    "content": regen_messages[0]["content"] + language_retry_suffix(reply_lang),
                }
                regen_prompt = _apply_chat_template(tok, regen_messages, enable_thinking=False)
                regen_raw = generate_reply(
                    mdl,
                    tok,
                    regen_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    min_new_tokens=first_min_tokens,
                )
                regen_response, regen_sanitized, regen_raw_text = _process_generated(
                    regen_raw, reply_lang, aggressive_trim=False
                )
                if not generation_has_script_leak(regen_response, reply_lang):
                    response = regen_response
                    output_sanitized = output_sanitized or regen_sanitized
                    model_raw_text = regen_raw_text
                    regenerated = True
            if _answer_is_degenerate(response):
                thinking_stripped_empty = True
                retry_raw = generate_reply(
                    mdl,
                    tok,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    min_new_tokens=first_min_tokens,
                )
                retry_response, retry_sanitized, retry_raw_text = _process_generated(
                    retry_raw, reply_lang, aggressive_trim=False
                )
                if not _answer_is_degenerate(retry_response):
                    response = retry_response
                    output_sanitized = output_sanitized or retry_sanitized
                    model_raw_text = retry_raw_text
        else:
            primary_thinking = False if lora_adapter else state not in _THERAPY_QC_STATES
            prompt = _apply_chat_template(tok, messages, enable_thinking=primary_thinking)
            final_tokens = len(tok.encode(prompt, add_special_tokens=False))
            logger.info("Prompt tokens ~%s", final_tokens)

            first_min_tokens = (
                LORA_THERAPY_MIN_NEW
                if lora_adapter and state in _THERAPY_QC_STATES
                else (120 if state in _THERAPY_QC_STATES else None)
            )

            model_candidates: list[str] = []

            raw_response = generate_reply(
                mdl,
                tok,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_new_tokens=first_min_tokens,
            )
            response, output_sanitized, _ = _process_generated(
                raw_response, reply_lang, aggressive_trim=True
            )
            model_candidates.append(response)

            retry_response = ""
            regen_response = ""
            if _answer_is_degenerate(response):
                thinking_stripped_empty = True
                retry_tokens = min(THINKING_RETRY_MAX_TOKENS, MAX_TOKENS_CAP)
                retry_raw = generate_reply(
                    mdl,
                    tok,
                    _apply_chat_template(tok, messages, enable_thinking=primary_thinking),
                    max_new_tokens=retry_tokens,
                    temperature=temperature,
                    min_new_tokens=first_min_tokens,
                )
                retry_response, retry_sanitized, _ = _process_generated(
                    retry_raw, reply_lang, aggressive_trim=True
                )
                model_candidates.append(retry_response)
                if not _answer_is_degenerate(retry_response):
                    response = retry_response
                    output_sanitized = output_sanitized or retry_sanitized
                else:
                    regenerated = True
                    direct_prompt = _apply_chat_template(tok, messages, enable_thinking=False)
                    regen_raw = generate_reply(
                        mdl,
                        tok,
                        direct_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        min_new_tokens=first_min_tokens or 56,
                        repetition_penalty=1.2,
                    )
                    regen_response, sanitized, _ = _process_generated(
                        regen_raw, reply_lang, aggressive_trim=True
                    )
                    model_candidates.append(regen_response)
                    if not _answer_is_degenerate(regen_response):
                        response = regen_response
                        output_sanitized = output_sanitized or sanitized

            if (
                lora_adapter
                and state in _THERAPY_QC_STATES
                and (
                    is_too_brief(response, state)
                    or is_therapist_role_break(response)
                    or is_exemplar_echo(response)
                )
            ):
                regen_msgs = list(messages)
                regen_msgs[0] = {
                    "role": "system",
                    "content": system_content + voice_regen_suffix(state, reply_lang),
                }
                regen_prompt = _apply_chat_template(tok, regen_msgs, enable_thinking=False)
                lora_fix_raw = generate_reply(
                    mdl,
                    tok,
                    regen_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    min_new_tokens=first_min_tokens,
                    repetition_penalty=1.1,
                )
                lora_fix, lora_fix_sanitized, _ = _process_generated(
                    lora_fix_raw, reply_lang, aggressive_trim=True
                )
                model_candidates.append(lora_fix)
                if (
                    not _answer_is_degenerate(lora_fix)
                    and not is_therapist_role_break(lora_fix)
                    and not is_too_brief(lora_fix, state)
                ):
                    response = lora_fix
                    output_sanitized = output_sanitized or lora_fix_sanitized
                else:
                    with_q = _pick_best_with_question(*model_candidates)
                    if with_q:
                        response = with_q
                        output_sanitized = output_sanitized or lora_fix_sanitized

            if state in _THERAPY_QC_STATES and not lora_adapter:
                qc_messages = list(messages)
                qc_messages[0] = {
                    "role": "system",
                    "content": system_content + voice_regen_suffix(state, reply_lang),
                }

                def _qc_passes(text: str) -> bool:
                    return not _answer_is_degenerate(text) and not violates_voice_contract(
                        text,
                        state,
                        reply_lang=reply_lang,
                        user_message=user_message_en,
                    )

                while voice_qc_regen_count < QC_MAX_REGENS and not _qc_passes(response):
                    voice_qc_regenerated = True
                    voice_qc_regen_count += 1
                    qc_msgs = list(qc_messages)
                    if voice_qc_regen_count == QC_MAX_REGENS:
                        qc_msgs[0] = {
                            "role": "system",
                            "content": qc_msgs[0]["content"] + _FINAL_QC_SUFFIX,
                        }
                    qc_budget = (
                        min(THINKING_RETRY_MAX_TOKENS, MAX_TOKENS_CAP)
                        if voice_qc_regen_count >= 2
                        else max_tokens
                    )
                    qc_prompt = _apply_chat_template(tok, qc_msgs, enable_thinking=False)
                    qc_raw = generate_reply(
                        mdl,
                        tok,
                        qc_prompt,
                        max_new_tokens=qc_budget,
                        temperature=temperature,
                        min_new_tokens=120,
                        repetition_penalty=1.15,
                    )
                    qc_response, qc_sanitized, _ = _process_generated(
                        qc_raw, reply_lang, aggressive_trim=True
                    )
                    model_candidates.append(qc_response)
                    if _qc_passes(qc_response):
                        response = qc_response
                        output_sanitized = output_sanitized or qc_sanitized
                        break
                    passing = _pick_best_qc_passing(
                        *model_candidates,
                        state=state,
                        reply_lang=reply_lang,
                        user_message=user_message_en,
                    )
                    if passing:
                        voice_qc_best_of_model = True
                        response = passing
                        output_sanitized = output_sanitized or qc_sanitized
                        break

                if not _qc_passes(response):
                    structural = _pick_best_structural(*model_candidates, state=state)
                    if structural:
                        voice_qc_best_of_model = True
                        response = structural
                    else:
                        passing = _pick_best_qc_passing(
                            *model_candidates,
                            state=state,
                            reply_lang=reply_lang,
                            user_message=user_message_en,
                        )
                        if passing:
                            voice_qc_best_of_model = True
                            response = passing
                        else:
                            loose = _pick_best_loose(*model_candidates, state=state)
                            if loose:
                                voice_qc_best_of_model = True
                                response = loose
                            else:
                                any_model = _pick_best_any_non_empty(*model_candidates)
                                if any_model:
                                    voice_qc_best_of_model = True
                                    response = any_model
                                elif _answer_is_degenerate(response):
                                    response = _degenerate_last_resort(reply_lang)
                                    output_sanitized = True

                if state in _THERAPY_QC_STATES and not _qc_passes(response):
                    passing = _pick_best_qc_passing(
                        *model_candidates,
                        state=state,
                        reply_lang=reply_lang,
                        user_message=user_message_en,
                    )
                    if not passing:
                        passing = _pick_best_structural(*model_candidates, state=state)
                    if not passing:
                        passing = _pick_best_loose(*model_candidates, state=state)
                    if passing:
                        voice_qc_best_of_model = True
                        response = passing
                    else:
                        any_model = _pick_best_any_non_empty(*model_candidates)
                        if any_model:
                            voice_qc_best_of_model = True
                            response = any_model
                        elif _answer_is_degenerate(response):
                            response = _degenerate_last_resort(reply_lang)
                            output_sanitized = True

        if trim_meta:
            layer_trace.append({"layer": 3, "name": "postprocess", **trim_meta})
        layer_trace.append(
            {
                "layer": 3,
                "name": "answer_floor",
                "regenerated": regenerated,
                "thinking_stripped_empty": thinking_stripped_empty,
                "max_tokens_used": max_tokens,
            }
        )
        layer_trace.append(
            {
                "layer": 3,
                "name": "voice_qc",
                "regenerated": voice_qc_regenerated,
                "regen_count": voice_qc_regen_count,
                "best_of_model": voice_qc_best_of_model,
            }
        )

        response = response.encode("utf-8", errors="replace").decode("utf-8").replace("\u0000", "")

        if use_translation and translator:
            response = translator.translate(response, "en", reply_lang, user_gender=user_gender)
            response = _apply_post_translate_qc(response, reply_lang)

        defer_memory = os.environ.get("DEFER_MEMORY_UPDATE", "true").lower() in ("1", "true", "yes")
        ai_profile = None
        if (
            not defer_memory
            and data.get("request_ai_profile")
            and (onboarding_summary or user_context or user_message)
        ):
            ai_profile = (
                translator.generate_user_profile(onboarding_summary, user_context, user_message[:500])
                if translator
                else None
            )
            if ai_profile and isinstance(ai_profile, dict):
                ai_profile["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        memory_update = None
        if not defer_memory and translator and translator.openai_client:
            memory_update = translator.generate_memory_update(user_message[:600], response[:600], reply_lang)

        out: dict = {
            "response": response,
            "persona_used": get_canonical_persona_key(persona),
            "protocol_used": "cbt",
            "language": reply_lang,
            "model": MODEL_LABEL,
            "inference_build": INFERENCE_BUILD,
            "inference_mode": INFERENCE_MODE or "default",
            "adapter_loaded": lora_adapter,
            "translation_enabled": bool(translator and translator.openai_client),
            "crisis_detected": False,
            "disclaimer_ru": DISCLAIMER_RU,
            "disclaimer_en": DISCLAIMER_EN,
        }
        if request_debug:
            out["debug_context"] = {
                "has_onboarding": bool(onboarding_summary),
                "has_memory": bool(user_context),
                "has_persona": bool(persona),
                "user_gender": user_gender,
                "prompt_tokens": final_tokens,
                "daisy_state": state,
                "phase": state,
                "intent_lang": intent_lang,
                "reply_lang": reply_lang,
                "translation_path": translation_path,
                "rag_retrieved": len(rag_passages),
                "rag_ready": rag.rag_ready(),
                "answer_regenerated": regenerated,
                "thinking_stripped_empty": thinking_stripped_empty,
                "max_tokens_used": max_tokens,
                "voice_qc_regenerated": voice_qc_regenerated,
                "voice_qc_regen_count": voice_qc_regen_count,
                "voice_qc_best_of_model": voice_qc_best_of_model,
                "output_sanitized": output_sanitized,
                "model_raw_text": model_raw_text[:2000] if model_raw_text else None,
                "trim_amputated": trim_meta.get("trim_amputated") if trim_meta else None,
                "has_user_image": bool(user_image_obj),
                "memory_deferred": defer_memory,
                "inference_build": INFERENCE_BUILD,
                "inference_mode": INFERENCE_MODE or "default",
                "adapter_loaded": adapter_loaded(),
                "layer_trace": layer_trace,
            }
        if ai_profile is not None:
            out["ai_profile"] = ai_profile
        if memory_update:
            out["memory_update"] = memory_update
        return json.dumps(out, ensure_ascii=False)

    except Exception as e:
        logger.exception("run() failed: %s", e)
        return json.dumps({"error": "internal_error", "model": MODEL_LABEL}, ensure_ascii=False)
