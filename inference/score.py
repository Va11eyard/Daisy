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
  Layer 3 - Answer floor           (regen once only if the answer is empty/degenerate)
"""

from __future__ import annotations

import json
import logging
import os
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
    generation_used_wrong_script,
    resolve_reply_language,
    strip_cjk_from_response,
)
from system_prompt import build_system_prompt
from translator import Translator
from user_image import format_user_image_for_prompt, parse_user_image_field

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

# Qwen3 reasoning needs room for the <think> block plus the visible answer.
MAX_CONTEXT_TOKENS = 8192
RESERVED_FOR_RESPONSE_TOKENS = 2048
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_CAP = 2048
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


def _process_generated(raw: str, reply_lang: str) -> tuple[str, bool]:
    if generation_used_wrong_script(raw, reply_lang):
        raw = strip_cjk_from_response(raw)
    return postprocess_model_response(raw, reply_lang)


def _answer_is_degenerate(text: str) -> bool:
    """Quality floor only: empty or content-free output (no template enforcement)."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    return not any(ch.isalnum() for ch in t)


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
        try:
            # Qwen3 thinking-mode recommended temperature is 0.6.
            temperature = float(data.get("temperature", 0.6))
        except (TypeError, ValueError):
            temperature = 0.6
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

        # ---- Layer 2: RAG context injection ----
        rag_passages = rag.retrieve(
            user_message_en,
            reply_lang=reply_lang,
            phase=state,
            top_k=RAG_TOP_K,
            history_snippet=history_snippet,
        )
        rag_block = rag.format_rag_block(rag_passages)
        layer_trace.append(
            {"layer": 2, "name": "rag_injection", "retrieved": len(rag_passages), "ready": rag.rag_ready()}
        )

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
            rag_block=rag_block,
        )
        # Phase (state) is a soft retrieval/tone hint only; the model decides its own
        # response shape, so we no longer inject rigid per-phase directives.

        messages = [{"role": "system", "content": system_content}]
        for msg in history_for_prompt:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message_en})

        max_input_tokens = max(512, MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE_TOKENS)
        messages = fit_messages_to_token_budget(messages, tok, max_input_tokens)
        prompt = _apply_chat_template(tok, messages)
        final_tokens = len(tok.encode(prompt, add_special_tokens=False))
        logger.info("Prompt tokens ~%s", final_tokens)

        first_min_tokens = 48 if state in ("intake", "disclosure", "action_planning") else None

        # ---- Single reasoning pass: Qwen3 thinks, then answers; <think> is stripped ----
        raw_response = generate_reply(
            mdl,
            tok,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            min_new_tokens=first_min_tokens,
        )
        response, output_sanitized = _process_generated(raw_response, reply_lang)

        # Quality floor only (no template): if the answer came back empty/content-free
        # (e.g. reasoning ran out of budget before closing </think>), regenerate once
        # with thinking OFF so we get a terminated, leak-free reply. No mandatory
        # question, no sentence count — the model still owns its shape.
        regenerated = False
        if _answer_is_degenerate(response):
            regenerated = True
            direct_prompt = _apply_chat_template(tok, messages, enable_thinking=False)
            regen_raw = generate_reply(
                mdl,
                tok,
                direct_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_new_tokens=80,
                repetition_penalty=1.2,
            )
            regen_response, sanitized = _process_generated(regen_raw, reply_lang)
            if not _answer_is_degenerate(regen_response):
                response = regen_response
                output_sanitized = output_sanitized or sanitized

        layer_trace.append(
            {
                "layer": 3,
                "name": "answer_floor",
                "regenerated": regenerated,
            }
        )

        response = response.encode("utf-8", errors="replace").decode("utf-8").replace("\u0000", "")

        if use_translation and translator:
            response = translator.translate(response, "en", reply_lang, user_gender=user_gender)

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
            "translation_enabled": bool(translator and translator.openai_client),
            "crisis_detected": False,
            "disclaimer_ru": DISCLAIMER_RU,
            "disclaimer_en": DISCLAIMER_EN,
        }
        if DEBUG_MODE:
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
                "output_sanitized": output_sanitized,
                "has_user_image": bool(user_image_obj),
                "memory_deferred": defer_memory,
                "inference_build": INFERENCE_BUILD,
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
