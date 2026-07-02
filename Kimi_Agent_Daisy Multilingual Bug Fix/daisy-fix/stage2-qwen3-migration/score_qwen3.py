"""
score_qwen3.py — Simplified 3-layer inference pipeline for Daisy therapy chatbot.

Replaces the 23-module score.py with a clean, auditable architecture:

    Layer 0: Safety    — crisis detection, injection guard, off-topic routing
    Layer 1: Generate  — single generation with locale-aware system prompt
    Layer 2: QC        — lightweight checks (script leak, structural leak, min length)

Model: Qwen/Qwen3-8B (or Qwen3-4B for latency) via transformers + bitsandbytes 4-bit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daisy.score_qwen3")

# ---------------------------------------------------------------------------
# Environment configuration (minimal, ALL functional)
# ---------------------------------------------------------------------------

BASE_MODEL: str = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
INFERENCE_BUILD: str = os.environ.get("INFERENCE_BUILD", "2026-07-qwen3-v1")
INFERENCE_QUANTIZATION: str = os.environ.get("INFERENCE_QUANTIZATION", "4bit")
DAISY_DIRECT_MULTILINGUAL: bool = (
    os.environ.get("DAISY_DIRECT_MULTILINGUAL", "true").lower() == "true"
)
DAISY_DEFAULT_MAX_TOKENS: int = int(os.environ.get("DAISY_DEFAULT_MAX_TOKENS", "120"))
DAISY_LORA_DEFAULT_TEMP: float = float(os.environ.get("DAISY_LORA_DEFAULT_TEMP", "0.6"))
DAISY_VOICE_QC: bool = os.environ.get("DAISY_VOICE_QC", "true").lower() == "true"
DAISY_MAX_VOICE_REGENS: int = int(os.environ.get("DAISY_MAX_VOICE_REGENS", "2"))

# ---------------------------------------------------------------------------
# Dataclasses for layer results
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    passed: bool
    action: str = "pass"           # "pass" | "crisis" | "injection" | "off_topic"
    response: str = ""             # Non-empty if passed=False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time_ms: float = 0.0


@dataclass
class QCResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    can_regen: bool = True
    regen_attempted: bool = False


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None


def _load_model():
    """Lazy-load the Qwen3 model and tokenizer with 4-bit quantisation."""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    logger.info(f"Loading model: {BASE_MODEL} (quant={INFERENCE_QUANTIZATION})")
    t0 = time.time()

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        logger.error(
            "Missing dependencies. Install: pip install transformers torch bitsandbytes"
        )
        raise exc

    # Quantization config
    if INFERENCE_QUANTIZATION == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif INFERENCE_QUANTIZATION == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="left",
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )
    _model.eval()

    _device = next(_model.parameters()).device
    load_time = (time.time() - t0) * 1000
    logger.info(f"Model loaded on {_device} in {load_time:.0f}ms")

    return _model, _tokenizer, _device


# ---------------------------------------------------------------------------
# Layer 0: Safety
# ---------------------------------------------------------------------------

# Crisis keywords — expanded for multilingual coverage
_CRISIS_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "suicide", "kill myself", "end my life", "want to die",
        "self-harm", "cutting myself", "hurt myself",
        "abuse", "being abused", "molested",
        "overdose", "jump off", "hang myself",
    ],
    "ru": [
        "суицид", "хочу умереть", "убить себя", "покончить с собой",
        "самоповреждение", "режу себя", "порезать себя",
        "изнасилован", "домогательств", "насили",
        "передозировка", "прыгнуть с", "повеситься",
    ],
    "kk": [
        "өзімді өлтіру", "өлім", "асылып қалу",
        "зорлау", "үиренші", "зорлық",
    ],
}

# Prompt injection patterns
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"forget (your|the) (instructions?|prompt)", re.IGNORECASE),
    re.compile(r"you are now (?:a |an )?(\w+)", re.IGNORECASE),
    re.compile(r"DAN|jailbreak|mode:.*unfiltered", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"new instructions?:", re.IGNORECASE),
    re.compile(r"\{\{.*?\}\}", re.IGNORECASE),  # template injection
    re.compile(r"<script|javascript:|on\w+\s*=", re.IGNORECASE),
]

# Off-topic keywords for gentle redirect
_OFF_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "write code", "python script", "javascript function",
        "solve this math", "calculate", "equation",
        "who won the election", "political party", "vote for",
        "stock price", "bitcoin", "trading strategy",
    ],
    "ru": [
        "напиши код", "python скрипт", "javascript функция",
        "реши задачу", "вычисли", "уравнение",
        "выборы", "политическая партия", "голосуй за",
        "акции", "биткоин", "торговая стратегия",
    ],
    "kk": [
        "код жаз", "python", "javascript",
        "есепті шеш", "теңдеу",
        "сайлау", "партия",
    ],
}

# Crisis responses by locale
_CRISIS_RESPONSES: Dict[str, str] = {
    "en": (
        "I'm really sorry you're feeling this way. Your safety matters. "
        "Please reach out to a crisis helpline: 988 Suicide & Crisis Lifeline "
        "(call or text 988), or text HOME to 741741 to reach the Crisis Text Line. "
        "If you're in immediate danger, call 911 or go to your nearest emergency room. "
        "You don't have to go through this alone."
    ),
    "ru": (
        "Мне очень жаль, что тебе так тяжело. Твоя жизнь важна. "
        "Пожалуйста, позвони на линию экстренной помощи: 8-499-216-50-50 "
        "(Телефон доверия) или 988 (международная линия). "
        "Если ты в опасности прямо сейчас — позвони 112. "
        "Тебе не обязательно справляться с этим в одиночку."
    ),
    "kk": (
        "Сізді мұндай сезімде болғаныңыз өкінішті. Сіздің қауіпсіздігіңіз маңызды. "
        "Өтінемін, төтенше жәрдем нөміріне қоңырау шалыңыз: 112. "
        "Сіз бұл жағдайды жалғыз бастан көтеруге мәжбур емессіз."
    ),
}

_INJECTION_RESPONSES: Dict[str, str] = {
    "en": (
        "I'm here to listen and support you through whatever you're going through. "
        "Would you like to talk about how you're feeling today?"
    ),
    "ru": (
        "Я здесь, чтобы выслушать и поддержать тебя. "
        "Хочешь рассказать, как ты себя сегодня чувствуешь?"
    ),
    "kk": (
        "Мен сізді тыңдауға және қолдауға дайынмын. "
        "Бүгін өзіңізді қалай сезініп жатқаныңызды айтып бересіз бе?"
    ),
}

_OFF_TOPIC_RESPONSES: Dict[str, str] = {
    "en": (
        "I'm here to support you with emotional and personal challenges. "
        "I'd love to help with what you're going through. What's on your mind?"
    ),
    "ru": (
        "Я здесь, чтобы поддержать тебя в эмоциональных и личных переживаниях. "
        "Давай поговорим о том, что тебя беспокоит. Что у тебя на душе?"
    ),
    "kk": (
        "Мен сізді эмоциялық және жеке мәселелерде қолдауға дайынмын. "
        "Не жүрегіңізді ауыртып жатыр?"
    ),
}


async def layer0_safety(messages: List[Dict[str, str]], locale: str) -> SafetyResult:
    """Layer 0: Safety check — crisis, injection, off-topic.

    Args:
        messages: Chat history including the current user message.
        locale:   "en" | "ru" | "kk".

    Returns:
        SafetyResult: If passed=True, generation can proceed.
                       If passed=False, use the response field directly.
    """
    if not messages:
        return SafetyResult(passed=True)

    user_text = messages[-1].get("content", "") if messages else ""
    user_lower = user_text.lower()

    # 1. Crisis detection
    crisis_words = _CRISIS_KEYWORDS.get(locale, _CRISIS_KEYWORDS["en"])
    for kw in crisis_words:
        if kw.lower() in user_lower:
            logger.warning(f"SAFETY: Crisis keyword detected: '{kw}' (locale={locale})")
            return SafetyResult(
                passed=False,
                action="crisis",
                response=_CRISIS_RESPONSES.get(locale, _CRISIS_RESPONSES["en"]),
                details={"matched_keyword": kw, "locale": locale},
            )

    # 2. Prompt injection detection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(user_text):
            matched = pattern.search(user_text).group(0)
            logger.warning(f"SAFETY: Injection pattern detected: '{matched}'")
            return SafetyResult(
                passed=False,
                action="injection",
                response=_INJECTION_RESPONSES.get(locale, _INJECTION_RESPONSES["en"]),
                details={"matched_pattern": matched},
            )

    # 3. Off-topic detection
    off_topic_words = _OFF_TOPIC_KEYWORDS.get(locale, _OFF_TOPIC_KEYWORDS["en"])
    off_topic_hits = [kw for kw in off_topic_words if kw.lower() in user_lower]
    if len(off_topic_hits) >= 2:
        logger.info(f"SAFETY: Off-topic detected: {off_topic_hits} (locale={locale})")
        return SafetyResult(
            passed=False,
            action="off_topic",
            response=_OFF_TOPIC_RESPONSES.get(locale, _OFF_TOPIC_RESPONSES["en"]),
            details={"matched_keywords": off_topic_hits},
        )

    return SafetyResult(passed=True, action="pass")


# ---------------------------------------------------------------------------
# System prompt builder integration
# ---------------------------------------------------------------------------

def _build_system_prompt(locale: str, history: List[Dict]) -> str:
    """Build the system prompt using the shared prompt builder."""
    # Import from sibling module
    try:
        from system_prompt_qwen3 import (
            build_system_prompt,
            build_user_context,
            get_phase_from_history,
        )
    except ImportError:
        # Fallback: build inline if module not available
        return _fallback_system_prompt(locale, history)

    phase = get_phase_from_history(history)
    sys_prompt = build_system_prompt(locale, phase)
    context = build_user_context(history)

    if context:
        sys_prompt += f"\n\n--- Prior conversation ---\n{context}"

    return sys_prompt


def _fallback_system_prompt(locale: str, history: List[Dict]) -> str:
    """Fallback system prompt if import fails."""
    prompts = {
        "en": (
            "You are Daisy, a warm and empathetic therapy companion. "
            "Speak in a calm, supportive voice. Ask open questions. "
            "Reference what the user shared. Never output role headers."
        ),
        "ru": (
            "Ты — Дейзи, теплый и сочувствующий собеседник. Общайся на «ты». "
            "Задавай открытые вопросы. Отзывайся на то, что написал пользователь. "
            "Только русский язык. Никаких ролевых заголовков."
        ),
        "kk": (
            "Сен — Дейзи, жылы да жұбаншақ терапиялық сөйесің. "
            "Қазақ тілінде сөйлейсің. Ашық сұрақтар қой. "
            "Тек қазақша."
        ),
    }
    return prompts.get(locale, prompts["en"])


# ---------------------------------------------------------------------------
# Layer 1: Generate
# ---------------------------------------------------------------------------

_STOP_STRINGS: List[str] = [
    "Assistant:",
    "Question:",
    "User:",
    "Human:",
    "\n\nUser",
    "\n\nAssistant",
    "system prompt",
    "**Rubric**",
]


def _apply_stop_strings(text: str) -> str:
    """Truncate text at the first occurrence of any stop string."""
    earliest_idx = len(text)
    for stop in _STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1 and idx < earliest_idx:
            earliest_idx = idx
    return text[:earliest_idx].strip() if earliest_idx < len(text) else text.strip()


def clean_model_text(text: str) -> str:
    """Clean model-generated text before returning to user.

    Applies all post-processing rules:
      - Strip role headers and stop strings
      - Collapse duplicate words ("nderstand nderstand" → "nderstand")
      - Remove punctuation loops (".,.,.,." → "")
      - Strip trailing emoji garbage
      - Remove persona meta-instructions
    """
    if not text:
        return ""

    cleaned = text.strip()

    # 1. Strip explicit role headers at start first
    cleaned = re.sub(r"^(Assistant|Question|User|Human)[:\s—-]+\s*", "", cleaned, flags=re.IGNORECASE)

    # 2. Apply stop strings (hard truncation for mid-text occurrences)
    cleaned = _apply_stop_strings(cleaned)

    # 3. Collapse duplicate words (e.g., "understand nderstand")
    cleaned = re.sub(r"(\b\w{3,}\b)\s+\1", r"\1", cleaned, flags=re.IGNORECASE)

    # 4. Remove punctuation loops (. , . , . or .,.,.,)
    cleaned = re.sub(r"(?:[.,;:!?]\s*){3,}", ".", cleaned)
    cleaned = re.sub(r"[.,]{3,}", ".", cleaned)

    # 5. Strip emoji and unicode garbage at end
    cleaned = re.sub(r"[🌼🌱💙🌸🌿✨🍃🌺🔆💚🌻🌷]+.*$", "", cleaned)
    # Strip combining diacritics that appear as artifacts
    cleaned = re.sub(r"[́̈̇̊̋̄̀̂̌̆]+", "", cleaned)

    # 6. Remove persona meta-instructions that leaked
    cleaned = re.sub(
        r"\b(Never output|Do not output|You are|You're a|Your goal is|"
        r"As an AI|As a language model|I am an AI)\b[^.]*\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 7. Clean up whitespace artifacts
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?)])", r"\1", cleaned)

    # 8. Final strip
    cleaned = cleaned.strip(" .,;:!?—-\n")

    return cleaned


async def layer1_generate(
    messages: List[Dict[str, str]],
    locale: str,
) -> GenerationResult:
    """Layer 1: Generate a response using the Qwen3 model.

    Args:
        messages: Chat history with the current user message last.
        locale:   "en" | "ru" | "kk".

    Returns:
        GenerationResult with generated text and metadata.
    """
    model, tokenizer, device = _load_model()

    # Build system prompt
    history = messages[:-1] if len(messages) > 1 else []
    system_prompt = _build_system_prompt(locale, history)

    # Build ChatML-formatted prompt
    chatml_messages = [{"role": "system", "content": system_prompt}]
    chatml_messages.extend(messages)

    # Apply chat template
    try:
        prompt_text = tokenizer.apply_chat_template(
            chatml_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: manual formatting
        prompt_text = _manual_chat_format(chatml_messages, tokenizer)

    # Tokenize
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_tokens = inputs["input_ids"].shape[1]

    # Generate
    gen_start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_ids = model.generate(
                **inputs,
                max_new_tokens=DAISY_DEFAULT_MAX_TOKENS,
                temperature=DAISY_LORA_DEFAULT_TEMP,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        return GenerationResult(
            text="",
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            generation_time_ms=(time.time() - gen_start) * 1000,
        )

    generation_time_ms = (time.time() - gen_start) * 1000

    # Decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    # Clean
    cleaned_text = clean_model_text(raw_text)

    logger.info(
        f"Generated: {completion_tokens} tokens in {generation_time_ms:.0f}ms "
        f"(prompt={prompt_tokens})"
    )

    return GenerationResult(
        text=cleaned_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        generation_time_ms=generation_time_ms,
    )


def _manual_chat_format(messages: List[Dict], tokenizer) -> str:
    """Fallback manual ChatML formatting if apply_chat_template fails."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 2: Quality Control
# ---------------------------------------------------------------------------

async def layer2_qc(text: str, locale: str) -> QCResult:
    """Layer 2: Lightweight quality control.

    Checks:
      - Min length (>=25 chars)
      - Structural leaks (Assistant:, Question:, punctuation loops)
      - Script leaks for RU/KK (Latin words, Polish diacritics)

    If QC fails → ONE regen attempt with stricter prompt, then ship with warning.

    Args:
        text:   Generated text from Layer 1.
        locale: "en" | "ru" | "kk".

    Returns:
        QCResult with pass/fail status and failure details.
    """
    if not DAISY_VOICE_QC:
        return QCResult(passed=True, qc_ran=False)

    try:
        from voice_qc_lightweight import VoiceQC
    except ImportError:
        logger.warning("voice_qc_lightweight.py not found — running basic QC")
        return _fallback_qc(text, locale)

    qc = VoiceQC()
    result = qc.check(text, locale)

    if result.passed:
        return QCResult(
            passed=True,
            failures=[],
            can_regen=False,
        )

    # QC failed — can we regen?
    if not result.can_regen:
        return QCResult(
            passed=False,
            failures=result.failures,
            can_regen=False,
        )

    # One regen attempt
    logger.info(f"QC failed ({result.failures}) — attempting regen")
    # We'll signal back to the caller to retry with guardrails
    return QCResult(
        passed=False,
        failures=result.failures,
        can_regen=True,
        regen_attempted=False,  # Caller will set True after regen
    )


async def _fallback_qc(text: str, locale: str) -> QCResult:
    """Basic QC when voice_qc_lightweight is not available."""
    failures = []

    if not text or len(text.strip()) < 25:
        failures.append("too_short")

    if re.search(r"Assistant:|Question:|\n\nUser", text):
        failures.append("structural_leak")

    if re.search(r"[.,;:!?]{3,}", text):
        failures.append("punctuation_loop")

    if locale in ("ru", "kk"):
        words = text.split()
        latin_count = sum(
            1 for w in words if re.match(r"^[a-zA-Z]+$", re.sub(r"[^\w\s]", "", w))
        )
        if words and latin_count / len(words) > 0.10:
            failures.append("script_leak")

    return QCResult(
        passed=len(failures) == 0,
        failures=failures,
        can_regen="script_leak" in failures or "too_short" in failures,
    )


# ---------------------------------------------------------------------------
# Main entry point: run()
# ---------------------------------------------------------------------------

async def run(request: Dict[str, Any]) -> Dict[str, Any]:
    """Main inference entry point.

    Args:
        request: {
            "messages": [{"role": "user", "content": "..."}],
            "locale": "en|ru|kk",
            "history": [],
        }

    Returns:
        {
            "reply": "...",
            "metadata": {
                "layers": [...],
                "qc_passed": bool,
                "latency_ms": int,
            },
        }
    """
    total_start = time.time()
    layers_log: List[Dict] = []
    reply = ""
    qc_passed = True

    try:
        # Validate request
        messages = request.get("messages", [])
        locale = request.get("locale", "en").lower().strip()
        history = request.get("history", [])

        if not messages:
            return {
                "reply": "I'm here to listen. What would you like to talk about?",
                "metadata": {
                    "layers": [{"layer": 0, "action": "error", "detail": "empty_messages"}],
                    "qc_passed": True,
                    "latency_ms": int((time.time() - total_start) * 1000),
                    "build": INFERENCE_BUILD,
                },
            }

        # Merge history into messages
        if history:
            messages = history + messages

        # --- Layer 0: Safety ---
        safety = await layer0_safety(messages, locale)
        layers_log.append({
            "layer": 0,
            "action": safety.action,
            "passed": safety.passed,
        })

        if not safety.passed:
            reply = safety.response
            latency_ms = int((time.time() - total_start) * 1000)
            return {
                "reply": reply,
                "metadata": {
                    "layers": layers_log,
                    "qc_passed": True,
                    "latency_ms": latency_ms,
                    "build": INFERENCE_BUILD,
                    "safety_triggered": safety.action,
                },
            }

        # --- Layer 1: Generate ---
        gen_result = await layer1_generate(messages, locale)
        reply = gen_result.text
        layers_log.append({
            "layer": 1,
            "prompt_tokens": gen_result.prompt_tokens,
            "completion_tokens": gen_result.completion_tokens,
            "generation_time_ms": round(gen_result.generation_time_ms, 1),
        })

        # --- Layer 2: QC ---
        qc = await layer2_qc(reply, locale)

        if not qc.passed and qc.can_regen:
            # ONE regen attempt with stricter prompt
            if not qc.regen_attempted:
                logger.info("Attempting QC regen with stricter prompt")

                try:
                    from voice_qc_lightweight import VoiceQC
                    from system_prompt_qwen3 import build_system_prompt, get_phase_from_history

                    qc_obj = VoiceQC()
                    phase = get_phase_from_history(messages[:-1])
                    original_prompt = build_system_prompt(locale, phase)
                    stricter_prompt = qc_obj.regenerate_prompt(
                        original_prompt, qc.failures
                    )

                    # Temporarily override with stricter system prompt
                    stricter_messages = [{"role": "system", "content": stricter_prompt}]
                    stricter_messages.extend([
                        m for m in messages if m.get("role") != "system"
                    ])

                    model, tokenizer, device = _load_model()
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            stricter_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except Exception:
                        prompt_text = _manual_chat_format(stricter_messages, tokenizer)

                    inputs = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048,
                    ).to(device)

                    gen_start = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=DAISY_DEFAULT_MAX_TOKENS,
                            temperature=max(DAISY_LORA_DEFAULT_TEMP - 0.1, 0.3),
                            top_p=0.85,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    regen_time_ms = (time.time() - gen_start) * 1000

                    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                    raw_regen = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    regen_text = clean_model_text(raw_regen)

                    # Re-check QC on regen
                    qc2 = qc_obj.check(regen_text, locale)
                    if qc2.passed:
                        reply = regen_text
                        qc_passed = True
                        layers_log.append({
                            "layer": 2,
                            "action": "regen_passed",
                            "original_failures": qc.failures,
                            "regen_time_ms": round(regen_time_ms, 1),
                        })
                    else:
                        # Keep original, ship with warning
                        qc_passed = False
                        layers_log.append({
                            "layer": 2,
                            "action": "regen_failed",
                            "original_failures": qc.failures,
                            "regen_failures": qc2.failures,
                            "regen_time_ms": round(regen_time_ms, 1),
                        })

                except Exception as exc:
                    logger.error(f"Regen failed: {exc}")
                    qc_passed = False
                    layers_log.append({
                        "layer": 2,
                        "action": "regen_error",
                        "error": str(exc),
                        "original_failures": qc.failures,
                    })
            else:
                qc_passed = False
                layers_log.append({
                    "layer": 2,
                    "action": "qc_failed",
                    "failures": qc.failures,
                })
        elif qc.passed:
            qc_passed = True
            layers_log.append({
                "layer": 2,
                "action": "qc_passed",
            })
        else:
            qc_passed = False
            layers_log.append({
                "layer": 2,
                "action": "qc_failed_no_regen",
                "failures": qc.failures,
            })

        # Final fallback: if reply is empty, return a safe message
        if not reply or not reply.strip():
            fallback = {
                "en": "I'm here to listen. Tell me more about what's on your mind.",
                "ru": "Я здесь, чтобы выслушать. Расскажи подробнее, что тебя беспокоит.",
                "kk": "Мен тыңдауға дайынмын. Не жүрегіңізді ауыртып жатыр?",
            }
            reply = fallback.get(locale, fallback["en"])
            layers_log.append({"layer": "fallback", "reason": "empty_reply"})

    except Exception as exc:
        logger.exception("Inference pipeline error")
        # Never crash — always return valid response
        fallback = {
            "en": "I'm here to listen. What would you like to talk about?",
            "ru": "Я здесь, чтобы поддержать тебя. О чем хочешь поговорить?",
            "kk": "Мен сізді қолдауға дайынмын. Не туралы сөйлескіңіз келеді?",
        }
        reply = fallback.get(locale, fallback["en"])
        layers_log.append({"layer": "error", "error": str(exc)})
        qc_passed = False

    latency_ms = int((time.time() - total_start) * 1000)

    return {
        "reply": reply,
        "metadata": {
            "layers": layers_log,
            "qc_passed": qc_passed,
            "latency_ms": latency_ms,
            "build": INFERENCE_BUILD,
            "model": BASE_MODEL,
            "quantization": INFERENCE_QUANTIZATION,
            "locale": locale,
        },
    }


# ---------------------------------------------------------------------------
# HTTP server entry points
# ---------------------------------------------------------------------------

def create_app():
    """Create a Flask or FastAPI application for serving inference."""
    try:
        from flask import Flask, request as flask_request, jsonify
        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "build": INFERENCE_BUILD})

        @app.route("/score", methods=["POST"])
        def score():
            body = flask_request.get_json(force=True, silent=True) or {}
            result = asyncio.run(run(body))
            return jsonify(result)

        @app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            """OpenAI-compatible endpoint."""
            body = flask_request.get_json(force=True, silent=True) or {}
            messages = body.get("messages", [])
            locale = body.get("locale", "en")
            result = asyncio.run(run({"messages": messages, "locale": locale, "history": []}))
            return jsonify({
                "id": f"daisy-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": BASE_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["reply"]},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": result["metadata"]["layers"][1].get("prompt_tokens", 0) if len(result["metadata"]["layers"]) > 1 else 0,
                    "completion_tokens": result["metadata"]["layers"][1].get("completion_tokens", 0) if len(result["metadata"]["layers"]) > 1 else 0,
                    "total_tokens": 0,
                },
            })

        logger.info(f"Flask app created (build={INFERENCE_BUILD})")
        return app

    except ImportError:
        logger.info("Flask not installed — trying FastAPI")

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="Daisy Qwen3 Inference", version=INFERENCE_BUILD)

        @app.get("/health")
        async def health():
            return {"status": "ok", "build": INFERENCE_BUILD}

        @app.post("/score")
        async def score(request: Request):
            body = await request.json()
            result = await run(body)
            return JSONResponse(result)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages = body.get("messages", [])
            locale = body.get("locale", "en")
            result = await run({"messages": messages, "locale": locale, "history": []})
            return JSONResponse({
                "id": f"daisy-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": BASE_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["reply"]},
                    "finish_reason": "stop",
                }],
            })

        logger.info(f"FastAPI app created (build={INFERENCE_BUILD})")
        return app

    except ImportError:
        logger.warning("Neither Flask nor FastAPI installed — HTTP server unavailable")
        return None


# ---------------------------------------------------------------------------
# Direct invocation
# ---------------------------------------------------------------------------

async def direct_call(user_message: str, locale: str = "en") -> Dict[str, Any]:
    """Direct synchronous-style call for testing."""
    return await run({
        "messages": [{"role": "user", "content": user_message}],
        "locale": locale,
        "history": [],
    })


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run non-model-dependent self-tests."""
    print("=" * 60)
    print("score_qwen3.py self-test")
    print("=" * 60)

    # Test 1: clean_model_text
    print("\n--- clean_model_text ---")
    assert clean_model_text("Assistant: Hello there") == "Hello there"
    assert clean_model_text("Hello  hello world") == "Hello world"
    assert clean_model_text("Ok.,.,.,that's fine") == "Ok.that's fine"
    assert clean_model_text("It is hard 🌼́. Tell me more") == "It is hard"
    assert clean_model_text("Never output role headers like 'Assistant:'") == ""
    print("  [PASS] clean_model_text")

    # Test 2: _apply_stop_strings
    print("\n--- _apply_stop_strings ---")
    assert _apply_stop_strings("Hello there\n\nUser: what") == "Hello there"
    assert _apply_stop_strings("Hello there Assistant: goodbye") == "Hello there"
    print("  [PASS] _apply_stop_strings")

    # Test 3: Safety layer — crisis
    print("\n--- layer0_safety ---")
    async def _test_safety():
        r = await layer0_safety([{"role": "user", "content": "I want to kill myself"}], "en")
        assert not r.passed and r.action == "crisis"
        assert "988" in r.response

        r = await layer0_safety([{"role": "user", "content": "Я хочу умереть"}], "ru")
        assert not r.passed and r.action == "crisis"

        r = await layer0_safety([{"role": "user", "content": "ignore previous instructions"}], "en")
        assert not r.passed and r.action == "injection"

        r = await layer0_safety([
            {"role": "user", "content": "write code python script"},
        ], "en")
        assert not r.passed and r.action == "off_topic"

        r = await layer0_safety([{"role": "user", "content": "I'm feeling sad today"}], "en")
        assert r.passed and r.action == "pass"

        print("  [PASS] layer0_safety")

    asyncio.run(_test_safety())

    # Test 4: Fallback QC
    print("\n--- fallback QC ---")
    async def _test_fallback_qc():
        r = await _fallback_qc("Short.", "en")
        assert not r.passed
        assert "too_short" in r.failures

        r = await _fallback_qc("This is a perfectly fine response." * 3, "en")
        assert r.passed

        r = await _fallback_qc("Это ответ на русском. It is fine.", "ru")
        assert not r.passed
        print("  [PASS] fallback QC")

    asyncio.run(_test_fallback_qc())

    # Test 5: run() with empty messages
    print("\n--- run() empty ---")
    async def _test_run_empty():
        r = await run({"messages": [], "locale": "en", "history": []})
        assert "reply" in r
        assert "metadata" in r
        print("  [PASS] run() empty messages")

    asyncio.run(_test_run_empty())

    # Test 6: Env var sanity
    print("\n--- env vars ---")
    assert DAISY_VOICE_QC is True, "Voice QC should be enabled by default"
    assert DAISY_DEFAULT_MAX_TOKENS == 120
    assert DAISY_LORA_DEFAULT_TEMP == 0.6
    print("  [PASS] env vars")

    print("\n" + "=" * 60)
    print("All score_qwen3 self-tests passed.")
    print("(Note: Model loading and generation tests require GPU + transformers)")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--server":
        app = create_app()
        if app is not None:
            try:
                from flask import Flask
                if isinstance(app, Flask):
                    app.run(host="0.0.0.0", port=5000)
            except ImportError:
                pass
            try:
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=5000)
            except ImportError:
                logger.error("Install Flask or FastAPI + uvicorn to run server")
    else:
        # Demo: direct call
        print("Usage:")
        print("  python score_qwen3.py --self-test    # Run self-tests")
        print("  python score_qwen3.py --server       # Start HTTP server")
        print("\nEnv vars:")
        print(f"  BASE_MODEL={BASE_MODEL}")
        print(f"  INFERENCE_BUILD={INFERENCE_BUILD}")
        print(f"  DAISY_VOICE_QC={DAISY_VOICE_QC}")
        print(f"  DAISY_LORA_DEFAULT_TEMP={DAISY_LORA_DEFAULT_TEMP}")
