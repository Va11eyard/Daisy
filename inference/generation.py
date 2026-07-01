"""Text generation + output cleanup.

Slimmed for the 5-layer pipeline: the old regex post-hoc patch zoo (degenerate /
brief / near-duplicate / casual-greeting detectors and retry loops) is gone.
Structural validation now lives in voice_qc.py (Layer 3) and confidence.py
(Layer 4). This module keeps only:

  generate_reply()            - single generation pass, optional streaming + logprobs
  generate_reply_stream()     - token generator for SSE-style hosting
  postprocess_model_response  - clean + trim to a complete sentence
  clean_model_text            - OCR/junk normalization
  trim_to_complete_sentence   - cut trailing partial/garbage sentences
  fallback_reply              - eval/ablation only; production uses best model candidate in score.py
  extract_json_object         - used by report_handlers.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

_DEGEN_PUNCT_CLUSTER = re.compile(
    r"(\.\s*){3,}|(\?\s*){2,}|\.\.\s*\.\s*\?|(?:\.\s+){2,}\?"
    r"|(?:[?:]\s*[\.\?:]\s*){2,}"
)
_SPACED_PUNCT_GARBAGE = re.compile(r"(?:[?:]\s*[\.\?:]\s*){2,}")
_OCR_JUNK_CHARS = "\u00b4\u02c6\u00b8\u00ba\u017d\u017a\u0142\u015b"
_TRAILING_EMOJI_PUNCT = re.compile(
    r"([\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\uFE0F\u2665\u2661]+)"
    rf"(?:\s*[{re.escape(_OCR_JUNK_CHARS)}\u0300-\u036f]*)?\s*[\.\?!…]+\s*$"
)

_CYR = r"а-яёА-ЯЁ"
_CYR_ACUTE = re.compile(rf"([{_CYR}])[{re.escape(_OCR_JUNK_CHARS)}]+(?=[{_CYR}])")

_TOPIC_BREAKUP = re.compile(
    r"\b(broke up|breakup|break up|left me|dumped|miss him|miss her|расстал|бросил|бросила)\b",
    re.I,
)
_TOPIC_SOMATIC = re.compile(
    r"\b(body|chest|stomach|throat|emptiness|heart area|тело|грудь|горло|пустота)\b",
    re.I,
)
_TOPIC_STRESS = re.compile(
    r"\b(stress|stressed|overwhelm|overwhelmed|burnout|burned out)\b|стресс\w*|перегруз\w*|выгоран\w*",
    re.I,
)
_TOPIC_CLARITY = re.compile(
    r"\b(sort out|untangle|organize|clarify).{0,24}\bthoughts?\b|\bhelp me (?:sort|think)\b|"
    r"разобраться.{0,24}мысл|помоги.{0,16}мысл",
    re.I,
)
_TOPIC_ANXIETY = re.compile(
    r"\b(anxious|anxiety|worried|worry|nervous|panick?y|тревог\w*|беспоко\w*)",
    re.I,
)
_TOPIC_WORK = re.compile(
    r"\b(ceo|boss|manager|coworker|work|workplace|job|fired|yelling|raging|"
    r"breaking the site|broke the site|начальник|руководител|коллег|работ\w*|сайт)",
    re.I,
)

_VARIED_FALLBACKS_RU = (
    "Я рядом. Что сейчас на уме — можно начать с чего угодно.",
    "Я рядом. Что сейчас сильнее — тревога, усталость или что-то ещё?",
    "Давай попробуем иначе: что произошло непосредственно перед тем, как стало тяжело?",
    "Слышу, что тебе сейчас непросто. Что из этого давит сильнее всего?",
)
_VARIED_FALLBACKS_KK = (
    "Мен осындамын. Ойыңда не бар — кез келген нәрсемен бастауға болады.",
    "Мен осындамын. Қазір не күштірек — мазасыздық па, шаршау ма?",
)
_VARIED_FALLBACKS_EN = (
    "I'm here. What feels strongest right now — anxiety, fatigue, or something else?",
    "I hear that things feel heavy right now. What feels most pressing in this moment?",
    "Hey — I'm glad you're here. What's been on your mind lately?",
)


def _normalize_for_compare(text: str) -> str:
    t = clean_model_text(text).lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


_META_HEADER_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^in plain language:\s*", re.I),
    re.compile(r"^here'?s a careful reading:\s*", re.I),
    re.compile(r"^one way to (?:think about|approach) (?:it|this):\s*", re.I),
)
_RUBRIC_LEAK = re.compile(
    r"[-\[]?\[?\s*open question\s*\]?\]?.*$",
    re.I | re.DOTALL,
)
_PERSONA_TONE_LEAK = re.compile(r"«[^»]+»\s*")
_TRAINING_REGISTER_LEAK = re.compile(
    r"\b(?:example tone|register reference|prefer|avoid):\s*",
    re.I,
)
# Corruption safety net — targets genuinely broken generation output (duplicated
# fragments, self-inserted "Question:" scaffolding, mid-string orphan emoji), NOT
# length. Runs unconditionally regardless of the aggressive_trim setting, because
# it removes garbage rather than legitimate multi-sentence replies.
_DUP_WORD_RUN = re.compile(r"\b(\w{3,})(?:\s+\1\b)+", re.I)
# Real English/Russian words rarely exceed ~18 letters; a longer unbroken alpha
# run is almost always a decode-corruption blob (missing internal spaces from
# concatenated/garbled subword tokens), e.g. "asurethattheyknowyouarelistening...".
_LONG_GARBLED_RUN = re.compile(r"\b[a-zà-ÿа-яё]{19,}\b", re.I)
_MISSING_SPACE_AFTER_PUNCT = re.compile(r"([.?!])(?=[A-Za-zА-Яа-я])")
_MIDTEXT_QUESTION_HEADER = re.compile(r"(?<=\S)\s*q?uestion:\s*", re.I)
_MID_EMOJI_ORPHAN = re.compile(
    r"\s*[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\uFE0F\u2665\u2661]+\s+(?=[a-z]{2,12}\b)"
)
# A single dangling lowercase fragment left as the very last token right after a
# completed sentence (typically exposed once emoji-orphan/dup-word stripping above
# removes what was surrounding it) — real replies don't trail off with one bare
# lowercase word and nothing else.
_TRAILING_ORPHAN_FRAGMENT = re.compile(r"(?<=[.?!])\s+[a-z]{2,12}$")
# Trailing decorative emoji followed by a short foreign/garbled fragment (any
# script, e.g. "🌱 ẩm."), optionally closed by its own stray punctuation, at the
# very end of the reply — the real sentence already ended before the emoji.
_TRAILING_EMOJI_GARBLE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\uFE0F\u2665\u2661]+"
    r"\s+[^\s.!?]{1,14}[.!?…]*\s*$"
)
# Small CJK fragments (below reply_language.py's retry threshold) still read as
# corruption when they trail off an en/ru/kk reply — strip unconditionally.
_CJK_FRAGMENT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]+")
# The model occasionally narrates its own behavioral instructions mid-reply
# instead of paraphrasing them into a persona quote (the persona/register leaks
# above catch the quoted/labeled forms; this catches the plain-English self-talk
# form, e.g. "Please remember to stay flexible ... Adapt to what the person
# needs in that moment.").
_META_INSTRUCTION_LEAK = re.compile(
    r"\b(?:remember to (?:stay|be|adapt)\b|stay flexible\b|a?\s*structured reply\b|"
    r"a reflective (?:one|reply)\b|adapt to what the person needs\b|"
    r"(?:the )?person may need space\b|validate their experience\b|"
    r"reflect back what (?:was|they) said\b|stay focused on what they shared\b|"
    r"never use unsolicited\b|rules? output only\b|open-ended invitations to share\b)",
    re.I,
)


def _strip_meta_instruction_leak(text: str) -> str:
    m = _META_INSTRUCTION_LEAK.search(text)
    if not m:
        return text
    head = text[: m.start()]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= 0:
        return head[: cut + 1].strip()
    return head.strip()


# Occasionally generation degenerates into echoing the literal system prompt
# (rubric / banned-phrase catalog / scope rules — see system_prompt.py) instead
# of a reply. These marker strings are near-unique fingerprints of our own
# instructions and essentially never occur in genuine therapeutic dialogue, so
# any match means "everything from here on is prompt leakage, not a reply."
_SYSTEM_PROMPT_ECHO = re.compile(
    r"\b(?:NEVER USE|NEVER CLOSE WITH|NEVER OPEN WITH|NEVER START WITH|NEVER EXPLAIN|"
    r"NEVER REPHRASE|PREFER PRECISE LANGUAGE|SCOPE)\s*:|"
    r"\bAlways respond in English\.|\bPlease select one style\s*:|"
    r"\bPlease choose a language for response\s*:|"
    r"\bgradable system output rules\b",
    re.I,
)


def _strip_system_prompt_echo(text: str) -> str:
    m = _SYSTEM_PROMPT_ECHO.search(text)
    if not m:
        return text
    head = text[: m.start()]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= 0:
        return head[: cut + 1].strip()
    return head.strip()


# Persona meta-instruction that leaks verbatim when flexible persona is active.
_PERSONA_META_QUOTE = re.compile(
    r"«\s*Подстраивайся под текущую потребность человека\.?\s*»",
    re.I,
)
# Cyrillic reply complete, then Latin/extended-Latin junk (English/Polish drift).
_CYRILLIC_THEN_LATIN_TAIL = re.compile(
    r"^([\s\S]*?[\u0400-\u04ff\u0451\u0401][\s\S]*?[.?!])\s+(?=[A-Za-z\u0100-\u017f])"
)


def _strip_cyrillic_reply_leak(text: str, lang: str) -> str:
    """Trim Latin/Polish tails and persona meta-quotes from RU/KK replies."""
    if lang not in ("ru", "kk") or not text:
        return text
    t = _PERSONA_META_QUOTE.sub("", text).strip()
    m = _CYRILLIC_THEN_LATIN_TAIL.search(t)
    if m:
        head = m.group(1).strip()
        if len(head) >= 25:
            t = head
    return t


def _strip_generation_corruption(text: str) -> str:
    """Remove duplicated-fragment / self-inserted-header corruption artifacts."""
    if not text:
        return text
    t = _MISSING_SPACE_AFTER_PUNCT.sub(r"\1 ", text)
    t = _CJK_FRAGMENT.sub("", t)
    t = _LONG_GARBLED_RUN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\.\s*\.", ".", t)
    t = _DUP_WORD_RUN.sub(r"\1", t)
    m = _MIDTEXT_QUESTION_HEADER.search(t)
    if m:
        t = t[: m.start()].strip()
    t = _MID_EMOJI_ORPHAN.sub(" ", t)
    t = _TRAILING_EMOJI_GARBLE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _TRAILING_ORPHAN_FRAGMENT.sub("", t).strip()
    t = _strip_meta_instruction_leak(t)
    return t


def clean_model_text(text: str, *, lang: str = "ru") -> str:
    """Remove OCR junk (acute accents between Cyrillic words) and normalize whitespace."""
    if not text or not text.strip():
        return text
    t = text.strip()
    for pat in _META_HEADER_PREFIXES:
        t = pat.sub("", t).strip()
    t = _CYR_ACUTE.sub(r"\1 ", t)
    t = re.sub(rf"[{re.escape(_OCR_JUNK_CHARS)}]+", " ", t)
    if re.search(rf"[{_CYR}]", t):
        t = t.replace("\u3002", ".").replace("\uff0c", ",").replace("\uff1f", "?").replace("\uff01", "!")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"([\.\?!…]){2,}", r"\1", t)
    if _SPACED_PUNCT_GARBAGE.search(t):
        t = _SPACED_PUNCT_GARBAGE.sub("?", t)
        t = re.sub(r"\s+", " ", t).strip()
    if _TRAILING_EMOJI_PUNCT.search(t):
        t = _TRAILING_EMOJI_PUNCT.sub("", t).strip()
    t = _strip_system_prompt_echo(t)
    t = _PERSONA_TONE_LEAK.sub("", t)
    t = _RUBRIC_LEAK.sub("", t).strip()
    if _TRAINING_REGISTER_LEAK.search(t):
        t = t.split("Example tone:", 1)[0].strip()
        t = t.split("REGISTER REFERENCE:", 1)[0].strip()
    t = _strip_generation_corruption(t)
    if lang in ("ru", "kk"):
        t = _strip_cyrillic_reply_leak(t, lang)
    return t


def _aggressive_sentence_trim() -> bool:
    return os.environ.get("DAISY_AGGRESSIVE_TRIM", "").lower() in ("1", "true", "yes")


def trim_to_complete_sentence(text: str, *, aggressive: bool | None = None) -> str:
    if not text or not text.strip():
        return text
    text = text.strip()
    aggressive = _aggressive_sentence_trim() if aggressive is None else aggressive
    boundaries: list[int] = []
    for end in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        start = 0
        while True:
            idx = text.find(end, start)
            if idx < 0:
                break
            boundaries.append(idx + len(end))
            start = idx + len(end)
    if not boundaries:
        return text
    boundaries = sorted(set(boundaries))

    def _bad_tail(tail: str) -> bool:
        if not tail:
            return False
        if aggressive and len(tail) > 40:
            return True
        return bool(re.match(r"^[\.\?\s!…]+$", tail) or _DEGEN_PUNCT_CLUSTER.search(tail))

    for pos in boundaries:
        head = text[:pos].strip()
        tail = text[pos:].strip()
        if not tail:
            return head
        if _bad_tail(tail):
            if aggressive and len(head) >= 30:
                return head
            continue
    return text


def postprocess_model_response(
    response: str,
    reply_lang: str,
    *,
    aggressive_trim: bool | None = None,
) -> tuple[str, bool]:
    """Clean, trim; return (text, sanitized_flag)."""
    raw = response
    cleaned = clean_model_text(response, lang=reply_lang)
    if aggressive_trim is None:
        aggressive_trim = _aggressive_sentence_trim()
    trimmed = trim_to_complete_sentence(cleaned, aggressive=aggressive_trim)
    return trimmed, trimmed != raw.strip()


def fallback_reply(
    reply_lang: str,
    *,
    avoid: str | None = None,
    also_avoid: str | None = None,
    avoid_recent: list[str] | None = None,
    user_message: str = "",
    history_snippet: str = "",
) -> str:
    """Topic-template replies for eval/ablation only — not used in production score.py."""
    context = f"{user_message} {history_snippet}".lower()

    if reply_lang == "ru":
        if _TOPIC_BREAKUP.search(context):
            pool = [
                "Расставание — это особая боль: человек есть, но уже недоступен. "
                "Что сейчас ощущается сильнее — пустота или то, что хочется ему сказать?",
                "Слышу, что скучаешь по нему. Что именно ты потеряла — человека или то, каким было ваше будущее?",
            ]
        elif _TOPIC_SOMATIC.search(context):
            pool = [
                "Эта тяжесть в груди — настоящая, тело держит то, что словами ещё не выразить. "
                "Как ощущается — больше как вес или как пустота?",
                "Что-то отозвалось в теле прямо сейчас. Похоже больше на давление или на отсутствие?",
            ]
        elif _TOPIC_STRESS.search(context):
            pool = [
                "Стресс часто копится незаметно, пока не становится слишком громким. "
                "Обычно за общим словом «стресс» стоит что-то конкретное. "
                "Что сейчас давит сильнее всего — темп, неопределённость или что-то ещё?",
            ]
        elif _TOPIC_CLARITY.search(context):
            pool = [
                "Когда мысли перепутаны, полезно замедлиться и взять одну нить за раз. "
                "Какая мысль сейчас крутится громче всего — сама тревога или то, чего вы из-за неё боитесь?",
            ]
        elif _TOPIC_ANXIETY.search(context):
            pool = [
                "Тревога часто накручивается быстрее, чем успеваешь её назвать — тело реагирует раньше слов. "
                "Сейчас она больше про будущее, про то, что уже случилось, или про оба сразу?",
            ]
        elif _TOPIC_WORK.search(context):
            pool = [
                "Когда на работе на тебя давят, легко смешать страх за результат с ощущением, что тебя не видят как человека. "
                "Что сейчас острее — страх последствий или сам тон, с которым к тебе обратились?",
            ]
        else:
            pool = list(_VARIED_FALLBACKS_RU) + [
                "Я слышу, что сейчас тяжело. Расскажи, что больше всего давит в этот момент?",
            ]
    elif reply_lang == "kk":
        pool = list(_VARIED_FALLBACKS_KK) + [
            "Мен естідім, қазір қиын. Қазіргі сәтте не ең ауыр сезінесіз?",
        ]
    else:
        if _TOPIC_BREAKUP.search(context):
            pool = [
                "Breakups leave a particular kind of hollow — the person is gone but the space they took up isn't. "
                "What's sitting heaviest right now, the missing or the anger?",
                "Missing someone after a breakup is its own grief. What part of him do you find yourself missing most?",
            ]
        elif _TOPIC_SOMATIC.search(context):
            pool = [
                "That heaviness in your chest is real — the body holds what words can't yet carry. "
                "What does it feel like it's protecting?",
                "Something is sitting in your body right now. Does it feel more like weight, or more like absence?",
            ]
        elif _TOPIC_STRESS.search(context):
            pool = [
                "Stress often builds quietly before you name it — like a hum you stop noticing until it gets loud. "
                "When someone brings it up, it usually means something specific is wearing them down. "
                "What part of stress is pressing on you most right now — the pace, the uncertainty, or something else?",
            ]
        elif _TOPIC_CLARITY.search(context):
            pool = [
                "When thoughts feel tangled, it often helps to slow down and pick one thread at a time. "
                "Which thought is looping loudest right now — the worry itself, or what you're afraid it means?",
            ]
        elif _TOPIC_ANXIETY.search(context):
            pool = [
                "Anxiety often ramps up faster than you can name it — the body reacts before the story is clear. "
                "Right now, is it pulling you toward the future, replaying something that already happened, or both?",
            ]
        elif _TOPIC_WORK.search(context):
            pool = [
                "When pressure lands at work, it's easy to mix fear about outcomes with feeling unseen as a person. "
                "What's sharper right now — worry about consequences, or the way you were spoken to?",
            ]
        else:
            pool = list(_VARIED_FALLBACKS_EN) + [
                "I hear that things feel heavy right now. What feels most pressing in this moment?",
            ]

    avoids = {
        _normalize_for_compare(a)
        for a in (*(avoid_recent or []), avoid, also_avoid)
        if a and a.strip()
    }
    for candidate in pool:
        if _normalize_for_compare(candidate) not in avoids:
            return candidate
    return pool[0]


_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"
_THINK_START_ID_FALLBACK = 151667  # Qwen3 <think> token id
_THINK_END_ID_FALLBACK = 151668  # Qwen3 </think> token id


def _special_id(tokenizer: "PreTrainedTokenizer", token: str, fallback: int) -> int:
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    return fallback


def split_think_tokens(
    tokenizer: "PreTrainedTokenizer", new_token_ids: list[int]
) -> tuple[list[int], list[int]]:
    """Split generated token ids into (answer_ids, thinking_ids).

    Qwen3 reasoning emits <think>...</think> before the visible answer. We split on
    the last </think> so the visible reply never contains the chain of thought.
    If reasoning was opened (<think>) but never closed (token budget ran out), the
    answer is treated as empty so the caller's quality floor can regenerate — this
    prevents leaking raw reasoning to the user. If neither tag appears, the model
    answered directly and the whole sequence is the answer.
    """
    end_id = _special_id(tokenizer, _THINK_END_TOKEN, _THINK_END_ID_FALLBACK)
    start_id = _special_id(tokenizer, _THINK_START_TOKEN, _THINK_START_ID_FALLBACK)
    last_end = -1
    for i, tid in enumerate(new_token_ids):
        if tid == end_id:
            last_end = i
    if last_end >= 0:
        return new_token_ids[last_end + 1 :], new_token_ids[: last_end + 1]
    if any(tid == start_id for tid in new_token_ids):
        return [], new_token_ids
    return new_token_ids, []


# Once the model has said its piece, it sometimes keeps sampling into
# hallucinated next-turn content or regurgitated pretraining-web-scrape text
# (a fresh "User:"/"用户" turn, page footers, doc/tool-call tags) instead of
# emitting EOS. None of that ever legitimately appears inside one therapeutic
# reply, so we hard-stop generation the moment any of it starts.
_GENERATION_STOP_STRINGS = [
    "\n\n",
    "用户",
    "\nUser:",
    "\nuser:",
    "<tool_call>",
    "<|im_start|>",
    "Powered by",
    "Impressum",
]


def _gen_kwargs(
    tokenizer: "PreTrainedTokenizer",
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    repetition_penalty: float | None,
    min_new_tokens: int | None,
) -> dict:
    penalty = repetition_penalty if repetition_penalty is not None else 1.15
    kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": penalty,
        "stop_strings": _GENERATION_STOP_STRINGS,
        "tokenizer": tokenizer,
    }
    if min_new_tokens is not None and min_new_tokens > 0:
        kwargs["min_new_tokens"] = min_new_tokens
    if do_sample:
        # Qwen3 thinking-mode sampling (greedy decoding is discouraged in thinking mode).
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95
        kwargs["top_k"] = 20
        kwargs["do_sample"] = True
    else:
        kwargs["do_sample"] = False
    return kwargs


def generate_reply(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    *,
    do_sample: bool = True,
    repetition_penalty: float | None = None,
    min_new_tokens: int | None = None,
    return_logprobs: bool = False,
):
    """Single generation pass with Qwen3 reasoning.

    The model thinks inside <think>...</think>; we return only the visible answer.
    When return_logprobs=True, returns (answer_text, list[float] | None) where the
    log-probabilities cover the answer span only (the chain of thought is excluded).
    """
    import torch

    floor = int(os.environ.get("DAISY_GEN_MIN_MAX_NEW_TOKENS", "32"))
    max_new_tokens = max(floor, max_new_tokens)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = _gen_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        min_new_tokens=min_new_tokens,
    )
    if return_logprobs:
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs["output_scores"] = True

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    if return_logprobs:
        sequences = out.sequences
        new_token_ids = sequences[0][input_len:].tolist()
        answer_ids, think_ids = split_think_tokens(tokenizer, new_token_ids)
        text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        token_logprobs: list[float] | None = None
        try:
            transition = model.compute_transition_scores(
                sequences, out.scores, normalize_logits=True
            )
            scores = [float(x) for x in transition[0].tolist()]
            token_logprobs = scores[len(think_ids) :] if think_ids else scores
        except Exception:
            logger.exception("compute_transition_scores failed; confidence gate will skip")
        return text, token_logprobs

    new_token_ids = out[0][input_len:].tolist()
    answer_ids, _ = split_think_tokens(tokenizer, new_token_ids)
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def generate_reply_stream(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    *,
    do_sample: bool = True,
    repetition_penalty: float | None = None,
    min_new_tokens: int | None = None,
):
    """Yield decoded token chunks as they are produced.

    Enables SSE-style hosting and early stopping. The Azure ML managed-endpoint
    run() contract still returns a full string, so this is used only when the host
    streams (e.g. a custom container); score.run() joins the chunks otherwise.
    """
    import torch
    from threading import Thread

    from transformers import TextIteratorStreamer

    floor = int(os.environ.get("DAISY_GEN_MIN_MAX_NEW_TOKENS", "32"))
    max_new_tokens = max(floor, max_new_tokens)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = _gen_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        min_new_tokens=min_new_tokens,
    )
    gen_kwargs.update(inputs)
    gen_kwargs["streamer"] = streamer

    def _run() -> None:
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = Thread(target=_run, daemon=True)
    thread.start()
    for chunk in streamer:
        if chunk:
            yield chunk
    thread.join()


def extract_json_object(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
