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
  fallback_reply              - safe language-aware reply when a layer rejects output
  extract_json_object         - used by report_handlers.py
"""

from __future__ import annotations

import json
import logging
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
_TRAILING_EMOJI_PUNCT = re.compile(
    r"([\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\uFE0F\u2665\u2661]+)\s*[\.\?!…]+\s*$"
)

_OCR_JUNK_CHARS = "\u00b4\u02c6\u00b8\u00ba\u017d\u017a\u0142\u015b"
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
    "Hey — I'm glad you're here. What's been on your mind lately?",
    "I'm here. What feels strongest right now — anxiety, fatigue, or something else?",
    "Breakups can leave everything feeling unsteady. What part is hitting you hardest?",
)


def _normalize_for_compare(text: str) -> str:
    t = clean_model_text(text).lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def clean_model_text(text: str, *, lang: str = "ru") -> str:
    """Remove OCR junk (acute accents between Cyrillic words) and normalize whitespace."""
    if not text or not text.strip():
        return text
    t = text.strip()
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
        t = _TRAILING_EMOJI_PUNCT.sub(r"\1", t).strip()
    return t


def trim_to_complete_sentence(text: str) -> str:
    if not text or not text.strip():
        return text
    text = text.strip()
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
        if len(tail) > 40:
            return True
        return bool(re.match(r"^[\.\?\s!…]+$", tail) or _DEGEN_PUNCT_CLUSTER.search(tail))

    for pos in boundaries:
        head = text[:pos].strip()
        tail = text[pos:].strip()
        if not tail:
            return head
        if _bad_tail(tail):
            if len(head) >= 30:
                return head
            continue
    return text


def postprocess_model_response(response: str, reply_lang: str) -> tuple[str, bool]:
    """Clean, trim; return (text, sanitized_flag)."""
    raw = response
    cleaned = clean_model_text(response, lang=reply_lang)
    trimmed = trim_to_complete_sentence(cleaned)
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
    """Short safe reply used when a layer rejects the model output."""
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
