"""Azure ML entrypoint wrapping score_qwen3 async pipeline for daisy-therapy."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import score_qwen3 as sq3
from model_loader import adapter_loaded, load_model_and_tokenizer

logger = logging.getLogger(__name__)

_model_ready = False


def init() -> None:
    global _model_ready
    hf_token = os.environ.get("HF_TOKEN")
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
    logger.info("score_qwen3_aml init base_model=%s", base_model)
    model, tokenizer = load_model_and_tokenizer(base_model, hf_token)
    sq3._model = model
    sq3._tokenizer = tokenizer
    sq3._device = next(model.parameters()).device
    _model_ready = True
    logger.info("score_qwen3_aml ready adapter_loaded=%s", adapter_loaded())


def _parse_request(raw_data: Any) -> dict[str, Any]:
    if isinstance(raw_data, (bytes, bytearray)):
        raw_data = raw_data.decode("utf-8")
    if isinstance(raw_data, str):
        data = json.loads(raw_data)
    elif isinstance(raw_data, dict):
        data = raw_data
    else:
        raise ValueError(f"Unsupported payload type: {type(raw_data)}")
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _to_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    inline = data.get("messages")
    if isinstance(inline, list) and inline:
        messages: list[dict[str, str]] = []
        for turn in inline:
            role = (turn.get("role") or "user").strip()
            content = (turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    history = data.get("history") or []
    message = (data.get("message") or "").strip()
    messages = []
    for turn in history:
        role = turn.get("role") or "user"
        content = (turn.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    if message:
        messages.append({"role": "user", "content": message})
    return messages


def run(raw_data: Any, request_headers: dict | None = None) -> dict[str, Any]:
    del request_headers  # unused; AML passes separately
    if not _model_ready:
        init()
    data = _parse_request(raw_data)
    locale = (data.get("locale") or "en").lower().strip()
    if locale not in ("en", "ru", "kk"):
        locale = "en"
    messages = _to_messages(data)
    if not messages:
        fallback = "I'm here to listen. What would you like to talk about?"
        return {
            "response": fallback,
            "inference_build": sq3.INFERENCE_BUILD,
            "adapter_loaded": adapter_loaded(),
            "inference_mode": "qwen3",
            "language": locale,
        }

    result = asyncio.run(
        sq3.run(
            {
                "messages": messages,
                "locale": locale,
                "history": [],
            }
        )
    )
    reply = (result.get("reply") or "").strip()
    meta = result.get("metadata") or {}
    return {
        "response": reply,
        "persona_used": data.get("persona") or "active_listener",
        "protocol_used": "cbt",
        "language": locale,
        "model": os.environ.get("MODEL_DISPLAY_NAME", "daisy-model"),
        "inference_build": sq3.INFERENCE_BUILD,
        "inference_mode": "qwen3",
        "adapter_loaded": adapter_loaded(),
        "debug_context": {
            "inference_build": sq3.INFERENCE_BUILD,
            "qwen3_metadata": meta,
        },
    }
