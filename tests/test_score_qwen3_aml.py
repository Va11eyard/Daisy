"""Tests for score_qwen3_aml payload mapping and post-translate QC."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))


def test_to_messages_from_inline_messages():
    import score_qwen3_aml as aml

    data = {
        "messages": [{"role": "user", "content": "Привет, тревожно"}],
        "locale": "ru",
    }
    msgs = aml._to_messages(data)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Привет, тревожно"


def test_to_messages_from_history_and_message():
    import score_qwen3_aml as aml

    data = {
        "message": "I feel anxious",
        "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        "locale": "en",
    }
    msgs = aml._to_messages(data)
    assert len(msgs) == 3
    assert msgs[-1] == {"role": "user", "content": "I feel anxious"}


def test_run_maps_reply_to_response():
    import score_qwen3_aml as aml

    aml._model_ready = True
    mock_result = {"reply": "Take a breath.", "metadata": {"qc_passed": True}}

    with patch.object(aml.sq3, "run", new_callable=AsyncMock, return_value=mock_result):
        out = aml.run(
            json.dumps(
                {
                    "message": "stress",
                    "history": [],
                    "locale": "ru",
                }
            )
        )

    assert out["response"] == "Take a breath."
    assert out["inference_mode"] == "qwen3"
    assert out["language"] == "ru"
    assert out["inference_build"] == aml.sq3.INFERENCE_BUILD


def test_post_translate_qc_rejects_short_ru(monkeypatch):
    import score as score_mod

    monkeypatch.setenv("DAISY_POST_TRANSLATE_QC", "true")
    monkeypatch.setenv("DAISY_TRANSLATE_MIN_LENGTH", "40")
    short = "Короткий ответ."
    out = score_mod._apply_post_translate_qc(short, "ru")
    assert out != short


def test_post_translate_qc_passes_clean_ru(monkeypatch):
    import score as score_mod

    monkeypatch.setenv("DAISY_POST_TRANSLATE_QC", "true")
    monkeypatch.setenv("DAISY_TRANSLATE_MIN_LENGTH", "20")
    good = (
        "Похоже, тебе сейчас тяжело. "
        "Расскажи, что именно вызывает это чувство?"
    )
    out = score_mod._apply_post_translate_qc(good, "ru")
    assert out == good


def test_post_translate_qc_disabled_by_default(monkeypatch):
    import score as score_mod

    monkeypatch.delenv("DAISY_POST_TRANSLATE_QC", raising=False)
    leaky = "It sounds hard. Мне жаль."
    out = score_mod._apply_post_translate_qc(leaky, "ru")
    assert out == leaky
