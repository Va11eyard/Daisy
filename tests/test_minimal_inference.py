"""Tests for inference/minimal_inference.py bare-minimum helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))

from minimal_inference import (
    MINIMAL_SYSTEM_PROMPTS,
    bare_minimum_enabled,
    build_minimal_system_prompt,
    cap_history,
    minimal_clean,
)


def test_bare_minimum_enabled_default(monkeypatch):
    monkeypatch.delenv("DAISY_BARE_MINIMUM", raising=False)
    assert bare_minimum_enabled() is True


def test_bare_minimum_disabled(monkeypatch):
    monkeypatch.setenv("DAISY_BARE_MINIMUM", "false")
    assert bare_minimum_enabled() is False


def test_minimal_clean_strips_assistant_leak():
    assert minimal_clean("Assistant: Hello") == "Hello"
    assert "icaponecessario" not in minimal_clean(
        "anxious icaponecessario Valentino Respini Ricciotti"
    )


def test_minimal_clean_collapses_word_repetition():
    out = minimal_clean("stress stress stress stress today")
    assert out.lower().count("stress") == 1


def test_minimal_clean_punctuation_debris():
    out = minimal_clean("Я понимаю, это.,?")
    assert ",.," not in out
    assert len(out) > 5


def test_cap_history():
    hist = [{"role": "user", "content": str(i)} for i in range(10)]
    capped = cap_history(hist, max_turns=6)
    assert len(capped) == 6
    assert capped[0]["content"] == "4"


def test_build_minimal_system_prompt_includes_context():
    prompt = build_minimal_system_prompt(
        "ru",
        history_summary="- Человек сказал: 'тревожно'",
        user_context="works with a difficult boss",
    )
    assert "Дэйзи" in prompt
    assert "тревожно" in prompt
    assert "boss" in prompt


def test_minimal_system_prompts_locales():
    for loc in ("en", "ru", "kk"):
        assert loc in MINIMAL_SYSTEM_PROMPTS
        assert len(MINIMAL_SYSTEM_PROMPTS[loc]) > 20
