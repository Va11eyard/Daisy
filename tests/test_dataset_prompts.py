"""Tests for training system strings aligned with inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_prompts import build_system_from_meta, default_training_system  # noqa: E402


def test_default_training_system_matches_daisy_core():
    s = default_training_system()
    assert "Daisy" in s
    assert "warm and caring" in s.lower() or "companion" in s.lower()


def test_meta_ru_warm_friend():
    s = build_system_from_meta({"locale": "ru", "persona": "warm_friend"})
    assert "русском" in s or "Отвечай" in s


def test_meta_includes_psych_profile():
    s = build_system_from_meta(
        {
            "locale": "en",
            "persona": "flexible",
            "psych_profile": {"ESI": 40.0, "riskLevel": "low"},
        }
    )
    assert "ESI" in s or "40" in s
    assert "risk" in s.lower()
