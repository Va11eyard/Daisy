"""Tests for scripts/md_to_dialogues.py (no API calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from md_to_dialogues import (  # noqa: E402
    _infer_locale,
    _infer_persona,
    build_heuristic_record,
)
from rich_md_content import build_rich_record  # noqa: E402


def test_infer_persona_emergency():
    assert _infer_persona("Emergency cases/foo.md") == "calm_mentor"


def test_infer_persona_default():
    assert _infer_persona("misc/unknown.md") == "flexible"


def test_infer_locale_ru():
    assert _infer_locale("Рус/Эмпатия/x.md") == "ru"


def test_build_rich_record_six_turn_when_long():
    long_chunk = "\n\n".join([f"Параграф {i} с достаточным количеством текста для разбиения." for i in range(8)])
    rec = build_rich_record(
        chunk=long_chunk,
        title="Тема",
        section_title="Раздел",
        rel_path="Рус/x.md",
        persona="warm_friend",
        locale="ru",
        archetype="explain",
        max_assistant_chars=800,
        seed="s",
        emergency=False,
        long_dialog=True,
    )
    assert len(rec["messages"]) == 6


def test_build_heuristic_record_shape():
    rec = build_heuristic_record(
        chunk="Параграф один.\n\nПараграф два с достаточной длиной для смысла.",
        title="Тестовая тема",
        rel_path="Рус/Тест/x.md",
        persona="warm_friend",
        locale="ru",
        max_assistant_chars=500,
        seed="seed",
    )
    assert "messages" in rec and "meta" in rec
    assert len(rec["messages"]) == 4
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][1]["role"] == "assistant"
    assert rec["meta"]["persona"] == "warm_friend"
    assert rec["meta"]["locale"] == "ru"
