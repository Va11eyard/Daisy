"""Parse onboarding_summary into model-friendly text (simplified)."""

from __future__ import annotations

import json
from typing import Any

_SCALE_FIELDS_EN = {
    "work_study": "Work/study",
    "romantic_relationships": "Romantic relationships",
    "family": "Family",
    "friends": "Friends",
    "autonomy": "Autonomy",
    "physical_health": "Physical health",
    "emotional_state": "Emotional state",
    "hobbies": "Hobbies",
    "housing": "Housing",
    "finances": "Finances",
}


def parse_onboarding(raw: Any) -> tuple[str, str | None]:
    """Returns (formatted_text, gender or None)."""
    if not raw:
        return "", None
    gender: str | None = None
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw, _extract_gender_from_text(raw)
    elif isinstance(raw, dict):
        data = raw
    else:
        return str(raw), None

    lines: list[str] = []
    g = data.get("gender")
    if isinstance(g, str):
        gl = g.lower()
        if gl in ("female", "f", "женский", "ж"):
            gender = "female"
        elif gl in ("male", "m", "мужской", "м"):
            gender = "male"

    for key, label in _SCALE_FIELDS_EN.items():
        val = data.get(key)
        if isinstance(val, dict):
            if val.get("has") is False:
                continue
            sc = val.get("score")
            if sc is not None:
                lines.append(f"{label}: {sc}/5")

    ap = data.get("ai_profile")
    if isinstance(ap, dict):
        if ap.get("summary"):
            lines.append(f"Summary: {str(ap['summary'])[:500]}")
        if isinstance(ap.get("goals"), list):
            lines.append("Goals: " + ", ".join(str(x) for x in ap["goals"][:5]))
        if isinstance(ap.get("concerns"), list):
            lines.append("Concerns: " + ", ".join(str(x) for x in ap["concerns"][:5]))

    return "\n".join(lines), gender


def _extract_gender_from_text(t: str) -> str | None:
    low = t.lower()
    if "gender: female" in low or "пол: женский" in low:
        return "female"
    if "gender: male" in low or "пол: мужской" in low:
        return "male"
    return None
