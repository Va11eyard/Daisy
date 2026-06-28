"""
Training-time system prompts aligned with inference `build_system_prompt` (system_prompt.py).

Use `meta` on each dataset row so LoRA sees the same structure as production (persona, locale,
onboarding, memory, psych_profile, user_image).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_inference = str(_ROOT / "inference")
if _inference not in sys.path:
    sys.path.insert(0, _inference)

from system_prompt import build_system_prompt  # noqa: E402
from user_image import format_user_image_for_prompt, normalize_user_image  # noqa: E402


def _user_image_block(meta: dict[str, Any]) -> str | None:
    raw = meta.get("user_image")
    if not raw:
        return None
    if isinstance(raw, dict):
        norm = normalize_user_image(raw)
    else:
        return None
    if not norm:
        return None
    loc = meta.get("locale") or "en"
    fe = bool(meta.get("force_english", False))
    return format_user_image_for_prompt(norm, force_english=fe, locale=str(loc)[:8])


def build_system_from_meta(meta: dict[str, Any]) -> str:
    """Build a single system string matching inference layout."""
    loc = meta.get("locale") or meta.get("detected_lang") or "en"
    if isinstance(loc, str):
        loc_s = loc.lower()[:8]
    else:
        loc_s = "en"
    detected = loc_s[:2] if len(loc_s) >= 2 else "en"

    return build_system_prompt(
        locale=loc_s,
        detected_lang=detected,
        onboarding_summary=str(meta.get("onboarding_summary") or ""),
        user_context=str(meta.get("user_context") or ""),
        persona=str(meta.get("persona") or "flexible"),
        force_english=bool(meta.get("force_english", False)),
        user_gender=meta.get("user_gender"),
        psych_profile=meta.get("psych_profile") if isinstance(meta.get("psych_profile"), dict) else None,
        is_onboarding=bool(meta.get("is_onboarding", False)),
        onboarding_step=int(meta.get("onboarding_step", 0)),
        user_image_block=_user_image_block(meta),
    )


def default_training_system() -> str:
    """Fallback when a row has no `meta` and no explicit `system` string."""
    return build_system_from_meta(
        {
            "locale": "en",
            "persona": "flexible",
        }
    )
