"""Load crisis hotlines from YAML (optional)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_cfg: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    global _cfg
    if _cfg is not None:
        return _cfg
    roots = [
        Path(__file__).resolve().parent.parent / "config" / "crisis_resources.yaml",
        Path(__file__).resolve().parent / "crisis_resources.yaml",
    ]
    for p in roots:
        if p.exists():
            with p.open(encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
                return _cfg
    _cfg = {}
    return _cfg


def get_crisis_resources(locale: str | None) -> list[dict[str, str]]:
    cfg = load_config()
    loc = (locale or "en").lower()[:2]
    yaml_key = {"kk": "kz", "kz": "kz", "ru": "ru"}.get(loc, loc)
    region = cfg.get(yaml_key) or cfg.get("en") or {}
    hotlines = region.get("hotlines", [])
    out: list[dict[str, str]] = []
    for h in hotlines:
        if loc in ("kk", "kz"):
            name = h.get("name_kk") or h.get("name_ru") or h.get("name", "")
            contact = h.get("contact", "")
        elif loc == "ru":
            name = h.get("name_ru") or h.get("name", "")
            contact = h.get("contact", "")
        else:
            name = h.get("name") or h.get("name_en", "")
            contact = h.get("contact", "")
        out.append({"name": name, "contact": contact})
    if not out and yaml_key != "en":
        return get_crisis_resources("en")
    return out
