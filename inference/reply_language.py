"""Reply language detection and Qwen CJK-drift guards."""

from __future__ import annotations

import re

SUPPORTED_REPLY_LANGS = frozenset({"en", "ru", "kk"})

_LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "kk": "Kazakh",
}

_KK_LETTERS = frozenset("ӘәІіҢңҒғҮүҰұҚқӨөҺһ")

# Han + kana + hangul — used to catch Qwen defaulting to Chinese/Japanese/Korean.
_CJK_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)

_CYRILLIC_WORD_RE = re.compile(r"[\u0400-\u04ff]+")

CYRILLIC_FLOOR_CHARS = 15
CYRILLIC_FLOOR_WORDS = 2


def _count_script_chars(text: str) -> tuple[int, int, bool]:
    """Return (latin_count, cyrillic_count, has_kk_specific)."""
    latin = 0
    cyr = 0
    has_kk = False
    for c in text:
        if not c.isalpha():
            continue
        if "\u0400" <= c <= "\u04ff":
            cyr += 1
            if c in _KK_LETTERS:
                has_kk = True
        elif c.isascii() and c.isalpha():
            latin += 1
    return latin, cyr, has_kk


def detect_intent_language(text: str) -> str | None:
    """
    Language of the user's intent in the current message.

    Mixed messages (Russian + English insert) resolve to the dominant intent language.
    Returns None when intent is ambiguous (empty / no letters) — caller uses UI locale.
    """
    if not text or not text.strip():
        return None

    latin, cyr, has_kk = _count_script_chars(text)
    total_alpha = latin + cyr
    if total_alpha == 0:
        return None

    cyr_words = len(_CYRILLIC_WORD_RE.findall(text))

    if has_kk and cyr >= CYRILLIC_FLOOR_WORDS:
        return "kk"

    if cyr >= CYRILLIC_FLOOR_CHARS or cyr_words >= CYRILLIC_FLOOR_WORDS:
        return "ru"

    if cyr > latin:
        return "ru"

    if latin > 0 and cyr == 0:
        return "en"

    if latin >= cyr:
        return "en"

    return "ru"


def detect_language(text: str) -> str:
    """Backward-compatible alias; ambiguous → en (legacy callers)."""
    return detect_intent_language(text) or "en"


def resolve_reply_language(intent: str | None, locale: str | None) -> str:
    """Intent wins over UI locale; UI locale is fallback only."""
    det = (intent or "").lower()[:2]
    loc = (locale or "").lower()[:2]
    if det in SUPPORTED_REPLY_LANGS:
        return det
    if loc in SUPPORTED_REPLY_LANGS:
        return loc
    return "en"


def language_lock_line(reply_lang: str, *, force_english: bool) -> str:
    """Strong, model-visible instruction — Qwen often ignores a single soft lang line."""
    if force_english:
        return (
            "LANGUAGE (mandatory): Reply in English only. "
            "Never use Chinese, Japanese, Korean, or mixed scripts."
        )
    if reply_lang == "ru":
        return (
            "LANGUAGE (mandatory): Reply in Russian only. "
            "Use informal «ты», not formal «Вы», unless the user clearly uses Вы. "
            "Never use Chinese, English, or mixed scripts unless quoting the user."
        )
    if reply_lang == "kk":
        return (
            "LANGUAGE (mandatory): Reply in Kazakh only. "
            "Never use Chinese, English, or mixed scripts unless quoting the user."
        )
    return (
        "LANGUAGE (mandatory): Reply in English only. "
        "Never use Chinese, Japanese, Korean, or mixed scripts."
    )


def language_retry_suffix(reply_lang: str) -> str:
    name = _LANG_NAMES.get(reply_lang, "English")
    return (
        f"CRITICAL — your previous draft used the wrong script. "
        f"Reply again in {name} only. Use zero Chinese characters."
    )


def generation_used_wrong_script(text: str, reply_lang: str) -> bool:
    """True when the model emitted CJK-heavy text but Daisy targets en/ru/kk."""
    if reply_lang not in SUPPORTED_REPLY_LANGS:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    cjk_chars = _CJK_RE.findall(stripped)
    if not cjk_chars:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    if letters == 0:
        return len(cjk_chars) >= 2
    if len(cjk_chars) >= 6:
        return True
    return len(cjk_chars) / letters >= 0.06


def strip_cjk_from_response(text: str) -> str:
    """Last-resort cleanup when retries still leak Han/kana/hangul."""
    cleaned = _CJK_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
