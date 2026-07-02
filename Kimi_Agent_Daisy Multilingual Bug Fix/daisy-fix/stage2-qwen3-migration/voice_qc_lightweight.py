"""
voice_qc_lightweight.py — Re-enabled voice quality control for Daisy Qwen3.

Simplified from the old multi-module QC pipeline.  Checks for banned patterns,
script leaks, structural leaks, minimum length, and canned responses.
One regen attempt on failure, then ships with warning flag.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("daisy.voice_qc")

# ---------------------------------------------------------------------------
# QCResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """Result of a VoiceQC check."""
    passed: bool
    failures: List[str] = field(default_factory=list)
    can_regen: bool = True          # Can we try regenerating?
    qc_ran: bool = True             # Did QC actually run?

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "can_regen": self.can_regen,
            "qc_ran": self.qc_ran,
        }


# ---------------------------------------------------------------------------
# VoiceQC class
# ---------------------------------------------------------------------------

class VoiceQC:
    """Lightweight voice quality control for Daisy therapy responses.

    Checks every generated response for:
      1. Banned patterns (canned greetings, role headers, emoji spam, etc.)
      2. Minimum length
      3. Script leaks in Cyrillic locales (Latin words, Polish diacritics)
      4. Structural leaks (punctuation loops, role headers)
      5. Canned response detection
    """

    # --- Class-level configuration ---

    BANNED_PATTERNS: List[str] = [
        r"^Hey\s*[-—]\s*I'm glad you're here",
        r"Assistant:\s*",
        r"Question:\s*",
        r"Human:\s*",
        r"User:\s*",
        r"\n\nUser",
        r"\n\nAssistant",
        r"\n\nQuestion",
        r"\.\s*,\s*\.\s*,",           # punctuation loops: . , . ,
        r"\.\,\.\,",                     # .,.,
        r"[🌼🌱💙🌸🌿✨🍃🌺🔆💚🌻🌷]{2,}",  # emoji spam (2+ consecutive)
        r"[🌼🌱💙🌸🌿✨]",                 # any single emoji from known leak set
        r"Score:\s*\d+",                 # rubric scoring leakage
        r"\*\*Rubric\*\*",
        r"# (Excellent|Good|Fair|Poor)",  # markdown rubric headers
    ]

    MIN_LENGTH: int = 25
    MAX_LATIN_RATIO_IN_CYRILLIC: float = 0.10  # down from 0.12

    # Known canned responses — exact or near-exact matches are rejected
    CANNED_RESPONSES: List[str] = [
        "hey -- i'm glad you're here. what's on your mind today?",
        "hey — i'm glad you're here. what's on your mind today?",
        "hey - i'm glad you're here. what's on your mind?",
        "hey there! i'm glad you're here. what brings you in today?",
        "i'm glad you're here. what's on your mind?",
        "i'm glad you're here. how can i help you today?",
        "hello! i'm glad you're here. what would you like to talk about?",
        "hey, i'm glad you're here. what's going on?",
    ]

    # Polish diacritics that leak into Cyrillic generation
    POLISH_DIACRITICS: str = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

    # Acronyms that are acceptable even in Cyrillic text
    ALLOWED_LATIN_ACRONYMS: List[str] = [
        "DBT", "CBT", "ACT", "EMDR", "PTSD", "OCD", "ADHD",
        "SSRIs", "SNRIs", "MRI", "CNS",
    ]

    # --- Compiled regexes (initialised lazily) ---
    _compiled_banned: List[re.Pattern] = []
    _compiled_canned: List[re.Pattern] = []

    def __init__(self) -> None:
        if not self._compiled_banned:
            self._compiled_banned = [
                re.compile(p, re.IGNORECASE) for p in self.BANNED_PATTERNS
            ]
        if not self._compiled_canned:
            # Compile canned patterns with word-boundary flexibility
            self._compiled_canned = [
                re.compile(re.escape(c), re.IGNORECASE)
                for c in self.CANNED_RESPONSES
            ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str, locale: str) -> QCResult:
        """Check *text* against all QC rules for the given locale.

        Args:
            text:   The model-generated assistant response.
            locale: One of "en", "ru", "kk".

        Returns:
            QCResult with passed/failed status and list of failure reasons.
        """
        failures: List[str] = []

        # 1. Empty / None check
        if not text or not text.strip():
            return QCResult(
                passed=False,
                failures=["empty_response"],
                can_regen=True,
            )

        cleaned = text.strip()

        # 2. Minimum length
        if len(cleaned) < self.MIN_LENGTH:
            failures.append(f"too_short:{len(cleaned)}")

        # 3. Banned patterns
        for pat in self._compiled_banned:
            if pat.search(cleaned):
                failures.append(f"banned_pattern:{pat.pattern[:40]}")

        # 4. Structural leaks
        if self._has_structural_leak(cleaned):
            failures.append("structural_leak")

        # 5. Script leaks (locale-specific)
        if locale in ("ru", "kk"):
            leak_result = self._check_script_leak(cleaned)
            if leak_result:
                failures.append(leak_result)

            # Check Polish diacritics
            if any(ch in cleaned for ch in self.POLISH_DIACRITICS):
                failures.append("polish_diacritics")

        # 6. Canned response check
        if self.is_canned(cleaned, locale):
            failures.append("canned_response")

        # 7. Locale correctness
        locale_ok, locale_reason = self._check_locale_correct(cleaned, locale)
        if not locale_ok:
            failures.append(locale_reason)

        passed = len(failures) == 0
        # If we have only length failure, we can regen
        can_regen = not any(
            f.startswith("canned") or f == "structural_leak"
            for f in failures
        )

        return QCResult(
            passed=passed,
            failures=failures,
            can_regen=can_regen,
        )

    def regenerate_prompt(self, original_prompt: str, failures: list) -> str:
        """Augment the system prompt with guardrails targeting the failures.

        Args:
            original_prompt: The current system prompt.
            failures:        List of failure strings from check().

        Returns:
            A stricter system prompt for the regen attempt.
        """
        guards: List[str] = []

        if any("banned_pattern" in f for f in failures):
            guards.append(
                "IMPORTANT: Never start with 'Hey' or 'I'm glad you're here'. "
                "Never output 'Assistant:', 'Question:', or any role label."
            )

        if "structural_leak" in failures:
            guards.append(
                "IMPORTANT: Do not output punctuation loops (.,.,.). "
                "Do not repeat words. Do not output role headers."
            )

        if any("script_leak" in f or "polish_diacritics" in f for f in failures):
            guards.append(
                "IMPORTANT: Write ONLY in the target language. "
                "No Latin words. No English parentheticals. "
                "No Polish or other diacritics."
            )

        if any("too_short" in f for f in failures):
            guards.append(
                "IMPORTANT: Write at least 2 full sentences. "
                "Be specific and reference what the user said."
            )

        if any("canned" in f for f in failures):
            guards.append(
                "IMPORTANT: Do NOT use generic greetings. "
                "Start by acknowledging something specific from the user's message."
            )

        if any("locale" in f for f in failures):
            guards.append(
                "IMPORTANT: Stay in the target language for the ENTIRE response. "
                "No mid-sentence language switches."
            )

        guard_block = "\n".join(f"- {g}" for g in guards)
        return f"{original_prompt}\n\n--- REGEN GUARDRAILS ---\n{guard_block}"

    def is_canned(self, text: str, locale: str) -> bool:
        """Check whether *text* matches a known canned response.

        Uses exact match for short canned greetings and Jaccard similarity
        for longer phrases.
        """
        cleaned = text.strip().lower()

        for canned_pat in self._compiled_canned:
            if canned_pat.search(cleaned):
                return True

        # Jaccard check for partial matches (>80% similarity)
        for canned in self.CANNED_RESPONSES:
            if self._jaccard(cleaned, canned.lower()) > 0.80:
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_structural_leak(self, text: str) -> bool:
        """Detect structural problems: duplicate words, punctuation loops."""
        # Duplicate word (e.g., "understand nderstand")
        if re.search(r"(\b\w+\b)\s+\1", text, re.IGNORECASE):
            return True
        # Punctuation loop: 3+ punctuation chars in a row
        if re.search(r"[.,;:!?]{3,}", text):
            return True
        # Rubric token leakage
        if re.search(r"\*\*Rubric\*\*|Score:\s*\d+|# (Excellent|Good|Fair|Poor)", text, re.IGNORECASE):
            return True
        return False

    def _check_script_leak(self, text: str) -> str:
        """Check for Latin word leaks in Cyrillic text.

        Returns:
            Empty string if clean, otherwise a failure reason string.
        """
        # Split into sentences / word groups
        sentences = re.split(r"[.!?\n]+", text)
        total_latin_words = 0
        total_words = 0

        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                # Strip punctuation
                clean_word = re.sub(r"[^\w\s]", "", word)
                if not clean_word:
                    continue
                total_words += 1
                # Check if word is Latin script (but not an allowed acronym)
                if re.match(r"^[a-zA-Z]+$", clean_word):
                    if clean_word.upper() not in self.ALLOWED_LATIN_ACRONYMS:
                        total_latin_words += 1

        if total_words == 0:
            return ""

        ratio = total_latin_words / total_words
        if ratio > self.MAX_LATIN_RATIO_IN_CYRILLIC:
            return f"script_leak:latin_ratio_{ratio:.2f}"

        # Also check for standalone English sentences (>5 Latin words in a row)
        latin_streak_pattern = re.compile(r"(?:[a-zA-Z]+\s+){5,}[a-zA-Z]+[^а-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]" if "ә" in text else r"(?:[a-zA-Z]+\s+){5,}[a-zA-Z]+")
        if latin_streak_pattern.search(text):
            return "script_leak:standalone_english_sentence"

        return ""

    def _check_locale_correct(self, text: str, locale: str) -> tuple:
        """Check that the response stays in the target language.

        Returns:
            (True, "") if OK, (False, reason) if not.
        """
        if locale == "en":
            return True, ""

        # For RU/KK: no sentence with >50% Latin words
        sentences = re.split(r"[.!?\n]+", text)
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) < 2:
                continue
            latin_count = sum(
                1 for w in words
                if re.match(r"^[a-zA-Z]+$", re.sub(r"[^\w\s]", "", w))
                and re.sub(r"[^\w\s]", "", w).upper() not in self.ALLOWED_LATIN_ACRONYMS
            )
            if latin_count / len(words) > 0.50:
                return False, f"locale_mismatch:>50%_latin_in_sentence"

        # Informal check for RU: should use "ты" not "вы"
        if locale == "ru":
            lower = text.lower()
            # If we see "вы " (formal you) without enough "ты ", flag it
            vy_count = lower.count(" вы ") + lower.count("вас ") + lower.count("вам ")
            ty_count = lower.count(" ты ") + lower.count("тебя ") + lower.count("тебе ")
            if vy_count > ty_count and ty_count == 0:
                return False, "locale_mismatch:formal_vy_instead_of_ty"

        return True, ""

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings (character bigrams)."""
        def _bigrams(s: str):
            return {s[i:i + 2] for i in range(len(s) - 1)}
        bg_a = _bigrams(a)
        bg_b = _bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        return len(bg_a & bg_b) / len(bg_a | bg_b)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_voice_qc() -> VoiceQC:
    """Return a singleton VoiceQC instance."""
    return VoiceQC()


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Run VoiceQC self-tests."""
    print("=" * 60)
    print("voice_qc_lightweight.py self-test")
    print("=" * 60)

    qc = VoiceQC()

    # --- Test 1: clean EN response ---
    en_clean = (
        "It sounds like you've been carrying a lot lately. "
        "Can you tell me more about what's been weighing on you?"
    )
    r = qc.check(en_clean, "en")
    assert r.passed, f"EN clean should pass: {r.failures}"
    print("  [PASS] EN clean response")

    # --- Test 2: canned greeting ---
    canned = "Hey -- I'm glad you're here. What's on your mind today?"
    r = qc.check(canned, "en")
    assert not r.passed, "Canned greeting should fail"
    assert any("canned" in f for f in r.failures)
    print("  [PASS] Canned greeting detected")

    # --- Test 3: role header leak ---
    role_leak = "Assistant: It sounds like you've been carrying a lot."
    r = qc.check(role_leak, "en")
    assert not r.passed, "Role header should fail"
    assert any("banned" in f for f in r.failures)
    print("  [PASS] Role header leak detected")

    # --- Test 4: too short ---
    short = "I see."
    r = qc.check(short, "en")
    assert not r.passed, "Short response should fail"
    assert any("too_short" in f for f in r.failures)
    print("  [PASS] Too short detected")

    # --- Test 5: punctuation loop ---
    loop = "I understand.,.,., that must be hard"
    r = qc.check(loop, "en")
    assert not r.passed, "Punctuation loop should fail"
    print("  [PASS] Punctuation loop detected")

    # --- Test 6: RU with Latin leak ---
    ru_leak = (
        "Мне жаль, что ты так себя чувствуешь. "
        "It sounds like you've been through a lot. "
        "Расскажи подробнее?"
    )
    r = qc.check(ru_leak, "ru")
    assert not r.passed, "RU with Latin leak should fail"
    assert any("script_leak" in f or "locale" in f for f in r.failures)
    print("  [PASS] RU Latin leak detected")

    # --- Test 7: RU clean ---
    ru_clean = (
        "Мне жаль, что ты так себя чувствуешь. "
        "Похоже, тебе пришлось через многое пройти. "
        "Расскажи подробнее, что именно тебя беспокоит?"
    )
    r = qc.check(ru_clean, "ru")
    assert r.passed, f"RU clean should pass: {r.failures}"
    print("  [PASS] RU clean response")

    # --- Test 8: Polish diacritics ---
    pl = "To jest bardzo trudne dla ciebie. ąćęłńóśźż"
    r = qc.check(pl, "ru")
    assert not r.passed, "Polish diacritics should fail"
    assert any("polish" in f for f in r.failures)
    print("  [PASS] Polish diacritics detected")

    # --- Test 9: emoji spam ---
    emoji = "It sounds hard 🌼🌼🌼💙. Tell me more."
    r = qc.check(emoji, "en")
    assert not r.passed, "Emoji spam should fail"
    print("  [PASS] Emoji spam detected")

    # --- Test 10: regenerate_prompt ---
    prompt = "You are a helpful assistant."
    failures = ["banned_pattern:Hey", "script_leak:latin_ratio_0.25"]
    regen = qc.regenerate_prompt(prompt, failures)
    assert "REGEN GUARDRAILS" in regen
    assert "Hey" in regen or "target language" in regen
    print("  [PASS] Regenerate prompt augmentation")

    # --- Test 11: allowed acronyms ---
    with_acronym = (
        "CBT может помочь в этой ситуации. "
        "Ты уже пробовал какие-то техники?"
    )
    r = qc.check(with_acronym, "ru")
    # Should pass — CBT is an allowed acronym
    assert r.passed, f"Allowed acronym should not trigger leak: {r.failures}"
    print("  [PASS] Allowed acronyms (CBT, DBT) not flagged")

    # --- Test 12: KK clean ---
    kk_clean = (
        "Мен сенің жағдайыңды түсінемін. "
        "Соңғы уақытта өзіңді қалай сезініп жүрсің?"
    )
    r = qc.check(kk_clean, "kk")
    assert r.passed, f"KK clean should pass: {r.failures}"
    print("  [PASS] KK clean response")

    print("\n" + "=" * 60)
    print("All VoiceQC self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
