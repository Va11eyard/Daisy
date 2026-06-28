"""
System strings for book-derived SFT rows: aligned with DAISY_VOICE=therapist at inference.

Use explicit `system` on each JSONL row so prepare_dataset does not fall back to companion-only defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_inference = str(_ROOT / "inference")
if _inference not in sys.path:
    sys.path.insert(0, _inference)

from personas import resolve_persona  # noqa: E402
from therapy_identity import get_persona_obedience_line, get_therapy_scope_guardrail, get_voice_lines  # noqa: E402


def _lang_line(locale: str) -> str:
    loc = (locale or "en").lower()[:2]
    if loc == "ru":
        return "Отвечай на русском языке."
    if loc == "kk":
        return "Қазақ тілінде жауап бер."
    return "Respond in the same language the user writes in."


def build_book_training_system(
    *,
    locale: str,
    persona: str,
    modality: str | None = None,
    source_book: str = "",
) -> str:
    """
    Full system prompt for book-grounded examples: therapist voice + persona + optional modality.
    """
    core1, core2 = get_voice_lines("therapist")
    lines = [core1, core2, _lang_line(locale), get_therapy_scope_guardrail(), get_persona_obedience_line()]

    lines.append(
        "Ground answers in established clinical and psychoeducational approaches (e.g. CBT, ACT, DBT, mindfulness, "
        "psychodynamic) when relevant. Do not invent citations or study results; stay within the material implied by the conversation."
    )
    if modality:
        lines.append(f"Primary orientation for this example: {modality}.")

    if source_book:
        lines.append(f"Training excerpt source (internal, do not quote as a fact about the user): {source_book}")

    pi, pe = resolve_persona(persona)
    lines.append("\n" + pi)
    if pe:
        lines.append("Example tone: " + pe)

    return "\n".join(lines)
