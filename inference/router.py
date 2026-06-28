"""Layer 1 helper - phase routing (no second model).

The old coordinator/router chain (a JSON-plan forward pass on a second model) is
collapsed into a single, deterministic phase decision derived from the
conversation via state_detector.detect_state. There is no extra model call; the
phase feeds the system prompt and RAG retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from state_detector import DaisyState, detect_state

if TYPE_CHECKING:
    from state_detector import Message

_PHASE_DIRECTIVES: dict[str, str] = {
    "intake": (
        "Phase: INTAKE. Reflect what you heard in 1-2 sentences, then ask one open "
        "question. Do not label or categorize the feeling yet."
    ),
    "disclosure": (
        "Phase: DISCLOSURE. Witness and reflect precisely what was said. No advice, "
        "reframes, or silver linings yet. Close with one gentle question."
    ),
    "psychoeducation": (
        "Phase: PSYCHOEDUCATION. Explain the mechanism in plain spoken language and "
        "define any clinical term immediately. Close with one question."
    ),
    "action_planning": (
        "Phase: ACTION_PLANNING. Offer at most three small, concrete steps tied to what "
        "they shared. Close with one question about which step feels doable."
    ),
    "crisis": (
        "Phase: CRISIS. Stay present, acknowledge the pain, and prioritize safety."
    ),
}


def detect_phase(messages: "list[Message]") -> DaisyState:
    """Single routing decision before generation.

    crisis is already gated by safety.crisis_tier() upstream in score.run(), so we
    pass check_crisis=False to avoid double-firing that branch.
    """
    if not messages:
        return "intake"
    return detect_state(messages, check_crisis=False)


def phase_directive(phase: str) -> str:
    return _PHASE_DIRECTIVES.get(phase, _PHASE_DIRECTIVES["intake"])


def merge_phase_into_system(base: str, phase: str) -> str:
    directive = phase_directive(phase)
    if not directive:
        return base
    return base + "\n\n" + directive
