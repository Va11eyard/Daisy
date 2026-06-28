/**
 * Cross-session tone-shift validation against ALLOWED_TRANSITIONS
 * (src/layers/shared/transitions.ts) plus the L2 A3 safety overlay.
 *
 * Used inline by generate_batch.ts (before writing) and by validate_batch.ts
 * (CHECK 7). A dialog only carries a transition when it explicitly declares a
 * multi-state emotion sequence; single-state dialogs validate trivially.
 */

import {
  ALLOWED_TRANSITIONS,
  findTransition,
} from "../../src/layers/shared/transitions.js";
import type { EmotionalStateLabel } from "../../src/layers/shared/types.js";

export interface PsychFlags {
  SSI?: boolean;
  riskLevel?: boolean;
  BSI?: boolean;
}

export interface TransitionStep {
  from: EmotionalStateLabel;
  to: EmotionalStateLabel;
  /** Hours elapsed between the two states, if known. */
  hoursGap?: number;
}

export interface TransitionVerdict {
  valid: boolean;
  reason: string;
}

const A3_MIN_HOURS = 336; // 14 days — L2 rule A3 hard floor.

/**
 * Validates a single emotional-state transition.
 *
 * Order mirrors the L2 engine: A3 (suicidal_ideation → flourishing) is checked
 * first as a hard human-review block, then ALLOWED_TRANSITIONS membership, then
 * timing (minHours/maxHours), then requiresCheck presence.
 */
export function validateTransition(step: TransitionStep, flags: PsychFlags = {}): TransitionVerdict {
  const { from, to, hoursGap } = step;

  if (from === to) return { valid: true, reason: "no-op (same state)" };

  // A3: suicidal_ideation → flourishing under 14d is forbidden → human_review.
  if (from === "suicidal_ideation" && to === "flourishing") {
    if (hoursGap === undefined || hoursGap < A3_MIN_HOURS) {
      return {
        valid: false,
        reason: "A3: suicidal_ideation → flourishing requires ≥336h and is human_review under 14d (do not generate)",
      };
    }
  }

  const edge = findTransition(from, to);
  if (!edge) {
    return { valid: false, reason: `A8: ${from} → ${to} not in ALLOWED_TRANSITIONS` };
  }

  if (hoursGap !== undefined) {
    if (hoursGap < edge.minHours) {
      return { valid: false, reason: `A8: ${from} → ${to} needs ≥${edge.minHours}h (got ${hoursGap}h)` };
    }
    if (hoursGap > edge.maxHours) {
      return { valid: false, reason: `A8: ${from} → ${to} exceeds max ${edge.maxHours}h (got ${hoursGap}h)` };
    }
  }

  if (edge.requiresCheck === "SSI" && !flags.SSI) {
    return { valid: false, reason: `A8: ${from} → ${to} requires psychProfile.SSI present` };
  }
  if (edge.requiresCheck === "riskLevel" && !flags.riskLevel) {
    return { valid: false, reason: `A8: ${from} → ${to} requires psychProfile.riskLevel present` };
  }
  if (edge.requiresCheck === "BSI" && !flags.BSI) {
    return { valid: false, reason: `A8: ${from} → ${to} requires psychProfile.BSI present` };
  }

  return { valid: true, reason: "ok" };
}

/** Validates each consecutive pair in an emotion-state sequence. */
export function validateSequence(
  labels: readonly EmotionalStateLabel[],
  hoursGaps: readonly number[] = [],
  flags: PsychFlags = {},
): TransitionVerdict {
  for (let i = 0; i < labels.length - 1; i++) {
    const step: TransitionStep = { from: labels[i]!, to: labels[i + 1]! };
    const gap = hoursGaps[i];
    if (gap !== undefined) step.hoursGap = gap;
    const verdict = validateTransition(step, flags);
    if (!verdict.valid) return verdict;
  }
  return { valid: true, reason: "ok" };
}

export { ALLOWED_TRANSITIONS };
