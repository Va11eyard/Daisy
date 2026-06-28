/**
 * ALLOWED_TRANSITIONS — spec §2.7.
 * Mirrors the const block in docs/Anti-Hallucination Spec.md verbatim.
 */

import type { EmotionalStateLabel } from "./types.js";

export interface TransitionEdge {
  from: EmotionalStateLabel;
  to: EmotionalStateLabel;
  minHours: number;
  maxHours: number;
  requiresCheck: "SSI" | "riskLevel" | "BSI" | null;
}

export const ALLOWED_TRANSITIONS: readonly TransitionEdge[] = [
  // Plutchik primary wheel adjacency (bidirectional)
  { from: "joy", to: "trust", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "trust", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "trust", to: "fear", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "fear", to: "trust", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "fear", to: "surprise", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "surprise", to: "fear", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "surprise", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "sadness", to: "surprise", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "sadness", to: "disgust", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "disgust", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "disgust", to: "anger", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "disgust", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "anticipation", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anticipation", to: "anger", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anticipation", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "joy", to: "anticipation", minHours: 0, maxHours: 168, requiresCheck: null },

  // Primary → clinical escalations
  { from: "fear", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "fear", to: "panic", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "fear", to: "hypervigilance", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anticipation", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anticipation", to: "fear", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "sadness", to: "grief", minHours: 0, maxHours: Infinity, requiresCheck: null },
  { from: "sadness", to: "depression", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "sadness", to: "hopelessness", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "sadness", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "guilt", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "shame", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "restlessness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "disgust", to: "shame", minHours: 0, maxHours: 168, requiresCheck: null },

  // Clinical ↔ clinical
  { from: "anxiety", to: "panic", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "anxiety", to: "rumination", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anxiety", to: "hypervigilance", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anxiety", to: "flooded", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anxiety", to: "exhaustion", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "panic", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "panic", to: "fear", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "panic", to: "flooded", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "panic", to: "exhaustion", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "rumination", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "rumination", to: "depression", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "rumination", to: "exhaustion", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "rumination", to: "shame", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "rumination", to: "guilt", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "depression", to: "sadness", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "depression", to: "hopelessness", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "depression", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "depression", to: "exhaustion", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "depression", to: "rumination", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "depression", to: "grief", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "suicidal_ideation", minHours: 0, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "hopelessness", to: "depression", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "exhaustion", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "grief", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "shame", to: "guilt", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "shame", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "shame", to: "depression", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "shame", to: "anger", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "shame", to: "rumination", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "guilt", to: "shame", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "guilt", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "guilt", to: "rumination", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "grief", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "grief", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "grief", to: "depression", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "grief", to: "anger", minHours: 0, maxHours: 168, requiresCheck: null },

  // Suicidal_ideation outgoing (always SSI; flourishing gated by rule A3)
  { from: "suicidal_ideation", to: "hopelessness", minHours: 0, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "suicidal_ideation", to: "depression", minHours: 0, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "suicidal_ideation", to: "relief", minHours: 0, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "suicidal_ideation", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "suicidal_ideation", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: "SSI" },
  { from: "suicidal_ideation", to: "flourishing", minHours: 336, maxHours: Infinity, requiresCheck: "SSI" },

  // Somatic ↔ emotional
  { from: "exhaustion", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "exhaustion", to: "depression", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "exhaustion", to: "restlessness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "numbness", to: "dissociation", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "numbness", to: "sadness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "numbness", to: "grief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "numbness", to: "flooded", minHours: 0, maxHours: 168, requiresCheck: "BSI" },
  { from: "dissociation", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "dissociation", to: "derealization", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "dissociation", to: "flooded", minHours: 0, maxHours: 168, requiresCheck: "BSI" },
  { from: "derealization", to: "dissociation", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "derealization", to: "numbness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "hypervigilance", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "hypervigilance", to: "fear", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "hypervigilance", to: "exhaustion", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "hypervigilance", to: "flooded", minHours: 0, maxHours: 168, requiresCheck: "BSI" },
  { from: "flooded", to: "exhaustion", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "flooded", to: "numbness", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "flooded", to: "dissociation", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "flooded", to: "panic", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "restlessness", to: "anxiety", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "restlessness", to: "anger", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "restlessness", to: "exhaustion", minHours: 0, maxHours: 168, requiresCheck: null },

  // Distress → positive trajectory re-entries
  { from: "fear", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anxiety", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anxiety", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "panic", to: "relief", minHours: 0, maxHours: 24, requiresCheck: null },
  { from: "panic", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "sadness", to: "acceptance", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "sadness", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "grief", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "grief", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "depression", to: "relief", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "depression", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "depression", to: "groundedness", minHours: 72, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "relief", minHours: 24, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "hopelessness", to: "groundedness", minHours: 72, maxHours: Infinity, requiresCheck: "riskLevel" },
  { from: "shame", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "shame", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "guilt", to: "acceptance", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "guilt", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "anger", to: "acceptance", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "anticipation", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "rumination", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "rumination", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "exhaustion", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "exhaustion", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "numbness", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "numbness", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "dissociation", to: "relief", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "dissociation", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "derealization", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "derealization", to: "relief", minHours: 0, maxHours: 168, requiresCheck: "riskLevel" },
  { from: "hypervigilance", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "hypervigilance", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "flooded", to: "relief", minHours: 0, maxHours: 24, requiresCheck: "BSI" },
  { from: "flooded", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: "BSI" },
  { from: "restlessness", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "restlessness", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "trust", to: "groundedness", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "trust", to: "acceptance", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "joy", to: "flourishing", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "joy", to: "acceptance", minHours: 0, maxHours: 168, requiresCheck: null },

  // Positive ↔ positive (intra-trajectory)
  { from: "relief", to: "groundedness", minHours: 0, maxHours: Infinity, requiresCheck: null },
  { from: "relief", to: "acceptance", minHours: 24, maxHours: Infinity, requiresCheck: null },
  { from: "relief", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "relief", to: "flourishing", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "groundedness", to: "acceptance", minHours: 0, maxHours: Infinity, requiresCheck: null },
  { from: "groundedness", to: "flourishing", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "groundedness", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "groundedness", to: "trust", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "groundedness", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "acceptance", to: "flourishing", minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "acceptance", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "acceptance", to: "relief", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "acceptance", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "acceptance", to: "trust", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "flourishing", to: "joy", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "flourishing", to: "groundedness", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "flourishing", to: "acceptance", minHours: 0, maxHours: 168, requiresCheck: null },
  { from: "flourishing", to: "trust", minHours: 0, maxHours: 168, requiresCheck: null },
];

/**
 * O(1) lookup helper. Returns the matching edge if (from, to) is allowed,
 * else null. Implementation is O(N) but the table is small (~150 edges).
 */
export function findTransition(
  from: EmotionalStateLabel,
  to: EmotionalStateLabel,
): TransitionEdge | null {
  for (const edge of ALLOWED_TRANSITIONS) {
    if (edge.from === from && edge.to === to) return edge;
  }
  return null;
}
