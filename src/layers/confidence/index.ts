/**
 * Layer 4 — Confidence-aware abstention (spec §1.4).
 *
 * Combined gate: both semantic consensus (from Layer 3) AND mean per-token
 * negative log-probability ("sequence entropy") must clear their thresholds.
 * Either gate failing → abstain with template T4.
 *
 * Caller MUST pass a converged Layer3Output. A divergent input is a contract
 * violation, not an abstention condition.
 */

import { hallucinationConfig } from "../shared/config.js";
import { Layer4OutputSchema } from "../shared/layerSchemas.zod.js";
import type { Layer4Input, Layer4Output } from "../shared/types.js";

export type { Layer4Input, Layer4Output };

function abstain(): Layer4Output {
  return Layer4OutputSchema.parse({
    verdict: "abstain_low_confidence",
    template: "T4",
  }) as Layer4Output;
}

/**
 * Decides whether to surface the converged candidate to the user.
 *
 * Order of checks:
 *   1. Caller-contract guard: divergent Layer3Output → throw.
 *   2. Consensus gate: consensusScore < semanticEntropyThreshold → abstain.
 *   3. Token log-prob retrieval: throws / empty → abstain.
 *   4. Entropy gate: meanEntropy ≥ sequenceEntropyMax → abstain.
 *   5. Otherwise return confident with confidence in (0, 1].
 */
export async function runConfidenceGate(input: Layer4Input): Promise<Layer4Output> {
  const { layer3Output, getTokenLogProbs } = input;

  if (layer3Output.verdict === "divergent") {
    throw new Error("Layer4 must not receive divergent Layer3Output");
  }

  const { candidate, consensusScore } = layer3Output;

  if (consensusScore < hallucinationConfig.semanticEntropyThreshold) {
    return abstain();
  }

  let logProbs: number[];
  try {
    logProbs = await getTokenLogProbs(candidate);
  } catch {
    return abstain();
  }

  if (logProbs.length === 0) {
    return abstain();
  }

  let sumNegLogProb = 0;
  for (const lp of logProbs) sumNegLogProb += -lp;
  const meanEntropy = sumNegLogProb / logProbs.length;

  const ceiling = hallucinationConfig.sequenceEntropyMax;
  // Both branches abstain at-or-above the ceiling: rule 1 (>) plus rule 6
  // (clamped confidence ≤ 0). Collapsing to >= avoids the zero-confidence
  // confident output entirely.
  if (meanEntropy >= ceiling) {
    return abstain();
  }

  const rawConfidence = 1 - meanEntropy / ceiling;
  const confidence = Math.max(0, Math.min(1, rawConfidence));

  if (confidence <= 0) {
    return abstain();
  }

  return Layer4OutputSchema.parse({
    verdict: "confident",
    answer: candidate,
    confidence,
  }) as Layer4Output;
}
