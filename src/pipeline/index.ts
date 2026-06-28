/**
 * Pipeline orchestrator — wires Layers 1–5 into a single gate chain.
 *
 * Runtime sequence (per design clarification):
 *   Step 0 — Group C pre-screen on raw user input (escalations only)
 *   Step 1 — L1: retrieveGroundedEvidence
 *   Step 2 — L3: runSelfConsistency (generates K candidates → consensus)
 *   Step 3 — L2: validateLayer2 on the L3 candidate as draft
 *   Step 4 — L4: runConfidenceGate
 *   Step 5 — L5: runAutoformalization (skipped when no claims extracted)
 *
 * Each step gates the next; first abstention or escalation short-circuits.
 */

import { runConfidenceGate } from "../layers/confidence/index.js";
import { runAutoformalization } from "../layers/autoformalization/index.js";
import { retrieveGroundedEvidence } from "../layers/retrieval/index.js";
import {
  __internal as rulesEngineInternal,
  validateLayer2,
} from "../layers/rules-engine/index.js";
import { runSelfConsistency } from "../layers/self-consistency/index.js";
import {
  PipelineInputSchema,
  SurfaceableAnswerSchema,
} from "../layers/shared/layerSchemas.zod.js";
import type {
  AnswerDraft,
  EmotionalClaim,
  EscalationRoute,
  Layer2Input,
  PipelineInput,
  PropositionalFormula,
  PsychProfile,
  SurfaceableAnswer,
  Timestamp,
} from "../layers/shared/types.js";

export type { PipelineInput, SurfaceableAnswer };

const C_RULE_PRESCREEN = [
  rulesEngineInternal.rules.c9,
  rulesEngineInternal.rules.c1,
  rulesEngineInternal.rules.c8,
  rulesEngineInternal.rules.c2,
  rulesEngineInternal.rules.c3,
  rulesEngineInternal.rules.c5,
  rulesEngineInternal.rules.c4,
  rulesEngineInternal.rules.c6,
  rulesEngineInternal.rules.c7,
];

function buildStubLayer2Input(
  query: string,
  psychProfile: PsychProfile,
  now: Timestamp,
  consentedRetentionWindowDays: number,
): Layer2Input {
  return {
    evidence: {
      state: "not_answerable",
      reason: "below_threshold",
      retrievedAt: now,
    },
    draft: {
      text: "",
      daisyState: "intake",
      language: "auto",
      cites: [],
      asserts: [],
      hedges: [],
    },
    rawUserInput: query,
    psychProfile,
    now,
    consentedRetentionWindowDays,
  };
}

/**
 * Step 0 — runs C1–C9 with a stub Layer2Input. Only honors results that carry
 * an `escalation` route; non-escalation rejections (e.g. C3's plain rejection
 * when SSI is high but no draft yet exists) are silently ignored, since the
 * regular L2 step will re-evaluate them with a real draft.
 */
export function runGroupCPrescreen(
  query: string,
  psychProfile: PsychProfile,
  now: Timestamp,
  consentedRetentionWindowDays: number,
): EscalationRoute | null {
  const stub = buildStubLayer2Input(
    query,
    psychProfile,
    now,
    consentedRetentionWindowDays,
  );
  for (const rule of C_RULE_PRESCREEN) {
    const result = rule(stub);
    if (result?.escalation) return result.escalation;
  }
  return null;
}

const NO_CLAIMS_FORMULA: PropositionalFormula = {
  raw: "No claims extracted; consistency check skipped",
  formal: "",
  labels: [],
  relations: [],
};

function answer(
  text: string,
  confidence: number,
  formula: PropositionalFormula,
): SurfaceableAnswer {
  return SurfaceableAnswerSchema.parse({
    type: "answer",
    text,
    confidence,
    formula,
  }) as SurfaceableAnswer;
}

function abstain(
  template: "T1" | "T2" | "T3" | "T4" | "T5",
  reason: string,
): SurfaceableAnswer {
  return SurfaceableAnswerSchema.parse({
    type: "abstention",
    template,
    reason,
  }) as SurfaceableAnswer;
}

function escalate(route: EscalationRoute): SurfaceableAnswer {
  return SurfaceableAnswerSchema.parse({
    type: "escalation",
    route,
  }) as SurfaceableAnswer;
}

export async function runPipeline(input: PipelineInput): Promise<SurfaceableAnswer> {
  // Programmer-error path: PipelineInput shape is wrong → throw, NOT abstain.
  PipelineInputSchema.parse(input);

  try {
    const now = new Date().toISOString() as Timestamp;

    const preEscalation = runGroupCPrescreen(
      input.query,
      input.psychProfile,
      now,
      input.consentedRetentionWindowDays,
    );
    if (preEscalation) return escalate(preEscalation);

    const evidenceSet = await retrieveGroundedEvidence({
      query: input.query,
      uploadedEntries: input.uploadedEntries,
      consentedRetentionWindowDays: input.consentedRetentionWindowDays,
      now,
      embeddingFn: input.embeddingFn,
    });

    if (evidenceSet.state === "not_answerable") {
      return abstain("T1", evidenceSet.reason);
    }

    const layer3 = await runSelfConsistency({
      query: input.query,
      evidenceSet,
      generateFn: input.generateFn,
      embedFn: input.embeddingFn,
    });

    if (layer3.verdict === "divergent") {
      return abstain("T3", "sample_divergence");
    }

    // Claims extracted between L3 and L2 so L2 transition rules see a
    // populated `draft.asserts`. Reused at L5 — extracted exactly once.
    const claims: EmotionalClaim[] = input.extractClaimsFn(layer3.candidate);

    const draft: AnswerDraft = {
      text: layer3.candidate,
      // TODO: thread actual DaisyState, language, and hedges
      // from session context — currently static defaults disable
      // E3 phase-sensitivity and B8 hedge-awareness
      daisyState: "disclosure",
      language: "auto",
      cites: evidenceSet.evidence,
      asserts: claims,
      hedges: [],
    };

    const layer2 = validateLayer2({
      evidence: evidenceSet,
      draft,
      psychProfile: input.psychProfile,
      rawUserInput: input.query,
      now,
      consentedRetentionWindowDays: input.consentedRetentionWindowDays,
    });

    if (layer2.verdict === "rejected") {
      if (layer2.escalation) return escalate(layer2.escalation);
      return abstain(
        "T2",
        `${layer2.violation.ruleId}: ${layer2.violation.message}`,
      );
    }

    const layer4 = await runConfidenceGate({
      layer3Output: layer3,
      getTokenLogProbs: input.getTokenLogProbs,
    });

    if (layer4.verdict === "abstain_low_confidence") {
      return abstain("T4", "low_confidence");
    }

    if (claims.length === 0) {
      return answer(layer4.answer, layer4.confidence, NO_CLAIMS_FORMULA);
    }

    const layer5 = await runAutoformalization({
      layer4Output: layer4,
      claims,
      knownFacts: input.knownFacts,
      formalizeFn: input.formalizeFn,
    });

    if (layer5.verdict === "inconsistent") {
      const reason =
        layer5.conflictingFacts.length > 0
          ? layer5.conflictingFacts.map((f) => f.id).join(", ")
          : layer5.formula.raw;
      return abstain("T5", reason);
    }

    return answer(layer4.answer, layer4.confidence, layer5.formula);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[pipeline] unexpected error:", err);
    return abstain("T4", `pipeline_error: ${message}`);
  }
}
