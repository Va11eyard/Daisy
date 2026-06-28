/**
 * Runtime validation for layer outputs:
 *   - EvidenceSet / EvidenceCitation (spec §1.1)
 *   - Layer3Output                   (spec §1.3)
 *   - Layer4Output                   (spec §1.4)
 *   - PropositionalFormula           (spec §1.5)
 *   - Layer5Output                   (spec §1.5)
 *   - PipelineInput / SurfaceableAnswer (orchestrator)
 *
 * Producing layers MUST parse through these before returning; consuming
 * layers assume validated shapes.
 */

import { z } from "zod";

const zSimilarity = z.number().min(0).max(1);
const zTimestamp = z.string().datetime({ offset: true });

export const EvidenceCitationSchema = z.object({
  entryId: z.string().min(1),
  sessionId: z.string().min(1),
  timestamp: zTimestamp,
  textExcerpt: z.string().min(1),
  similarity: zSimilarity,
  emotionLabels: z.array(z.string()).optional(),
  intensity: z.enum(["mild", "moderate", "severe"]).optional(),
});

export const EvidenceSetSchema = z.discriminatedUnion("state", [
  z.object({
    state: z.literal("fully_answerable"),
    evidence: EvidenceCitationSchema.array().min(1),
    aggregateSimilarity: zSimilarity,
    retrievedAt: zTimestamp,
  }),
  z.object({
    state: z.literal("partially_answerable"),
    evidence: EvidenceCitationSchema.array().min(1),
    gaps: z.array(z.string()),
    aggregateSimilarity: zSimilarity,
    retrievedAt: zTimestamp,
  }),
  z.object({
    state: z.literal("not_answerable"),
    reason: z.enum([
      "below_threshold",
      "no_data_in_window",
      "data_excluded_by_consent",
      "embedding_failure",
    ]),
    suggestedClarifyingQuestion: z.string().optional(),
    retrievedAt: zTimestamp,
  }),
]);

/** Validates and returns a typed EvidenceSet; throws ZodError on invalid builder output. */
export function parseEvidenceSet(data: unknown) {
  return EvidenceSetSchema.parse(data);
}

export const Layer3OutputSchema = z.discriminatedUnion("verdict", [
  z.object({
    verdict: z.literal("converged"),
    candidate: z.string().min(1),
    consensusScore: z.number().min(0).max(1),
  }),
  z.object({
    verdict: z.literal("divergent"),
    reason: z.literal("sample_divergence"),
    samples: z.array(z.string()),
  }),
]);

export const Layer4OutputSchema = z.discriminatedUnion("verdict", [
  z.object({
    verdict: z.literal("confident"),
    answer: z.string().min(1),
    confidence: z.number().min(0).max(1),
  }),
  z.object({
    verdict: z.literal("abstain_low_confidence"),
    template: z.literal("T4"),
  }),
]);

/**
 * Permissive on `formal` / `labels` / `relations`: empty values are valid for
 * the all-claims-failed-to-formalize sentinel formula returned in that branch.
 * `raw` must be present so the failure reason is always carried.
 */
export const PropositionalFormulaSchema = z.object({
  raw: z.string().min(1),
  formal: z.string(),
  labels: z.array(z.string()),
  relations: z.array(z.string()),
});

const PropositionalFactSchema = z.object({
  id: z.string().min(1),
  formula: z.string().min(1),
  sourceEntryId: z.string().min(1),
  sessionId: z.string().min(1),
  timestamp: z.string().min(1),
});

export const Layer5OutputSchema = z.discriminatedUnion("verdict", [
  z.object({
    verdict: z.literal("consistent"),
    formula: PropositionalFormulaSchema,
  }),
  z.object({
    verdict: z.literal("inconsistent"),
    conflictingFacts: z.array(PropositionalFactSchema),
    formula: PropositionalFormulaSchema,
  }),
]);

const JournalEntrySchema = z.object({
  entryId: z.string().min(1),
  sessionId: z.string().min(1),
  timestamp: z.string().min(1),
  text: z.string(),
});

const PsychProfileSchema = z.object({
  ESI: z.number().optional(),
  BSI: z.number().optional(),
  SSI: z.number().optional(),
  MRI: z.number().optional(),
  riskLevel: z.enum(["low", "moderate", "high"]).optional(),
});

const fnSchema = z.custom<(...args: never[]) => unknown>(
  (val) => typeof val === "function",
  "Expected function",
);

export const PipelineInputSchema = z.object({
  query: z.string().min(1),
  uploadedEntries: z.array(JournalEntrySchema),
  consentedRetentionWindowDays: z.number().int().positive(),
  psychProfile: PsychProfileSchema,
  knownFacts: z.array(PropositionalFactSchema),
  embeddingFn: fnSchema,
  generateFn: fnSchema,
  getTokenLogProbs: fnSchema,
  formalizeFn: fnSchema,
  extractClaimsFn: fnSchema,
});

const EscalationRouteSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("crisis_template"),
    tier: z.union([z.literal(1), z.literal(2)]),
  }),
  z.object({ kind: z.literal("identity_template") }),
  z.object({ kind: z.literal("injection_block") }),
  z.object({ kind: z.literal("human_review") }),
]);

export const SurfaceableAnswerSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("answer"),
    text: z.string().min(1),
    confidence: z.number().min(0).max(1),
    formula: PropositionalFormulaSchema,
  }),
  z.object({
    type: z.literal("abstention"),
    template: z.enum(["T1", "T2", "T3", "T4", "T5"]),
    reason: z.string(),
  }),
  z.object({
    type: z.literal("escalation"),
    route: EscalationRouteSchema,
  }),
]);
