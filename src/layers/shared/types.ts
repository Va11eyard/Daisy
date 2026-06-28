/**
 * Shared types for the 5-layer anti-hallucination pipeline.
 * Source-of-truth: docs/Anti-Hallucination Spec.md (§1, §2.6).
 */

type Brand<T, K> = T & { readonly __brand: K };

export type UserId = Brand<string, "UserId">;
export type SessionId = Brand<string, "SessionId">;
export type EntryId = Brand<string, "EntryId">;
export type Timestamp = Brand<string, "ISO8601">;
export type Confidence = Brand<number, "Confidence_0_1">;
export type Similarity = Brand<number, "Cosine_0_1">;

export type Language = "en" | "ru" | "kk" | "auto";

export type DaisyState =
  | "intake"
  | "disclosure"
  | "psychoeducation"
  | "action_planning"
  | "crisis";

export type Intensity = "mild" | "moderate" | "severe";

/** Spec §2.2 alias — brief uses `IntensityLevel`, repo uses `Intensity`. */
export type IntensityLevel = Intensity;

/**
 * Closed emotional vocabulary — spec §2.1.
 * 28 labels: 8 Plutchik primaries + 10 clinical states + 6 somatic markers
 * + 4 positive trajectory states.
 */
export type EmotionLabel =
  // A. Plutchik primaries (8)
  | "joy"
  | "trust"
  | "fear"
  | "surprise"
  | "sadness"
  | "disgust"
  | "anger"
  | "anticipation"
  // B. Clinical states (10)
  | "anxiety"
  | "depression"
  | "dissociation"
  | "hopelessness"
  | "shame"
  | "guilt"
  | "grief"
  | "suicidal_ideation"
  | "panic"
  | "rumination"
  // C. Somatic / regulation markers (6)
  | "exhaustion"
  | "numbness"
  | "hypervigilance"
  | "flooded"
  | "restlessness"
  | "derealization"
  // D. Positive trajectory states (4)
  | "relief"
  | "groundedness"
  | "acceptance"
  | "flourishing";

/**
 * Alias for spec §2.7 ALLOWED_TRANSITIONS — see docs/Anti-Hallucination Spec.md.
 * The §2.7 const declaration uses `EmotionalStateLabel`; everywhere else in the
 * spec uses `EmotionLabel`. This alias reconciles the two without duplicating
 * the literal union.
 */
export type EmotionalStateLabel = EmotionLabel;

export type TemporalRelation =
  | "precedes"
  | "co_occurs"
  | "persists"
  | "recurs"
  | "escalates"
  | "de_escalates"
  | "transitions"
  | "oscillates";

export const ALL_EMOTION_LABELS: readonly EmotionLabel[] = [
  "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
  "anxiety", "depression", "dissociation", "hopelessness", "shame", "guilt", "grief",
  "suicidal_ideation", "panic", "rumination",
  "exhaustion", "numbness", "hypervigilance", "flooded", "restlessness", "derealization",
  "relief", "groundedness", "acceptance", "flourishing",
] as const;

export interface EvidenceCitation {
  entryId: EntryId;
  sessionId: SessionId;
  timestamp: Timestamp;
  textExcerpt: string;
  similarity: Similarity;
  emotionLabels?: EmotionLabel[];
  intensity?: Intensity;
}

/**
 * Layer 3 — Self-consistency input. Spec §1.3.
 * `evidenceSet` is forwarded for caller-contract enforcement (must not be
 * `not_answerable`); `generateFn` is treated as a closure over evidence-aware
 * prompting upstream of L3.
 */
export interface Layer3Input {
  query: string;
  evidenceSet: EvidenceSet;
  generateFn: (prompt: string) => Promise<string>;
  embedFn: (text: string) => Promise<number[]>;
}

export type Layer3Output =
  | { verdict: "converged"; candidate: string; consensusScore: number }
  | { verdict: "divergent"; reason: "sample_divergence"; samples: string[] };

/**
 * Layer 4 — Confidence-aware abstention input. Spec §1.4.
 *
 * Wraps the full Layer3Output (rather than carrying literal `candidate` /
 * `consensusScore` fields) so the divergent-input guard in rule 4 of the
 * implementation contract is type-safe. The fields described in §1.4 are
 * destructured from `layer3Output` inside the layer body.
 */
export interface Layer4Input {
  layer3Output: Layer3Output;
  getTokenLogProbs: (text: string) => Promise<number[]>;
}

export type Layer4Output =
  | { verdict: "confident"; answer: string; confidence: number }
  | { verdict: "abstain_low_confidence"; template: "T4" };

/** Spec §1.5 — known propositional fact in the user's persistent psych record. */
export interface PropositionalFact {
  id: string;
  formula: string;
  sourceEntryId: string;
  sessionId: string;
  timestamp: string;
}

/** Spec §1.5 — output of autoformalization (NL → propositional logic). */
export interface PropositionalFormula {
  raw: string;
  formal: string;
  labels: EmotionalStateLabel[];
  relations: TemporalRelation[];
}

/** Spec §1.5 — closed vocabulary the formalizer is bounded to. */
export interface FormalVocabulary {
  stateLabels: EmotionalStateLabel[];
  intensityLevels: IntensityLevel[];
  temporalRelations: TemporalRelation[];
  logicalOperators: string[];
}

/**
 * Layer 5 — Autoformalization input. Spec §1.5.
 *
 * Wraps `Layer4Output` so the abstain-input guard is type-safe; the brief's
 * field-level `answer: string` is destructured from `layer4Output` inside the
 * layer body. Same pattern as `Layer4Input` wrapping `Layer3Output`.
 */
export interface Layer5Input {
  layer4Output: Layer4Output;
  claims: EmotionalClaim[];
  knownFacts: PropositionalFact[];
  formalizeFn: (
    text: string,
    vocabulary: FormalVocabulary,
  ) => Promise<PropositionalFormula>;
}

export type Layer5Output =
  | { verdict: "consistent"; formula: PropositionalFormula }
  | {
      verdict: "inconsistent";
      conflictingFacts: PropositionalFact[];
      formula: PropositionalFormula;
    };

/**
 * Pipeline orchestrator input. Spec wiring §1 + spec §1.5 knownFacts.
 *
 * `psychProfile` is REQUIRED (not optional) so C3/C4 safety gates cannot be
 * silently disabled. `now` is intentionally NOT carried here — the
 * orchestrator stamps it per-invocation.
 */
export interface PipelineInput {
  query: string;
  uploadedEntries: JournalEntry[];
  consentedRetentionWindowDays: number;
  psychProfile: PsychProfile;
  knownFacts: PropositionalFact[];
  embeddingFn: (text: string) => Promise<number[]>;
  generateFn: (prompt: string) => Promise<string>;
  getTokenLogProbs: (text: string) => Promise<number[]>;
  formalizeFn: (
    text: string,
    vocabulary: FormalVocabulary,
  ) => Promise<PropositionalFormula>;
  extractClaimsFn: (answer: string) => EmotionalClaim[];
}

export type AbstentionTemplate = "T1" | "T2" | "T3" | "T4" | "T5";

export type SurfaceableAnswer =
  | {
      type: "answer";
      text: string;
      confidence: number;
      formula: PropositionalFormula;
    }
  | { type: "abstention"; template: AbstentionTemplate; reason: string }
  | { type: "escalation"; route: EscalationRoute };

/** Full journal row supplied to Layer 1 in-memory retrieval (upload bundle). */
export interface JournalEntry {
  entryId: EntryId;
  sessionId: SessionId;
  timestamp: Timestamp;
  text: string;
}

/**
 * Layer 1 retrieval input — upload-time RAG over `uploadedEntries`.
 * `now` anchors recency/consent filtering and `EvidenceSet.retrievedAt` (spec §1.1).
 */
export interface Layer1Input {
  query: string;
  uploadedEntries: JournalEntry[];
  consentedRetentionWindowDays: number;
  now: Timestamp;
  embeddingFn: (text: string) => Promise<number[]>;
}

export interface PsychProfile {
  ESI?: number;
  BSI?: number;
  SSI?: number;
  MRI?: number;
  riskLevel?: "low" | "moderate" | "high";
}

export interface HedgeMarker {
  span: { start: number; end: number };
  kind: "epistemic" | "tentative" | "negation" | "frequency";
}

export interface EmotionalClaim {
  subject: "user";
  label: EmotionLabel;
  intensity?: Intensity;
  timestamp?: Timestamp | "now" | "trend";
  trend?: { window: string; direction: "increasing" | "decreasing" | "stable" };
  citations: EvidenceCitation[];
}

export interface AnswerDraft {
  text: string;
  daisyState: DaisyState;
  language: Language;
  cites: EvidenceCitation[];
  asserts: EmotionalClaim[];
  hedges: HedgeMarker[];
}

export type EvidenceSet =
  | EvidenceSetFull
  | EvidenceSetPartial
  | EvidenceSetAbstain;

export interface EvidenceSetFull {
  state: "fully_answerable";
  evidence: EvidenceCitation[];
  aggregateSimilarity: Similarity;
  retrievedAt: Timestamp;
}

export interface EvidenceSetPartial {
  state: "partially_answerable";
  evidence: EvidenceCitation[];
  gaps: string[];
  aggregateSimilarity: Similarity;
  retrievedAt: Timestamp;
}

export interface EvidenceSetAbstain {
  state: "not_answerable";
  reason:
    | "below_threshold"
    | "no_data_in_window"
    | "data_excluded_by_consent"
    | "embedding_failure";
  suggestedClarifyingQuestion?: string;
  retrievedAt: Timestamp;
}

export interface Layer2Input {
  evidence: EvidenceSet;
  draft: AnswerDraft;
  psychProfile?: PsychProfile;
  rawUserInput: string;
  now: Timestamp;
  /** Carried forward from Layer1Input; required so D6 cannot be skipped. */
  consentedRetentionWindowDays: number;
}

export type RuleId =
  | "A1" | "A2" | "A3" | "A4" | "A5" | "A6" | "A7" | "A8"
  | "B1" | "B2" | "B3" | "B4" | "B5" | "B6" | "B7" | "B8" | "B9"
  | "C1" | "C2" | "C3" | "C4" | "C5" | "C6" | "C7" | "C8" | "C9"
  | "D1" | "D2" | "D3" | "D4" | "D5" | "D6"
  | "E1" | "E2" | "E3" | "E4" | "E5" | "E6" | "E7" | "E8";

/**
 * The unit a rule produces when it fires.
 * Per the implementation brief: this is a typed record, not a generic Error.
 */
export interface ConstraintViolation {
  ruleId: string;
  severity: "block" | "warn";
  retriable: boolean;
  message: string;
}

export type EscalationRoute =
  | { kind: "crisis_template"; tier: 1 | 2 }
  | { kind: "identity_template" }
  | { kind: "injection_block" }
  | { kind: "human_review" };

/**
 * Layer 2 output — discriminated union per the implementation brief.
 * Note: this is a tightened form of the spec §1.2 contract (single
 * `violation`, not an array; no `passedRules` field).
 */
export type Layer2Output =
  | { verdict: "passed"; draft: AnswerDraft }
  | {
      verdict: "rejected";
      violation: ConstraintViolation;
      escalation?: EscalationRoute;
    };
