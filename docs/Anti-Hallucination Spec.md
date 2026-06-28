# Daisy Anti-Hallucination — Implementation Spec

Companion to `Anti-Hallucination Architecture for Daisy.md`. That document
explains *why*; this one fixes the contracts, vocabulary, rules, and
abstention surface needed to *implement* the five layers. It is the
source-of-truth for layer boundaries.

## Scope

This spec defines exactly four artifacts:

1. TypeScript interface contracts (with parallel Zod schemas) for Layers 1–5.
2. The closed emotional vocabulary used by Layer 5 autoformalization.
3. The exhaustive Layer 2 rule set.
4. Five abstention framing templates, one per layer-failure source.

Out of scope: UI, persistence schema, retrieval index implementation,
deployment topology, model selection, feature-flag plumbing.

## Conventions

- Every layer boundary is schema-validated. No layer reads unvalidated text
  from a prior stage.
- Branded primitives prevent cross-mixing (e.g. `UserId` vs `SessionId`).
- Discriminated unions on `state` / `verdict` encode each layer's gating
  decision so the type system enforces the three-state pattern from the
  source doc (full / partial / abstain).
- Existing Daisy module names are referenced verbatim. The TS layer is
  expected to mirror these as constants/enums in a single shared module
  (e.g. `inference/contracts.ts`):
  - `DaisyState` — `inference/state_detector.py`
  - `crisis_tier`, `CRISIS_TIER1`, `CRISIS_TIER2`, `INJECTION_PATTERNS`,
    `is_meta_question`, `OFF_TOPIC_PHRASES`, `therapy_relevant` —
    `inference/safety.py`
  - `BANNED_PHRASES`, `HOLLOW_CLOSINGS`, `PRECISION_VOCABULARY`,
    `STRUCTURAL_RULES`, `GLOBAL_RULES`, `FEW_SHOT_PAIRS` —
    `inference/voice_contract.py`
  - `psychProfile.{ESI, BSI, SSI, MRI, riskLevel}` — used in
    `inference/system_prompt.py`

---

## 1. Layer Contracts (TypeScript + Zod)

### 1.0 Common types

```ts
type Brand<T, K> = T & { readonly __brand: K };

export type UserId     = Brand<string, "UserId">;
export type SessionId  = Brand<string, "SessionId">;
export type EntryId    = Brand<string, "EntryId">;
export type Timestamp  = Brand<string, "ISO8601">;
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
export type EmotionLabel = /* see §2.1, 28 string-literal labels */ string;
export type TemporalRelation =
  | "precedes" | "co_occurs" | "persists" | "recurs"
  | "escalates" | "de_escalates" | "transitions" | "oscillates";
```

```ts
export interface EvidenceCitation {
  entryId: EntryId;
  sessionId: SessionId;
  timestamp: Timestamp;
  textExcerpt: string;            // verbatim or near-verbatim from user
  similarity: Similarity;         // cosine to query embedding
  emotionLabels?: EmotionLabel[]; // pre-extracted, if available
  intensity?: Intensity;
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
  citations: EvidenceCitation[];  // every claim cites at least one
}

export interface AnswerDraft {
  text: string;
  daisyState: DaisyState;
  language: Language;
  cites: EvidenceCitation[];
  asserts: EmotionalClaim[];
  hedges: HedgeMarker[];
}
```

```ts
import { z } from "zod";

const zSimilarity  = z.number().min(0).max(1);
const zConfidence  = z.number().min(0).max(1);
const zTimestamp   = z.string().datetime({ offset: true });
const zEntryId     = z.string().min(1);
const zSessionId   = z.string().min(1);

const zEmotionLabel = z.enum([/* 28 labels from §2.1 */]) as z.ZodEnum<[string, ...string[]]>;
const zIntensity    = z.enum(["mild", "moderate", "severe"]);

const zEvidenceCitation = z.object({
  entryId: zEntryId,
  sessionId: zSessionId,
  timestamp: zTimestamp,
  textExcerpt: z.string().min(1),
  similarity: zSimilarity,
  emotionLabels: zEmotionLabel.array().optional(),
  intensity: zIntensity.optional(),
});

const zEmotionalClaim = z.object({
  subject: z.literal("user"),
  label: zEmotionLabel,
  intensity: zIntensity.optional(),
  timestamp: z.union([zTimestamp, z.literal("now"), z.literal("trend")]).optional(),
  trend: z.object({
    window: z.string(),
    direction: z.enum(["increasing", "decreasing", "stable"]),
  }).optional(),
  citations: zEvidenceCitation.array().min(1),
});
```

### 1.1 Layer 1 — Grounded Retrieval

```ts
export interface Layer1Input {
  userId: UserId;
  query: string;                     // current user message
  daisyState: DaisyState;
  now: Timestamp;
  retrieval: {
    topK: number;
    cosineThreshold: number;         // threshold_full
    partialThreshold: number;        // threshold_partial < threshold_full
    tokenBudget: number;             // up to 32K
    recencyWindowDays: number;       // for "recently"/"lately" anchoring
  };
  index: { handle: string };         // opaque ref to the user's vector index
  consentedRetentionWindowDays: number;
}

export type EvidenceSet =
  | EvidenceSetFull
  | EvidenceSetPartial
  | EvidenceSetAbstain;

export interface EvidenceSetFull {
  state: "fully_answerable";
  evidence: EvidenceCitation[];      // ≥1, all ≥ threshold_full
  aggregateSimilarity: Similarity;
  retrievedAt: Timestamp;
}

export interface EvidenceSetPartial {
  state: "partially_answerable";
  evidence: EvidenceCitation[];      // ≥1, all ≥ threshold_partial, some < threshold_full
  gaps: string[];                    // e.g. "no entries in last 14 days"
  aggregateSimilarity: Similarity;
  retrievedAt: Timestamp;
}

export interface EvidenceSetAbstain {
  state: "not_answerable";
  reason: "below_threshold" | "no_data_in_window" | "data_excluded_by_consent";
  suggestedClarifyingQuestion?: string;
  retrievedAt: Timestamp;
}
```

```ts
export const EvidenceSetSchema = z.discriminatedUnion("state", [
  z.object({
    state: z.literal("fully_answerable"),
    evidence: zEvidenceCitation.array().min(1),
    aggregateSimilarity: zSimilarity,
    retrievedAt: zTimestamp,
  }),
  z.object({
    state: z.literal("partially_answerable"),
    evidence: zEvidenceCitation.array().min(1),
    gaps: z.string().array(),
    aggregateSimilarity: zSimilarity,
    retrievedAt: zTimestamp,
  }),
  z.object({
    state: z.literal("not_answerable"),
    reason: z.enum(["below_threshold", "no_data_in_window", "data_excluded_by_consent"]),
    suggestedClarifyingQuestion: z.string().optional(),
    retrievedAt: zTimestamp,
  }),
]);
```

### 1.2 Layer 2 — Symbolic Constraint Validation

```ts
export interface Layer2Input {
  evidence: EvidenceSet;
  draft: AnswerDraft;                // generated by Qwen using L1 evidence
  psychProfile?: PsychProfile;
  rawUserInput: string;              // original input, for crisis/injection scan
  now: Timestamp;
  /** Carried forward from Layer1Input; required so D6 cannot be skipped. */
  consentedRetentionWindowDays: number;
}

export type Layer2Output = Layer2Passed | Layer2Rejected;

export interface Layer2Passed {
  verdict: "passed";
  draft: AnswerDraft;
}

export interface Layer2Rejected {
  verdict: "rejected";
  violation: ConstraintViolation;    // single violation per call — unambiguous retry / escalation
  escalation?: EscalationRoute;
}

export type RuleId =
  | "A1" | "A2" | "A3" | "A4" | "A5" | "A6" | "A7" | "A8"
  | "B1" | "B2" | "B3" | "B4" | "B5" | "B6" | "B7" | "B8"
  | "C1" | "C2" | "C3" | "C4" | "C5" | "C6" | "C7" | "C8" | "C9"
  | "D1" | "D2" | "D3" | "D4" | "D5" | "D6"
  | "E1" | "E2" | "E3" | "E4" | "E5" | "E6" | "E7" | "E8";

export interface ConstraintViolation {
  ruleId: string;                    // e.g. "A3", "C1"
  severity: "block" | "warn";
  retriable: boolean;                // false → escalation or abstain
  message: string;
}

export type EscalationRoute =
  | { kind: "crisis_template"; tier: 1 | 2 }
  | { kind: "identity_template" }
  | { kind: "injection_block" }
  | { kind: "human_review" };
```

```ts
const zConstraintViolation = z.object({
  ruleId: z.string(),
  severity: z.enum(["block", "warn"]),
  retriable: z.boolean(),
  message: z.string(),
});

export const Layer2OutputSchema = z.discriminatedUnion("verdict", [
  z.object({
    verdict: z.literal("passed"),
    draft: zAnswerDraft,
  }),
  z.object({
    verdict: z.literal("rejected"),
    violation: zConstraintViolation,
    escalation: zEscalationRoute.optional(),
  }),
]);
```

### 1.3 Layer 3 — Self-Consistency Sampling

```ts
export interface Layer3Input {
  draft: AnswerDraft;                // from Layer2Passed
  evidence: EvidenceSet;
  config: {
    K: number;                       // sample count, 3–5 recommended
    temperature: number;
    semanticEntropyThreshold: number;
    minClusterRatio: number;         // e.g. ≥ 0.6 of K must agree
    consortium?: { secondModelHandle: string };
  };
}

export interface AnswerCluster {
  centroidText: string;
  members: AnswerDraft[];
  size: number;
  meanSimilarity: Similarity;
}

export type Layer3Output = Layer3Converged | Layer3Divergent;

export interface Layer3Converged {
  verdict: "converged";
  answer: AnswerDraft;               // selected from majority cluster
  consensusSize: number;
  sampleCount: number;
  semanticEntropy: number;
  consortiumEntropy?: number;
}

export interface Layer3Divergent {
  verdict: "divergent";
  clusters: AnswerCluster[];
  semanticEntropy: number;
  reason: "no_majority" | "high_entropy" | "consortium_disagrees";
}
```

### 1.4 Layer 4 — Confidence-Aware Abstention

```ts
export interface Layer4Input {
  candidate: Layer3Converged;
  signals: {
    sequenceEntropy?: number;        // sequence-level
    perTokenEntropy?: number[];      // optional, for span-level abstain
    probeConfidence?: Confidence;    // LSTM probe over mid-layer activations
  };
  thresholds: {
    sequenceEntropyMax: number;
    probeMin: Confidence;
  };
  calibration: {
    method: "huber" | "platt" | "identity";
    params?: number[];
  };
}

export type Layer4Output = Layer4Confident | Layer4Abstain;

export interface Layer4Confident {
  verdict: "confident";
  answer: AnswerDraft;
  confidence: Confidence;
  signals: { sequenceEntropy?: number; probeConfidence?: Confidence };
}

export interface Layer4Abstain {
  verdict: "abstain_low_confidence";
  confidence: Confidence;
  routedTo: "user_clarification" | "human_review";
  hypothesisHint?: string;           // best-effort soft framing for T4
}
```

### 1.5 Layer 5 — Autoformalization

```ts
export interface Proposition {
  label: EmotionLabel;
  intensity?: Intensity;
  timestamp?: Timestamp | "now";
  evidenceRef?: EntryId;             // required for atoms with timestamp
}

export type PropositionalFormula =
  | { kind: "atom"; proposition: Proposition }
  | { kind: "not"; arg: PropositionalFormula }
  | { kind: "and"  | "or" | "implies" | "iff"; args: PropositionalFormula[] }
  | { kind: "rel"; relation: TemporalRelation; args: Proposition[] };

export interface LogicalInconsistency {
  conflict: [PropositionalFormula, PropositionalFormula];
  witnessFactIds: string[];          // ids in knownFacts that conflict
  explanation: string;
}

export interface Layer5Input {
  answer: AnswerDraft;
  knownFacts: PropositionalFormula[]; // user data + prior validated session conclusions
  factIds: string[];                  // 1:1 with knownFacts
  config: {
    roundTrip: boolean;               // PL → NL → check
    satTimeoutMs: number;
  };
}

export type Layer5Output = Layer5Consistent | Layer5Inconsistent;

export interface Layer5Consistent {
  verdict: "consistent";
  answer: AnswerDraft;
  formal: {
    propositions: PropositionalFormula[];
    cnf: string;                      // serialized CNF
    satResult: "SAT" | "TRIVIALLY_TRUE";
  };
  roundTripValid?: boolean;
}

export interface Layer5Inconsistent {
  verdict: "inconsistent";
  inconsistencies: LogicalInconsistency[];
  satResult: "UNSAT";
  conflictingFactIds: string[];
}
```

### 1.6 Pipeline-level surface (informational)

The driver stitches the layers and emits exactly one of:

```ts
export type AbstentionTemplateId = "T1" | "T2" | "T3" | "T4" | "T5";

export type SurfaceableAnswer =
  | { kind: "answer"; answer: AnswerDraft }
  | { kind: "escalation"; route: EscalationRoute; rendered: string }
  | { kind: "abstention"; template: AbstentionTemplateId; slots: Record<string, string>; rendered: string };

export interface PipelineRun {
  l1: EvidenceSet;
  l2: Layer2Output;
  l3?: Layer3Output;
  l4?: Layer4Output;
  l5?: Layer5Output;
  surface: SurfaceableAnswer;
}
```

---

## 2. Emotional Vocabulary Taxonomy (Layer 5)

Closed vocabulary. The autoformalizer MUST refuse to translate any NL
emotion word that does not map — directly, or via the `PRECISION_VOCABULARY`
synonym table — to a label in §2.1. OOV terms produce rule **B5** in
Layer 2.

### 2.1 State labels (28)

#### A. Plutchik primaries (8)

1. `joy`
2. `trust`
3. `fear`
4. `surprise`
5. `sadness`
6. `disgust`
7. `anger`
8. `anticipation`

#### B. Clinical / psychiatric states (10)

9.  `anxiety` — anticipatory worry beyond contextual cause
10. `depression` — sustained low mood + anhedonia
11. `dissociation`
12. `hopelessness`
13. `shame`
14. `guilt`
15. `grief`
16. `suicidal_ideation` — gating label; presence forces L2 escalation route (rules C1–C3, C5)
17. `panic`
18. `rumination`

#### C. Somatic / regulation markers (6)

19. `exhaustion`
20. `numbness`
21. `hypervigilance`
22. `flooded`
23. `restlessness`
24. `derealization`

#### D. Positive trajectory states (4)

25. `relief`
26. `groundedness`
27. `acceptance`
28. `flourishing`

### 2.2 Intensity levels (3)

| Level | Operational definition |
|---|---|
| `mild` | Acknowledged but does not dominate the user's report; no explicit functional impairment. |
| `moderate` | User names it directly or it visibly shapes behavior across the session. |
| `severe` | Disabling / overwhelming / dominant; explicit functional impairment. |

The mapping from the existing `PRECISION_VOCABULARY` to (label, intensity)
is part of this taxonomy. Examples:

| Precision word | Maps to |
|---|---|
| `grieving` | `(grief, moderate)` |
| `deflated`, `bleak`, `hollow` | `(depression, mild|moderate)` |
| `numb` | `(numbness, moderate)` |
| `bracing`, `dreading` | `(anxiety, moderate)` |
| `hypervigilant` | `(hypervigilance, moderate)` |
| `flooded`, `at capacity`, `spinning` | `(flooded, moderate|severe)` |
| `depleted`, `burned out`, `running on fumes` | `(exhaustion, moderate|severe)` |
| `holding together`, `going through the motions` | `(numbness, mild)` |

### 2.3 Temporal relations (8)

For propositions `P_t = (label, intensity, t, evidenceRef)`:

| Relation | Arity | Semantics |
|---|---|---|
| `precedes(P1, P2)` | 2 | `t1 < t2` |
| `co_occurs(P1, P2)` | 2 | `t1 ≈ t2` (same session window) |
| `persists(P, [t1, t2])` | 1 + interval | `P` holds across the interval; ≥3 cited points required |
| `recurs(P, n, window)` | 1 + counts | `P` appears `n` times in `window` |
| `escalates(P_t1 → P_t2)` | 2 same-label | `intensity(P_t2) > intensity(P_t1)` |
| `de_escalates(P_t1 → P_t2)` | 2 same-label | `intensity(P_t2) < intensity(P_t1)` |
| `transitions(P_t1 → P_t2)` | 2 different-label | direct replacement at adjacent time points |
| `oscillates(P_a, P_b, window)` | 2 + window | alternation within window, ≥2 cycles |

### 2.4 Logical operators

`AND`, `OR`, `NOT`, `IMPLIES`, `IFF`. CNF-convertible; SAT-compatible.

### 2.5 Allowed transitions (Plutchik adjacency / clinical pathway)

A `transitions(A → B)` claim in an `AnswerDraft` is admissible only if
`(A, B)` is in `ALLOWED_TRANSITIONS`. The set is the union of:

- **Plutchik adjacency**: each primary may transition to either of its
  wheel-neighbors (e.g. `joy ↔ trust`, `joy ↔ anticipation`,
  `sadness ↔ disgust`, `fear ↔ surprise`).
- **Clinical pathways**:
  - `sadness → grief → acceptance`
  - `anxiety ↔ panic`
  - `anxiety → rumination`
  - `numbness ↔ dissociation`
  - `dissociation ↔ derealization`
  - `flooded → exhaustion`
  - `hopelessness → suicidal_ideation` is **forbidden as an auto-claim**
    (must be user-stated, not asserted by Daisy)
  - `depression ↛ flourishing` (direct); requires `relief` or
    `acceptance` as intermediate
  - `severe.* ↛ groundedness` within `< 72h` without an explicit
    user-cited intervention (rule A2)
- **Same-label persistence / escalation / de-escalation** is governed by
  `escalates` / `de_escalates` / `persists`, not `transitions`.

Any `transitions(A → B)` not in `ALLOWED_TRANSITIONS` is a Layer-2
violation under rule **A8**.

### 2.6 Formula schema

Already declared in §1.5 (`Proposition`, `PropositionalFormula`). The
serializer used by Layer 5 MUST emit DIMACS CNF for the chosen SAT solver
and MUST round-trip back to NL for the optional `roundTripValid` check.

### 2.7 ALLOWED_TRANSITIONS matrix

```ts
interface TransitionEdge {
  from: EmotionalStateLabel;
  to: EmotionalStateLabel;
  minHours: number;
  maxHours: number;
  requiresCheck: 'SSI' | 'riskLevel' | 'BSI' | null;
}

export const ALLOWED_TRANSITIONS: TransitionEdge[] = [
  // ── 1. Plutchik primary wheel adjacency (bidirectional) ──
  { from: "joy",          to: "trust",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "trust",        to: "joy",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "trust",        to: "fear",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "fear",         to: "trust",        minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "fear",         to: "surprise",     minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "surprise",     to: "fear",         minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "surprise",     to: "sadness",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "sadness",      to: "surprise",     minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "sadness",      to: "disgust",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "disgust",      to: "sadness",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "disgust",      to: "anger",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",        to: "disgust",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",        to: "anticipation", minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anticipation", to: "anger",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anticipation", to: "joy",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "joy",          to: "anticipation", minHours: 0,   maxHours: 168,      requiresCheck: null },

  // ── 2a. Primary → clinical escalations ──
  { from: "fear",         to: "anxiety",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "fear",         to: "panic",           minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "fear",         to: "hypervigilance",  minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anticipation", to: "anxiety",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anticipation", to: "fear",            minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "sadness",      to: "grief",           minHours: 0,   maxHours: Infinity, requiresCheck: null },
  { from: "sadness",      to: "depression",      minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "sadness",      to: "hopelessness",    minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "sadness",      to: "numbness",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",        to: "guilt",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",        to: "shame",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",        to: "restlessness",    minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "disgust",      to: "shame",           minHours: 0,   maxHours: 168,      requiresCheck: null },

  // ── 2b. Clinical ↔ clinical (lateral, escalation, de-escalation) ──
  { from: "anxiety",      to: "panic",           minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "anxiety",      to: "rumination",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anxiety",      to: "hypervigilance",  minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anxiety",      to: "flooded",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anxiety",      to: "exhaustion",      minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "panic",        to: "anxiety",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "panic",        to: "fear",            minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "panic",        to: "flooded",         minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "panic",        to: "exhaustion",      minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "rumination",   to: "anxiety",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "rumination",   to: "depression",      minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "rumination",   to: "exhaustion",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "rumination",   to: "shame",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "rumination",   to: "guilt",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "depression",   to: "sadness",         minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "depression",   to: "hopelessness",    minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "depression",   to: "numbness",        minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "depression",   to: "exhaustion",      minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "depression",   to: "rumination",      minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "depression",   to: "grief",           minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "hopelessness", to: "suicidal_ideation", minHours: 0, maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "hopelessness", to: "depression",      minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "hopelessness", to: "numbness",        minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "hopelessness", to: "exhaustion",      minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "hopelessness", to: "grief",           minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "shame",        to: "guilt",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "shame",        to: "sadness",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "shame",        to: "depression",      minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "shame",        to: "anger",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "shame",        to: "rumination",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "guilt",        to: "shame",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "guilt",        to: "sadness",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "guilt",        to: "rumination",      minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "grief",        to: "sadness",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "grief",        to: "numbness",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "grief",        to: "depression",      minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "grief",        to: "anger",           minHours: 0,   maxHours: 168,      requiresCheck: null },

  // ── 2c. Suicidal_ideation outgoing (always SSI; flourishing gated by rule A3) ──
  { from: "suicidal_ideation", to: "hopelessness",  minHours: 0,   maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "suicidal_ideation", to: "depression",    minHours: 0,   maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "suicidal_ideation", to: "relief",        minHours: 0,   maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "suicidal_ideation", to: "groundedness",  minHours: 24,  maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "suicidal_ideation", to: "acceptance",    minHours: 168, maxHours: Infinity, requiresCheck: 'SSI' },
  { from: "suicidal_ideation", to: "flourishing",   minHours: 336, maxHours: Infinity, requiresCheck: 'SSI' },

  // ── 3. Somatic ↔ emotional ──
  { from: "exhaustion",     to: "numbness",       minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "exhaustion",     to: "depression",     minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "exhaustion",     to: "restlessness",   minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "numbness",       to: "dissociation",   minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "numbness",       to: "sadness",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "numbness",       to: "grief",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "numbness",       to: "flooded",        minHours: 0,   maxHours: 168,      requiresCheck: 'BSI' },
  { from: "dissociation",   to: "numbness",       minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "dissociation",   to: "derealization",  minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "dissociation",   to: "flooded",        minHours: 0,   maxHours: 168,      requiresCheck: 'BSI' },
  { from: "derealization",  to: "dissociation",   minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "derealization",  to: "numbness",       minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "hypervigilance", to: "anxiety",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "hypervigilance", to: "fear",           minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "hypervigilance", to: "exhaustion",     minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "hypervigilance", to: "flooded",        minHours: 0,   maxHours: 168,      requiresCheck: 'BSI' },
  { from: "flooded",        to: "exhaustion",     minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "flooded",        to: "numbness",       minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "flooded",        to: "dissociation",   minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "flooded",        to: "panic",          minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "restlessness",   to: "anxiety",        minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "restlessness",   to: "anger",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "restlessness",   to: "exhaustion",     minHours: 0,   maxHours: 168,      requiresCheck: null },

  // ── 4a. Distress → positive trajectory re-entries ──
  { from: "fear",           to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anxiety",        to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anxiety",        to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "panic",          to: "relief",         minHours: 0,   maxHours: 24,       requiresCheck: null },
  { from: "panic",          to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "sadness",        to: "acceptance",     minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "sadness",        to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "grief",          to: "acceptance",     minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "grief",          to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "depression",     to: "relief",         minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "depression",     to: "acceptance",     minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "depression",     to: "groundedness",   minHours: 72,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "hopelessness",   to: "relief",         minHours: 24,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "hopelessness",   to: "acceptance",     minHours: 168, maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "hopelessness",   to: "groundedness",   minHours: 72,  maxHours: Infinity, requiresCheck: 'riskLevel' },
  { from: "shame",          to: "acceptance",     minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "shame",          to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "guilt",          to: "acceptance",     minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "guilt",          to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",          to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "anger",          to: "acceptance",     minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "anticipation",   to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "rumination",     to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "rumination",     to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "exhaustion",     to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "exhaustion",     to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "numbness",       to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "numbness",       to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "dissociation",   to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "dissociation",   to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "derealization",  to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "derealization",  to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: 'riskLevel' },
  { from: "hypervigilance", to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "hypervigilance", to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "flooded",        to: "relief",         minHours: 0,   maxHours: 24,       requiresCheck: 'BSI' },
  { from: "flooded",        to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: 'BSI' },
  { from: "restlessness",   to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "restlessness",   to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "trust",          to: "groundedness",   minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "trust",          to: "acceptance",     minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "joy",            to: "flourishing",    minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "joy",            to: "acceptance",     minHours: 0,   maxHours: 168,      requiresCheck: null },

  // ── 4b. Positive ↔ positive (intra-trajectory) ──
  { from: "relief",         to: "groundedness",   minHours: 0,   maxHours: Infinity, requiresCheck: null },
  { from: "relief",         to: "acceptance",     minHours: 24,  maxHours: Infinity, requiresCheck: null },
  { from: "relief",         to: "joy",            minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "relief",         to: "flourishing",    minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "groundedness",   to: "acceptance",     minHours: 0,   maxHours: Infinity, requiresCheck: null },
  { from: "groundedness",   to: "flourishing",    minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "groundedness",   to: "joy",            minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "groundedness",   to: "trust",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "groundedness",   to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "acceptance",     to: "flourishing",    minHours: 168, maxHours: Infinity, requiresCheck: null },
  { from: "acceptance",     to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "acceptance",     to: "relief",         minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "acceptance",     to: "joy",            minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "acceptance",     to: "trust",          minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "flourishing",    to: "joy",            minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "flourishing",    to: "groundedness",   minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "flourishing",    to: "acceptance",     minHours: 0,   maxHours: 168,      requiresCheck: null },
  { from: "flourishing",    to: "trust",          minHours: 0,   maxHours: 168,      requiresCheck: null },
];
```

---

## 3. Layer 2 Constraint Rules (39 total)

Each rule has: `id | category | severity | retriable | action`. "Retriable"
means the LLM may regenerate (up to N attempts, configurable per pipeline
run). Non-retriable violations route directly to abstention or escalation.

### Group A — Temporal Monotonicity (8)

| ID | Rule | Severity | Retriable |
|---|---|---|---|
| **A1** | No `transitions(severe.* → flourishing)` within a single session. | block | yes |
| **A2** | No `transitions(severe.* → groundedness)` in `< 72h` without an explicit user-cited intervention. | block | yes |
| **A3** | `transitions(suicidal_ideation → flourishing)` within `< 14d` is forbidden in any form. | block | no → escalation `human_review` |
| **A4** | Trend claims spanning window `W` require ≥ 3 evidence citations spanning ≥ `0.7 * W`. | block | yes |
| **A5** | "Recently" / "lately" / "недавно" / "соңғы кезде" must resolve to ≥ 1 citation with `timestamp ≥ now − retrieval.recencyWindowDays`. | block | yes |
| **A6** | Claims of "improvement" require ≥ 2 same-label citations with strictly decreasing intensity over time. | block | yes |
| **A7** | Claims of "deterioration" require ≥ 2 same-label citations with strictly increasing intensity over time. | block | yes |
| **A8** | Any `transitions(A → B)` MUST be in `ALLOWED_TRANSITIONS` (§2.5). | block | yes |

### Group B — Internal Consistency (8)

| ID | Rule | Severity | Retriable |
|---|---|---|---|
| **B1** | `co_occurs(flourishing, hopelessness)` is forbidden in the same session. | block | yes |
| **B2** | `co_occurs(joy.severe, depression.severe)` is forbidden. | block | yes |
| **B3** | `co_occurs(groundedness, dissociation)` is forbidden. | block | yes |
| **B4** | `co_occurs(numbness.severe, flooded.severe)` is forbidden. | block | yes |
| **B5** | Every `EmotionalClaim.label` MUST be in §2.1. Out-of-vocabulary → reject. | block | yes |
| **B6** | Every `EmotionalClaim.timestamp` (when concrete) MUST resolve to a citation present in `EvidenceSet`. | block | yes |
| **B7** | Aggregate intensity assertions ("very anxious overall") MUST be derivable from cited proposition intensities (≤ max, or ≈ mean). | block | yes |
| **B8** | Negated emotion claims ("not anxious") require explicit citations of *absence* (user said so). Absence of evidence is **not** evidence of absence. | warn | yes |

### Group C — Escalation Protocols (9)

| ID | Rule | Severity | Retriable |
|---|---|---|---|
| **C1** | If `crisis_tier(rawUserInput) == 1` → bypass L3–L5; `escalation = { crisis_template, tier: 1 }`; force `daisyState = "crisis"`. | block | no |
| **C2** | If `crisis_tier(rawUserInput) == 2` → output MUST contain a safety check question; suppress `action_planning` mode for this turn. | block | no |
| **C3** | If `psychProfile.SSI ≥ threshold_SSI` → output MUST include a crisis-line resource (e.g. "988"). | block | no |
| **C4** | If `psychProfile.riskLevel == "high"` → suppress speculative trend claims; trend claims downgraded to single-session observations. | block | yes |
| **C5** | Any output text containing terms in `CRISIS_TIER1` MUST include a crisis-line resource. | block | yes |
| **C6** | No medication, dosage, or substance recommendations. (Medication-name lexicon + verbs `should take` / `try` / `stop taking` / `up the dose`.) | block | no |
| **C7** | No DSM/ICD diagnostic labels asserted as fact. Permitted framing: "a pattern resembling X" / "what some clinicians call X". | block | yes |
| **C8** | If `is_meta_question(rawUserInput)` → `escalation = { identity_template }`; bypass generation. | block | no |
| **C9** | If `INJECTION_PATTERNS` matches `rawUserInput` → `escalation = { injection_block }`; do not enter L1. (Logically pre-L1; enforced via L2 contract for completeness.) | block | no |

### Group D — Temporal Grounding / Data Anchoring (6)

| ID | Rule | Severity | Retriable |
|---|---|---|---|
| **D1** | Every relative time reference ("3 weeks ago", "yesterday", "last session") MUST be backed by ≥ 1 `EvidenceCitation`. | block | yes |
| **D2** | Every citation MUST include `entryId`, `sessionId`, `timestamp`. | block | yes |
| **D3** | "You said X" / "you mentioned X" attributions MUST cite verbatim or near-verbatim user text (Levenshtein ratio ≥ 0.8 vs. `textExcerpt`). | block | yes |
| **D4** | Frequency aggregations ("often", "frequently", "always", "часто", "жиі") require ≥ 3 supporting citations within the implied window. | block | yes |
| **D5** | Comparative claims ("more than last week", "less than before") require citations from BOTH compared windows. | block | yes |
| **D6** | No reference to data older than `Layer2Input.consentedRetentionWindowDays` (must match `Layer1Input`; required on L2 so the rule cannot be skipped). | block | no |

### Group E — Voice / Format Alignment (8) — re-uses existing repo terminology

| ID | Rule | Severity | Retriable |
|---|---|---|---|
| **E1** | Output text MUST NOT contain any phrase in `BANNED_PHRASES` (case-insensitive substring match). | block | yes |
| **E2** | Output MUST NOT end with any phrase in `HOLLOW_CLOSINGS`. | block | yes |
| **E3** | Sentence count MUST satisfy `STRUCTURAL_RULES[draft.daisyState]` (`min_sentences..max_sentences`). | block | yes |
| **E4** | If `draft.daisyState == "action_planning"`, action-step count ≤ `STRUCTURAL_RULES.action_planning.max_steps` (= 3). | block | yes |
| **E5** | At most one question per response; if present, it MUST be the final sentence (per `GLOBAL_RULES`). | block | yes |
| **E6** | No verbatim quotes from books, papers, or clinical literature; no chapter / footnote / citation reproduction (mirrors `CRITICAL OUTPUT RULES` in `inference/system_prompt.py`). | block | yes |
| **E7** | No academic / psychoanalytic terminology without an immediate plain-language paraphrase in the same sentence. | warn | yes |
| **E8** | If a coarse emotion word keyed in `PRECISION_VOCABULARY` appears AND the cited intensity is `moderate` or `severe`, output MUST replace it with one of the precision alternatives. | warn | yes |

**Total: 39 rules.** All rules are evaluable from the inputs of `Layer2Input`
(`AnswerDraft`, `EvidenceSet`, `PsychProfile`, `rawUserInput`, `now`,
`consentedRetentionWindowDays`); none require model internals or external
network calls.

---

## 4. Abstention Framing Templates (5 — by failure source)

One template per layer's failure mode. Each template MUST itself satisfy
the Group-E rules. All templates use the "good" register from
`FEW_SHOT_PAIRS` — observation-led, specific, anchored to what the user
has actually said. No phrase from `BANNED_PHRASES` / `HOLLOW_CLOSINGS`.

The C-group escalations (crisis / identity / injection) produce
**escalation outputs**, not abstentions, and use separate templates that
are out of scope for this section.

### Common metadata schema

```ts
export interface AbstentionTemplate {
  id: AbstentionTemplateId;          // "T1".."T5"
  source: "L1" | "L2" | "L3" | "L4" | "L5";
  slots: string[];
  canonical: { en: string; ru?: string; kk?: string };
  surfaces: boolean;                 // false → internal retry only
  retryAllowed: boolean;
  validatedBy: RuleId[];             // E-group rules this template MUST pass
}
```

---

### T1 — Layer 1 (no/low evidence)

**Trigger:** `EvidenceSet.state == "not_answerable"`, OR `partially_answerable`
with no citations covering the user's specific topic.

**Slots:** `{topic}`, `{anchor}` (concrete time prompt, e.g. "the last
time you noticed it").

**Canonical (en):**

> "I don't have enough from what you've shared with me to answer that with
> any precision. Could you tell me more about {topic} — when {anchor} was,
> what was happening around it?"

**ru / kk pointers:** mirror with "Мне не хватает того, что ты уже
рассказал, чтобы…" / "Сен бұрын айтқан нәрсе бұл сұраққа жауап беруге
жетпейді…"

**Surfaces:** yes. **Retry:** no (surface to user; user response becomes
next-turn evidence).

**validatedBy:** `E1, E2, E3, E5`.

---

### T2 — Layer 2 (rule violation)

**Trigger:** `Layer2Rejected`. Two paths:

- **2a (retriable, internal):** regenerate with the offending claim
  removed and a system-message addendum naming the violated `RuleId`.
  Not surfaced.
- **2b (non-retriable C-group):** route to the corresponding escalation
  template (crisis / identity / injection) — NOT this abstention.
- **2c (retriable A/B/D/E exhausted):** soft-surface fallback below.

**Slots (2c):** `{tentative_observation}` — a weaker, evidence-backed
reframe of the offending claim.

**Canonical (en):**

> "Let me back up. I want to stay closer to what you've actually told me:
> {tentative_observation}. Does that read closer to it than what I just
> said?"

**Surfaces:** path 2c only. **Retry:** path 2a — yes, up to N; paths 2b/2c — no.

**validatedBy:** `E1, E2, E3, E5, E7, E8`.

---

### T3 — Layer 3 (sample divergence)

**Trigger:** `Layer3Divergent`.

**Slots:** `{majority_reframe}` (centroid of the largest cluster, soft-framed),
`{check_question}`.

**Canonical (en):**

> "I have a few different reads of this and I'm not confident which one
> fits you. The pattern I keep coming back to is {majority_reframe}.
> {check_question}"

**Default `{check_question}`:** "Does that match how it actually sits
with you, or is it pointing the wrong way?"

**Surfaces:** yes. **Retry:** no (the user's answer becomes next-turn
evidence; do not silently re-sample).

**validatedBy:** `E1, E2, E3, E5`.

---

### T4 — Layer 4 (low calibrated confidence)

**Trigger:** `Layer4Abstain`.

**Slots:** `{hypothesis}` (best-effort soft framing of the candidate
answer; passed in as `Layer4Abstain.hypothesisHint`),
`{alt_invitation}`.

**Canonical (en):**

> "I'm holding this loosely. From what you've shared, it could be
> {hypothesis} — or it could be something I'm not seeing yet.
> {alt_invitation}"

**Default `{alt_invitation}`:** "Which feels closer — that, or something
else?"

**Surfaces:** yes. If `routedTo == "human_review"`, also flag for
downstream review queue (queue mechanics out of scope).

**validatedBy:** `E1, E2, E3, E5, E7`.

---

### T5 — Layer 5 (logical inconsistency)

**Trigger:** `Layer5Inconsistent`.

**Slots:** `{claim_a}` (the prior claim from `knownFacts` — or earlier in
the answer — that conflicts), `{claim_b}` (the current candidate claim),
`{disambiguating_question}`.

**Canonical (en):**

> "I want to walk something back. A moment ago I noted {claim_a}, and
> just now I'm leaning toward {claim_b} — and those don't quite line up.
> {disambiguating_question}"

**Default `{disambiguating_question}`:** "Which one is more true for you
right now?"

**Surfaces:** yes. **Retry:** no. The user resolves; do NOT silently
retract the prior claim.

**validatedBy:** `E1, E2, E3, E5`.

---

## 5. Default Configuration

All layer logic MUST import thresholds from this single object. No magic
numbers inline anywhere in `inference/` or in TS layer modules.

```ts
export const hallucinationConfig = {
  // Cosine similarity threshold for Layer 1 "fully_answerable" gating.
  // Basis: production RAG tuning typically falls in 0.75–0.85 (DPR-style
  // dense retrievers; "Designing High-Precision LLM RAG Systems", 2026).
  // TODO: tune after first eval run.
  threshold_full: 0.78,

  // Lower retention bound for Layer 1 "partially_answerable". Below this,
  // evidence is treated as ambient and the layer emits "not_answerable".
  // Basis: 20% absolute margin below threshold_full is the standard partial band
  // in the 3-state evidence-sufficiency literature.
  // TODO: tune after first eval run.
  threshold_partial: 0.60,

  // SSI score that triggers Layer 2 escalation routing (rule C3).
  // Basis: Beck Scale for Suicidal Ideation (BSS) clinical cutoff ≥ 10
  // (Beck & Steer, 1991), aligned with Joiner et al.'s interpersonal
  // theory of suicide as a high-risk threshold.
  // TODO: tune after first eval run.
  threshold_SSI: 10,

  // Number of samples for Layer 3 self-consistency voting.
  // Basis: SelfCheckGPT recommends K = 3–5 for cost-effective production
  // sampling; K = 5 balances signal quality vs. per-query cost.
  // TODO: tune after first eval run.
  K: 5,

  // Semantic-entropy ceiling above which Layer 3 marks the candidate
  // divergent and abstains.
  // Basis: Farquhar et al. ("Detecting hallucinations in large language
  // models using semantic entropy", Nature 2024) abstain threshold ≈ 0.6
  // nats for general-domain QA.
  // TODO: tune after first eval run.
  semanticEntropyThreshold: 0.6,

  // Sequence-level (token-aggregate) entropy ceiling for Layer 4.
  // Basis: "Sequence-Level Entropy as a Confidence Signal for LLM
  // Reasoning" (arXiv:2510.08146); production thresholds reported in the
  // 1.0–1.5 range; 1.2 is mid-band and conservative for high-stakes
  // emotional output.
  // TODO: tune after first eval run.
  sequenceEntropyMax: 1.2,

  // Window (in days) that anchors "recently" / "lately" / "недавно" /
  // "соңғы кезде" for Layer 2 rule A5.
  // Basis: 14 days is the standard current-state window in clinical
  // psychometrics (PHQ-9 and GAD-7 both ask about the "past 2 weeks").
  // TODO: tune after first eval run.
  recencyWindowDays: 14,
} as const;
```

---

## Cross-references (for implementers)

- `DaisyState`, state classifier — `inference/state_detector.py`
- `crisis_tier`, `CRISIS_TIER1`, `CRISIS_TIER2`, `INJECTION_PATTERNS`,
  `is_meta_question`, `OFF_TOPIC_PHRASES`, `therapy_relevant` —
  `inference/safety.py`
- `BANNED_PHRASES`, `HOLLOW_CLOSINGS`, `PRECISION_VOCABULARY`,
  `STRUCTURAL_RULES`, `GLOBAL_RULES`, `FEW_SHOT_PAIRS`, `BASE_PERSONA` —
  `inference/voice_contract.py`
- `psychProfile.{ESI, BSI, SSI, MRI, riskLevel}` — used in
  `inference/system_prompt.py` (origin: psych-profile pipeline)

The TS layer SHOULD re-export these as module constants/enums so a single
edit to the Python source updates the TS contract via codegen (codegen
pipeline itself is out of scope for this spec).
