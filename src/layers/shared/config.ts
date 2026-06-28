/**
 * Default configuration constants — spec §5.
 * All layer logic MUST import thresholds from this object.
 * No magic numbers inline anywhere.
 */

export const hallucinationConfig = {
  // Layer 1 cosine similarity threshold for "fully_answerable".
  // Basis: production RAG tuning typically 0.75–0.85 (DPR-style).
  // TODO: tune after first eval run.
  threshold_full: 0.78,

  // Layer 1 lower retention bound for "partially_answerable".
  // Basis: 20% absolute margin below threshold_full is the standard partial band.
  // TODO: tune after first eval run.
  threshold_partial: 0.6,

  // SSI score that triggers Layer 2 escalation (rule C3).
  // Basis: Beck Scale for Suicidal Ideation (BSS) clinical cutoff ≥ 10
  // (Beck & Steer, 1991; Joiner et al. interpersonal theory).
  // TODO: tune after first eval run.
  threshold_SSI: 10,

  // Layer 3 self-consistency sample count.
  // Basis: SelfCheckGPT recommends K=3–5; K=5 balances signal vs cost.
  // TODO: tune after first eval run.
  K: 5,

  // Cosine similarity floor for self-consistency consensus.
  // consensusScore < this → divergent (abstain).
  // Value ≈ 0.6 is a reasonable starting point for semantic
  // similarity; original Farquhar entropy framing does not apply here.
  // TODO: tune after first eval run
  semanticEntropyThreshold: 0.6,

  // Layer 4 sequence-level entropy ceiling.
  // Basis: arXiv:2510.08146 production threshold range 1.0–1.5.
  // TODO: tune after first eval run.
  sequenceEntropyMax: 1.2,

  // Window (days) anchoring "recently"/"lately" for rule A5.
  // Basis: PHQ-9 / GAD-7 use "past 2 weeks" as the standard current-state window.
  // TODO: tune after first eval run.
  recencyWindowDays: 14,
} as const;

export type HallucinationConfig = typeof hallucinationConfig;
