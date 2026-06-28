/**
 * Layer 1 — Grounded retrieval + evidence sufficiency gating (spec §1.1).
 *
 * Output is the shared `EvidenceSet` discriminant (evidence / aggregateSimilarity / retrievedAt)
 * so it plugs directly into `Layer2Input.evidence` without reshaping.
 */

import { hallucinationConfig } from "../shared/config.js";
import { EvidenceSetSchema } from "../shared/layerSchemas.zod.js";
import type {
  EvidenceCitation,
  EvidenceSet,
  JournalEntry,
  Layer1Input,
  Similarity,
  Timestamp,
} from "../shared/types.js";

export type { Layer1Input, JournalEntry };

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

function citationFromEntry(entry: JournalEntry, score: number): EvidenceCitation {
  return {
    entryId: entry.entryId,
    sessionId: entry.sessionId,
    timestamp: entry.timestamp,
    textExcerpt: entry.text,
    similarity: score as Similarity,
  };
}

/** Entries whose `timestamp` is within the last `days` from `now` (inclusive). */
function filterWithinRetentionDays(
  entries: readonly JournalEntry[],
  now: Timestamp,
  days: number,
): JournalEntry[] {
  const nowMs = Date.parse(now as string);
  if (Number.isNaN(nowMs)) return [];
  const cutoffMs = nowMs - days * 24 * 60 * 60 * 1000;
  return entries.filter((e) => {
    const t = Date.parse(e.timestamp as string);
    return !Number.isNaN(t) && t >= cutoffMs;
  });
}

/**
 * Runs dense retrieval over `uploadedEntries`, applies dual recency/consent window,
 * gates on `threshold_full` / `threshold_partial`, validates with Zod, returns EvidenceSet.
 */
export async function retrieveGroundedEvidence(input: Layer1Input): Promise<EvidenceSet> {
  const retrievedAt = input.now;

  if (input.uploadedEntries.length === 0) {
    return EvidenceSetSchema.parse({
      state: "not_answerable",
      reason: "no_data_in_window",
      retrievedAt,
    }) as EvidenceSet;
  }

  const consentEligible = filterWithinRetentionDays(
    input.uploadedEntries,
    input.now,
    input.consentedRetentionWindowDays,
  );

  if (consentEligible.length === 0) {
    return EvidenceSetSchema.parse({
      state: "not_answerable",
      reason: "data_excluded_by_consent",
      retrievedAt,
    }) as EvidenceSet;
  }

  const eligible = filterWithinRetentionDays(
    consentEligible,
    input.now,
    hallucinationConfig.recencyWindowDays,
  );

  if (eligible.length === 0) {
    return EvidenceSetSchema.parse({
      state: "not_answerable",
      reason: "no_data_in_window",
      retrievedAt,
    }) as EvidenceSet;
  }

  const { threshold_full, threshold_partial } = hallucinationConfig;

  try {
    const queryVec = await input.embeddingFn(input.query);
    const scored: Array<{ entry: JournalEntry; score: number }> = [];

    for (const entry of eligible) {
      const entryVec = await input.embeddingFn(entry.text);
      const score = cosineSimilarity(queryVec, entryVec);
      scored.push({ entry, score });
    }

    scored.sort((a, b) => b.score - a.score);
    const maxScore = scored[0]!.score;

    if (maxScore < threshold_partial) {
      return EvidenceSetSchema.parse({
        state: "not_answerable",
        reason: "below_threshold",
        retrievedAt,
      }) as EvidenceSet;
    }

    if (maxScore >= threshold_full) {
      const picked = scored.filter((s) => s.score >= threshold_full);
      const evidence = picked.map((s) => citationFromEntry(s.entry, s.score));
      const aggregateSimilarity = mean(picked.map((p) => p.score));
      return EvidenceSetSchema.parse({
        state: "fully_answerable",
        evidence,
        aggregateSimilarity,
        retrievedAt,
      }) as EvidenceSet;
    }

    const picked = scored.filter((s) => s.score >= threshold_partial);
    const evidence = picked.map((s) => citationFromEntry(s.entry, s.score));
    const aggregateSimilarity = mean(picked.map((p) => p.score));
    return EvidenceSetSchema.parse({
      state: "partially_answerable",
      evidence,
      gaps: ["No citation reaches threshold_full; partial context only."],
      aggregateSimilarity,
      retrievedAt,
    }) as EvidenceSet;
  } catch {
    return EvidenceSetSchema.parse({
      state: "not_answerable",
      reason: "embedding_failure",
      retrievedAt,
    }) as EvidenceSet;
  }
}
