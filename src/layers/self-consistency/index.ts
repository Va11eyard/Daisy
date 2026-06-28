/**
 * Layer 3 — Self-consistency sampling + semantic voting (spec §1.3).
 *
 * Generates K samples in parallel, embeds them, and decides convergence by
 * mean pairwise cosine similarity. Caller MUST NOT pass a `not_answerable`
 * EvidenceSet — that is a contract violation, not a runtime error to wrap.
 */

import { hallucinationConfig } from "../shared/config.js";
import { Layer3OutputSchema } from "../shared/layerSchemas.zod.js";
import type { Layer3Input, Layer3Output } from "../shared/types.js";

export type { Layer3Input, Layer3Output };

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

function divergent(samples: string[]): Layer3Output {
  return Layer3OutputSchema.parse({
    verdict: "divergent",
    reason: "sample_divergence",
    samples,
  }) as Layer3Output;
}

/**
 * Runs K-sample self-consistency over `input.query`. Returns a converged
 * candidate (centroid sample) when mean pairwise similarity ≥
 * `hallucinationConfig.semanticEntropyThreshold`, otherwise abstains.
 */
export async function runSelfConsistency(input: Layer3Input): Promise<Layer3Output> {
  if (input.evidenceSet.state === "not_answerable") {
    throw new Error("Layer3 must not receive not_answerable EvidenceSet");
  }

  const K = hallucinationConfig.K;
  const minSuccess = Math.ceil(K / 2);

  const settled = await Promise.allSettled(
    Array.from({ length: K }, () => input.generateFn(input.query)),
  );

  const samples: string[] = [];
  for (const r of settled) {
    if (r.status === "fulfilled") samples.push(r.value);
  }

  if (samples.length < minSuccess) {
    return divergent(samples);
  }

  const embeddings = await Promise.all(samples.map((s) => input.embedFn(s)));

  const n = samples.length;
  const sim: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    sim[i]![i] = 1;
    for (let j = i + 1; j < n; j++) {
      const s = cosineSimilarity(embeddings[i]!, embeddings[j]!);
      sim[i]![j] = s;
      sim[j]![i] = s;
    }
  }

  let pairSum = 0;
  let pairCount = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      pairSum += sim[i]![j]!;
      pairCount++;
    }
  }
  const consensusScore = pairCount === 0 ? 1 : pairSum / pairCount;

  if (consensusScore < hallucinationConfig.semanticEntropyThreshold) {
    return divergent(samples);
  }

  let bestIdx = 0;
  let bestMean = -Infinity;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      sum += sim[i]![j]!;
      count++;
    }
    // Tie-break: strict `>` keeps the first-seen index.
    const mean = count === 0 ? 1 : sum / count;
    if (mean > bestMean) {
      bestMean = mean;
      bestIdx = i;
    }
  }

  return Layer3OutputSchema.parse({
    verdict: "converged",
    candidate: samples[bestIdx]!,
    consensusScore,
  }) as Layer3Output;
}
