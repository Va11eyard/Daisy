/**
 * Sentence embeddings via @xenova/transformers (all-MiniLM-L6-v2, local,
 * no API). Used for cosine dedup in validate_batch.ts (CHECK 8) and
 * merge_and_deduplicate.ts. Model is lazy-loaded once per process.
 */

import { pipeline } from "@xenova/transformers";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

type FeatureExtractor = (
  text: string | string[],
  opts: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array | number[] }>;

let extractorPromise: Promise<FeatureExtractor> | null = null;

async function getExtractor(): Promise<FeatureExtractor> {
  if (extractorPromise === null) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID) as unknown as Promise<FeatureExtractor>;
  }
  return extractorPromise;
}

export async function embed(text: string): Promise<number[]> {
  const extractor = await getExtractor();
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(out.data as Float32Array);
}

export async function embedMany(texts: string[]): Promise<number[][]> {
  const result: number[][] = [];
  for (const t of texts) {
    result.push(await embed(t));
  }
  return result;
}

export function cosine(a: readonly number[], b: readonly number[]): number {
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
