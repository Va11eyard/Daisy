/** Absolute path helpers anchored at the repo root (resilient to CWD). */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

/** scripts/lib → repo root is two levels up. */
export const REPO_ROOT = resolve(here, "..", "..");

export function fromRoot(...parts: string[]): string {
  return resolve(REPO_ROOT, ...parts);
}

export const PATHS = {
  plan: fromRoot("data", "generation_plan.json"),
  checkpoint: fromRoot("data", ".generation_checkpoint.json"),
  referenceCorpus: fromRoot("data", "train_v2.jsonl"),
  rawGenerated: fromRoot("data", "synthesized", "raw_generated.jsonl"),
  synthesizedDir: fromRoot("data", "synthesized"),
  batchesDir: fromRoot("data", "synthesized", "batches"),
  trainV3: fromRoot("data", "train_v3.jsonl"),
  valV3: fromRoot("data", "val_v3.jsonl"),
  datasetReport: fromRoot("data", "dataset_report.md"),
  usageLog: fromRoot("logs", "usage.jsonl"),
  workersDir: fromRoot("prompts", "workers"),
} as const;
