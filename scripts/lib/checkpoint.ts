/** Idempotency checkpoint: which batches already completed successfully. */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { ensureDir } from "./jsonl.js";
import { PATHS } from "./paths.js";
import type { CheckpointEntry } from "./types.js";

interface CheckpointFile {
  entries: CheckpointEntry[];
}

export function loadCheckpoint(path: string = PATHS.checkpoint): CheckpointFile {
  if (!existsSync(path)) return { entries: [] };
  try {
    const parsed = JSON.parse(readFileSync(path, "utf-8")) as CheckpointFile;
    if (!parsed || !Array.isArray(parsed.entries)) return { entries: [] };
    return parsed;
  } catch {
    return { entries: [] };
  }
}

export function completedBatchKeys(path: string = PATHS.checkpoint): Set<string> {
  const cp = loadCheckpoint(path);
  return new Set(cp.entries.map((e) => `${e.batchId}::${e.tone}::${e.lang}`));
}

export function batchKey(batchId: string, tone: string, lang: string): string {
  return `${batchId}::${tone}::${lang}`;
}

export function recordBatch(entry: CheckpointEntry, path: string = PATHS.checkpoint): void {
  const cp = loadCheckpoint(path);
  const key = batchKey(entry.batchId, entry.tone, entry.lang);
  const filtered = cp.entries.filter((e) => batchKey(e.batchId, e.tone, e.lang) !== key);
  filtered.push(entry);
  ensureDir(path);
  writeFileSync(path, JSON.stringify({ entries: filtered }, null, 2), "utf-8");
}
