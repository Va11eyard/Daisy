/** Minimal JSONL read/append helpers with directory auto-creation. */

import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";

export function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

export function readJsonlLines(filePath: string): string[] {
  if (!existsSync(filePath)) return [];
  const raw = readFileSync(filePath, "utf-8");
  return raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
}

export function readJsonl<T = unknown>(filePath: string): T[] {
  return readJsonlLines(filePath).map((l) => JSON.parse(l) as T);
}

export function appendJsonl(filePath: string, record: unknown): void {
  ensureDir(filePath);
  appendFileSync(filePath, JSON.stringify(record) + "\n", "utf-8");
}

export function appendJsonlMany(filePath: string, records: readonly unknown[]): void {
  if (records.length === 0) return;
  ensureDir(filePath);
  const block = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  appendFileSync(filePath, block, "utf-8");
}
