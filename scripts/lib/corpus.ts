/**
 * Reads the gold reference corpus (data/train_v2.jsonl, ChatML `text` format),
 * extracts (user, assistant, mode) triples, and selects relevant few-shot
 * examples by lexical overlap with a tone's query terms. Examples are returned
 * verbatim — never invented.
 */

import { PATHS } from "./paths.js";
import { readJsonl } from "./jsonl.js";

export interface CorpusExample {
  user: string;
  assistant: string;
  mode: string;
}

const USER_RE = /<\|im_start\|>user\n([\s\S]*?)<\|im_end\|>/;
const ASSISTANT_RE = /<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>/;
const MODE_RE = /CURRENT INTERACTION MODE:\s*([A-Z_]+)/;

let cache: CorpusExample[] | null = null;

export function loadCorpus(): CorpusExample[] {
  if (cache !== null) return cache;
  const rows = readJsonl<{ text?: string }>(PATHS.referenceCorpus);
  const out: CorpusExample[] = [];
  for (const row of rows) {
    const text = row.text;
    if (typeof text !== "string") continue;
    const u = USER_RE.exec(text);
    const a = ASSISTANT_RE.exec(text);
    if (!u || !a) continue;
    const mode = MODE_RE.exec(text);
    out.push({
      user: u[1]!.trim(),
      assistant: a[1]!.trim(),
      mode: (mode?.[1] ?? "").toLowerCase(),
    });
  }
  cache = out;
  return out;
}

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[^a-zа-яё0-9\s]/giu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

/**
 * Select up to `n` examples whose user+assistant text best overlaps the tone's
 * query terms (tone name + mapped emotion labels). Falls back to highest-scoring
 * examples when fewer than `n` have nonzero overlap ("nearest adjacent tone").
 */
export function selectFewShot(
  tone: string,
  emotionLabels: readonly string[],
  n = 3,
): CorpusExample[] {
  const corpus = loadCorpus();
  const queryTerms = new Set<string>([
    ...tokenize(tone.replace(/_/g, " ")),
    ...emotionLabels.flatMap((l) => tokenize(l.replace(/_/g, " "))),
  ]);

  const scored = corpus.map((ex) => {
    const bag = new Set(tokenize(ex.user + " " + ex.assistant));
    let score = 0;
    for (const q of queryTerms) if (bag.has(q)) score += 1;
    return { ex, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, n).map((s) => s.ex);
}

export function formatFewShot(examples: readonly CorpusExample[]): string {
  if (examples.length === 0) return "(no matching reference examples found)";
  return examples
    .map(
      (ex, i) =>
        `Example ${i + 1} (gold, mode=${ex.mode || "n/a"}):\n` +
        `USER: ${ex.user}\n` +
        `DAISY: ${ex.assistant}`,
    )
    .join("\n\n");
}
