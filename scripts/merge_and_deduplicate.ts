/**
 * PHASE 2 — Merge + deduplicate + split.
 *
 * Loads the gold corpus (data/train_v2.jsonl, carried verbatim into train) and
 * the synthesized set (data/synthesized/raw_generated.jsonl). Drops near-dup
 * generated dialogs (cosine > 0.92 vs gold or vs an already-kept generated
 * dialog), performs a stratified 90/10 train/val split per tone, and writes
 * ChatML `text` rows for both files. The system prompt is read from source
 * (src/prompts/voiceContract.ts), never hardcoded here.
 *
 * Usage: ts-node scripts/merge_and_deduplicate.ts
 */

import { writeFileSync } from "node:fs";
import type { DaisyState } from "../src/layers/shared/types.js";
import { buildDaisySystemPrompt } from "../src/prompts/voiceContract.js";
import { cosine, embed } from "./lib/embeddings.js";
import { ensureDir, readJsonl, readJsonlLines } from "./lib/jsonl.js";
import { PATHS } from "./lib/paths.js";
import { type Dialog, DialogSchema, firstAssistantTurn, firstUserTurn } from "./lib/types.js";
import type { SessionPhase } from "./lib/types.js";

const DUP_COSINE = 0.92;
const VAL_FRACTION = 0.1;

const PHASE_TO_STATE: Record<SessionPhase, DaisyState> = {
  opening: "intake",
  mid_session: "disclosure",
  escalation: "crisis",
  de_escalation: "psychoeducation",
  closing: "action_planning",
};

function chatmlText(dialog: Dialog): string {
  const state = PHASE_TO_STATE[dialog.session_phase];
  const system = buildDaisySystemPrompt({ state, forceEnglish: dialog.lang === "en" });
  const user = firstUserTurn(dialog);
  const assistant = firstAssistantTurn(dialog);
  return (
    `<|im_start|>system\n${system}<|im_end|>\n` +
    `<|im_start|>user\n${user}<|im_end|>\n` +
    `<|im_start|>assistant\n${assistant}<|im_end|>\n`
  );
}

function shuffleStable<T>(arr: T[], seed: number): T[] {
  // Deterministic LCG shuffle so re-runs produce identical splits.
  const out = [...arr];
  let s = seed >>> 0;
  const rand = (): number => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

interface ToneRow {
  tone: string;
  trainCount: number;
  valCount: number;
  dedupRemoved: number;
}

async function main(): Promise<void> {
  // Gold corpus: kept verbatim, all to train.
  const goldLines = readJsonlLines(PATHS.referenceCorpus);
  console.log(`[merge] gold corpus: ${goldLines.length} examples (verbatim → train)`);

  // Generated dialogs.
  const rawRows = readJsonl<unknown>(PATHS.rawGenerated);
  const generated: Dialog[] = [];
  for (const r of rawRows) {
    const parsed = DialogSchema.safeParse(r);
    if (parsed.success) generated.push(parsed.data);
  }
  console.log(`[merge] generated: ${generated.length} dialogs`);

  // Embed gold assistant turns for dedup reference.
  const goldEmbeddings: number[][] = [];
  for (const line of goldLines) {
    const obj = JSON.parse(line) as { text?: string };
    const m = /<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>/.exec(obj.text ?? "");
    if (m) goldEmbeddings.push(await embed(m[1]!.trim()));
  }

  // Dedup generated against gold + already-kept generated.
  const keptByTone = new Map<string, Dialog[]>();
  const removedByTone = new Map<string, number>();
  const keptEmbeddings: number[][] = [];

  for (const d of generated) {
    const emb = await embed(firstAssistantTurn(d));
    let dup = false;
    for (const g of goldEmbeddings) {
      if (cosine(emb, g) > DUP_COSINE) { dup = true; break; }
    }
    if (!dup) {
      for (const k of keptEmbeddings) {
        if (cosine(emb, k) > DUP_COSINE) { dup = true; break; }
      }
    }
    if (dup) {
      removedByTone.set(d.tone, (removedByTone.get(d.tone) ?? 0) + 1);
      continue;
    }
    keptEmbeddings.push(emb);
    const arr = keptByTone.get(d.tone) ?? [];
    arr.push(d);
    keptByTone.set(d.tone, arr);
  }

  // Stratified 90/10 split per tone.
  const trainTexts: string[] = goldLines.map((l) => {
    const obj = JSON.parse(l) as { text: string };
    return JSON.stringify({ text: obj.text });
  });
  const valTexts: string[] = [];
  const rows: ToneRow[] = [];

  for (const [tone, dialogs] of [...keptByTone.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const shuffled = shuffleStable(dialogs, tone.length * 7919 + dialogs.length);
    const valN = Math.round(shuffled.length * VAL_FRACTION);
    const valSlice = shuffled.slice(0, valN);
    const trainSlice = shuffled.slice(valN);
    for (const d of trainSlice) trainTexts.push(JSON.stringify({ text: chatmlText(d) }));
    for (const d of valSlice) valTexts.push(JSON.stringify({ text: chatmlText(d) }));
    rows.push({
      tone,
      trainCount: trainSlice.length,
      valCount: valSlice.length,
      dedupRemoved: removedByTone.get(tone) ?? 0,
    });
  }

  ensureDir(PATHS.trainV3);
  writeFileSync(PATHS.trainV3, trainTexts.join("\n") + "\n", "utf-8");
  writeFileSync(PATHS.valV3, valTexts.length > 0 ? valTexts.join("\n") + "\n" : "", "utf-8");

  const totalTrain = trainTexts.length;
  const totalVal = valTexts.length;
  const grandTotal = totalTrain + totalVal;

  console.log("\ntone | train_count | val_count | deduped_removed | % of total");
  console.log("-----|-------------|-----------|-----------------|-----------");
  const goldTrain = goldLines.length;
  console.log(
    `gold_seed | ${goldTrain} | 0 | 0 | ${((goldTrain / grandTotal) * 100).toFixed(1)}%`,
  );
  for (const r of rows) {
    const pct = (((r.trainCount + r.valCount) / grandTotal) * 100).toFixed(1);
    console.log(`${r.tone} | ${r.trainCount} | ${r.valCount} | ${r.dedupRemoved} | ${pct}%`);
  }
  const totalRemoved = [...removedByTone.values()].reduce((s, n) => s + n, 0);
  console.log("-----|-------------|-----------|-----------------|-----------");
  console.log(`TOTAL | ${totalTrain} | ${totalVal} | ${totalRemoved} | 100%`);
  console.log(`\n[merge] wrote ${PATHS.trainV3} (${totalTrain}) and ${PATHS.valV3} (${totalVal})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
