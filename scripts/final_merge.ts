/**
 * Final merge (revised architecture) — replaces merge_and_deduplicate.ts as the
 * entry point. Reads the gold corpus (data/train_v2.jsonl, carried verbatim
 * into train) plus every data/synthesized/{model_id}/validated.jsonl produced
 * by run_validation.ts. Drops exact assistant-turn duplicates (cosine/@xenova
 * disabled). Performs a stratified 90/10 split per tone,
 * and writes ChatML `text` rows. The system prompt is read from source
 * (src/prompts/voiceContract.ts), never hardcoded here.
 *
 * Usage: ts-node scripts/final_merge.ts
 */

import { existsSync, readdirSync, writeFileSync } from "node:fs";
import type { DaisyState } from "../src/layers/shared/types.js";
import { buildDaisySystemPrompt } from "../src/prompts/voiceContract.js";
import { ensureDir, readJsonl, readJsonlLines } from "./lib/jsonl.js";
import { fromRoot, PATHS } from "./lib/paths.js";
import { loadPlan } from "./lib/plan.js";
import {
  type Dialog,
  DialogSchema,
  firstAssistantTurn,
  firstUserTurn,
  type SessionPhase,
} from "./lib/types.js";

const VAL_FRACTION = 0.1;

/** Normalize assistant text for exact-match dedup (embedding step disabled). */
function normAssistant(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

function extractGoldAssistant(chatmlText: string): string | null {
  const m =
    /<\|im_start\|>assistant\n([\s\S]*?)<\|(?:im_end|redacted_im_end)\|>/.exec(chatmlText);
  return m ? m[1]!.trim() : null;
}

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

function loadAllValidated(modelIds: string[]): Dialog[] {
  const dialogs: Dialog[] = [];
  for (const model of modelIds) {
    const path = fromRoot("data", "synthesized", model, "validated.jsonl");
    if (!existsSync(path)) {
      console.log(`  [skip] no validated.jsonl for ${model}`);
      continue;
    }
    const rows = readJsonl<unknown>(path);
    let n = 0;
    for (const r of rows) {
      const parsed = DialogSchema.safeParse(r);
      if (parsed.success) {
        dialogs.push(parsed.data);
        n += 1;
      }
    }
    console.log(`  [load] ${model}: ${n} validated dialogs`);
  }
  return dialogs;
}

interface ToneRow {
  tone: string;
  trainCount: number;
  valCount: number;
  dedupRemoved: number;
}

async function main(): Promise<void> {
  const plan = loadPlan();

  // Discover model dirs: plan models ∪ any extra dirs under data/synthesized.
  const synthRoot = PATHS.synthesizedDir;
  const discovered = existsSync(synthRoot)
    ? readdirSync(synthRoot, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)
    : [];
  const modelIds = [...new Set([...Object.keys(plan.models), ...discovered])];

  const goldLines = readJsonlLines(PATHS.referenceCorpus);
  console.log(`[final_merge] gold corpus: ${goldLines.length} examples (verbatim → train)`);

  const generated = loadAllValidated(modelIds);
  console.log(`[final_merge] generated (validated): ${generated.length} dialogs`);

  // Exact-match dedup on assistant content (cosine/@xenova disabled — sharp/win32).
  const seenAssistant = new Set<string>();
  for (const line of goldLines) {
    const obj = JSON.parse(line) as { text?: string };
    const assistant = extractGoldAssistant(obj.text ?? "");
    if (assistant) seenAssistant.add(normAssistant(assistant));
  }

  const keptByTone = new Map<string, Dialog[]>();
  const removedByTone = new Map<string, number>();

  for (const d of generated) {
    const key = normAssistant(firstAssistantTurn(d));
    if (seenAssistant.has(key)) {
      removedByTone.set(d.tone, (removedByTone.get(d.tone) ?? 0) + 1);
      continue;
    }
    seenAssistant.add(key);
    const arr = keptByTone.get(d.tone) ?? [];
    arr.push(d);
    keptByTone.set(d.tone, arr);
  }

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
  console.log(`gold_seed | ${goldLines.length} | 0 | 0 | ${((goldLines.length / grandTotal) * 100).toFixed(1)}%`);
  for (const r of rows) {
    const pct = (((r.trainCount + r.valCount) / grandTotal) * 100).toFixed(1);
    console.log(`${r.tone} | ${r.trainCount} | ${r.valCount} | ${r.dedupRemoved} | ${pct}%`);
  }
  const totalRemoved = [...removedByTone.values()].reduce((s, n) => s + n, 0);
  console.log("-----|-------------|-----------|-----------------|-----------");
  console.log(`TOTAL | ${totalTrain} | ${totalVal} | ${totalRemoved} | 100%`);
  console.log(`\n[final_merge] wrote ${PATHS.trainV3} (${totalTrain}) and ${PATHS.valV3} (${totalVal})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
