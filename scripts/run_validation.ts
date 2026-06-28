/**
 * Manual validation runner (revised architecture).
 *
 * Reads the JSONL batches a Cursor agent session wrote to
 * data/synthesized/{model_id}/batch_*.jsonl, runs the same 8 validation checks
 * as validate_batch.ts (CHECK 8 cosine dedup is a no-op there; dedup runs in
 * final_merge.ts), prints a per-tone pass/fail table, and writes the
 * passing dialogs to data/synthesized/{model_id}/validated.jsonl.
 *
 * Usage:
 *   ts-node scripts/run_validation.ts --model claude-opus-4-8
 *   ts-node scripts/run_validation.ts --all
 */

import { existsSync, readdirSync, writeFileSync } from "node:fs";
import { validateBatch } from "./validate_batch.js";
import { ensureDir, readJsonl } from "./lib/jsonl.js";
import { fromRoot } from "./lib/paths.js";
import { loadPlan } from "./lib/plan.js";
import { type Dialog, DialogSchema, type GenerationPlan } from "./lib/types.js";

const PASS_THRESHOLD = 0.85;

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        out[key] = next;
        i++;
      } else {
        out[key] = "true";
      }
    }
  }
  return out;
}

function modelDir(modelId: string): string {
  return fromRoot("data", "synthesized", modelId);
}

function loadModelDialogs(modelId: string): Dialog[] {
  const dir = modelDir(modelId);
  if (!existsSync(dir)) return [];
  const batchFiles = readdirSync(dir)
    .filter((f) => /^batch_\d+\.jsonl$/.test(f))
    .sort();
  const dialogs: Dialog[] = [];
  for (const f of batchFiles) {
    const rows = readJsonl<unknown>(`${dir}/${f}`);
    for (const r of rows) {
      const parsed = DialogSchema.safeParse(r);
      if (parsed.success) dialogs.push(parsed.data);
    }
  }
  return dialogs;
}

interface ToneTally {
  total: number;
  passed: number;
}

async function validateModel(modelId: string, plan: GenerationPlan): Promise<boolean> {
  const dialogs = loadModelDialogs(modelId);
  console.log(`\n=== ${modelId} ===`);
  if (dialogs.length === 0) {
    console.log(`  no batches found in ${modelDir(modelId)} (batch_*.jsonl)`);
    return true;
  }

  const { results, passRate, passed } = await validateBatch(dialogs, plan);

  const perTone = new Map<string, ToneTally>();
  const failReasons = new Map<string, number>();
  for (let i = 0; i < dialogs.length; i++) {
    const tone = dialogs[i]!.tone;
    const r = results[i]!;
    const t = perTone.get(tone) ?? { total: 0, passed: 0 };
    t.total += 1;
    if (r.passed) t.passed += 1;
    perTone.set(tone, t);
    for (const fc of r.failed_checks) failReasons.set(fc, (failReasons.get(fc) ?? 0) + 1);
  }

  console.log("tone | total | passed | failed | pass_rate");
  console.log("-----|-------|--------|--------|----------");
  for (const [tone, t] of [...perTone.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const rate = ((t.passed / t.total) * 100).toFixed(1);
    console.log(`${tone} | ${t.total} | ${t.passed} | ${t.total - t.passed} | ${rate}%`);
  }
  console.log("-----|-------|--------|--------|----------");
  console.log(`TOTAL | ${dialogs.length} | ${passed.length} | ${dialogs.length - passed.length} | ${(passRate * 100).toFixed(1)}%`);

  if (failReasons.size > 0) {
    console.log("\nfailure breakdown:");
    for (const [check, n] of [...failReasons.entries()].sort((a, b) => b[1] - a[1])) {
      console.log(`  ${check}: ${n}`);
    }
  }

  const validatedPath = `${modelDir(modelId)}/validated.jsonl`;
  ensureDir(validatedPath);
  writeFileSync(validatedPath, passed.map((d) => JSON.stringify(d)).join("\n") + (passed.length ? "\n" : ""), "utf-8");
  console.log(`\nwrote ${passed.length} validated dialogs → ${validatedPath}`);

  const ok = passRate >= PASS_THRESHOLD;
  if (!ok) {
    console.log(`\n⚠️  PASS RATE ${(passRate * 100).toFixed(1)}% < 85% for ${modelId} — regenerate weak tones before merge.`);
  }
  return ok;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const plan = loadPlan();
  const models = args.all === "true" ? Object.keys(plan.models) : args.model ? [args.model] : [];

  if (models.length === 0) {
    console.error("Usage: run_validation.ts --model <model_id> | --all");
    process.exit(2);
  }

  let allOk = true;
  for (const model of models) {
    if (!plan.models[model]) {
      console.error(`Unknown model "${model}" (not in generation_plan.json models)`);
      allOk = false;
      continue;
    }
    const ok = await validateModel(model, plan);
    allOk = allOk && ok;
  }

  process.exit(allOk ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
