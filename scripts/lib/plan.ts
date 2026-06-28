/** Loads + validates data/generation_plan.json and expands it into batch specs. */

import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { PATHS } from "./paths.js";
import {
  type BatchSpec,
  type GenerationMode,
  type GenerationPlan,
  GenerationPlanSchema,
  type Lang,
  type SessionPhase,
} from "./types.js";

export function loadPlan(planPath: string = PATHS.plan): GenerationPlan {
  const raw = readFileSync(planPath, "utf-8");
  const parsed = JSON.parse(raw);
  return GenerationPlanSchema.parse(parsed);
}

function batchId(tone: string, lang: Lang, index: number): string {
  return `${tone}__${lang}__b${String(index).padStart(3, "0")}`;
}

/**
 * Expands the plan distribution into per-(tone, lang) batches of `batch_size`.
 * Deterministic batch ordering and stable batchIds make orchestration idempotent.
 */
export function expandBatches(plan: GenerationPlan): BatchSpec[] {
  const specs: BatchSpec[] = [];
  const batchSize = plan.batch_size;

  for (const [tone, entry] of Object.entries(plan.distribution)) {
    const phase: SessionPhase = plan.session_phase_by_tone[tone] ?? "mid_session";
    const mode: GenerationMode = entry.generation_mode;

    for (const lang of ["en", "ru"] as const) {
      const total = entry.lang_split[lang];
      let remaining = total;
      let index = 0;
      while (remaining > 0) {
        const count = Math.min(batchSize, remaining);
        specs.push({
          batchId: batchId(tone, lang, index),
          model: entry.assigned_model,
          tone,
          lang,
          count,
          generationMode: mode,
          sessionPhase: phase,
          outputPath: `${PATHS.batchesDir}/${tone}__${lang}.jsonl`,
        });
        remaining -= count;
        index += 1;
      }
    }
  }
  return specs;
}

export function newRunId(): string {
  return randomUUID();
}

export function allowedEmotionLabels(plan: GenerationPlan, tone: string): string[] {
  return plan.tone_emotion_map[tone] ?? [];
}
