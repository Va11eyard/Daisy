/**
 * Loads a worker prompt template by model id and renders it with per-batch
 * variables, few-shot examples, the allowed emotion-label set, and an optional
 * STRICT MODE suffix (used by the orchestrator on re-queue).
 */

import { existsSync, readFileSync } from "node:fs";
import { PATHS } from "./paths.js";
import { selectFewShot, formatFewShot } from "./corpus.js";
import type { BatchSpec } from "./types.js";

const WORKER_FILE_BY_MODEL: Record<string, string> = {
  "claude-opus-4-8": "opus_worker.md",
  "claude-sonnet-4-6": "sonnet_worker.md",
  "gpt-5.5": "gpt_worker.md",
  "gemini-3-1-pro": "gemini_worker.md",
};

export function workerFileForModel(modelId: string): string {
  const file = WORKER_FILE_BY_MODEL[modelId];
  if (!file) throw new Error(`No worker prompt mapped for model "${modelId}"`);
  return `${PATHS.workersDir}/${file}`;
}

const templateCache = new Map<string, string>();

function loadTemplate(modelId: string): string {
  const path = workerFileForModel(modelId);
  const cached = templateCache.get(path);
  if (cached !== undefined) return cached;
  if (!existsSync(path)) throw new Error(`Worker prompt not found: ${path}`);
  const text = readFileSync(path, "utf-8");
  templateCache.set(path, text);
  return text;
}

export interface RenderOptions {
  spec: BatchSpec;
  allowedEmotionLabels: readonly string[];
  strict?: boolean;
}

const STRICT_SUFFIX =
  "\n\nSTRICT MODE: double-check all voice rules. Re-read every VOICE RULE before " +
  "emitting each dialog. Reject your own output and rewrite it if it contains any " +
  "hollow closing, diagnosis phrasing, B9 forbidden substring, or runs outside the " +
  "3–6 sentence / 40–200 token window.";

export function renderWorkerPrompt(opts: RenderOptions): string {
  const { spec, allowedEmotionLabels, strict } = opts;
  const template = loadTemplate(spec.model);

  const emotionList =
    allowedEmotionLabels.length > 0
      ? allowedEmotionLabels.join(", ")
      : "none (use the literal string \"none\" — this is a neutral/off-register dialog)";

  const fewShot = formatFewShot(selectFewShot(spec.tone, allowedEmotionLabels, 3));

  let rendered = template
    .replaceAll("{{COUNT}}", String(spec.count))
    .replaceAll("{{TONE_LABEL}}", spec.tone)
    .replaceAll("{{LANG}}", spec.lang)
    .replaceAll("{{GENERATION_MODE}}", spec.generationMode)
    .replaceAll("{{MODEL_ID}}", spec.model)
    .replaceAll("{{PHASE}}", spec.sessionPhase)
    .replaceAll("{{ALLOWED_EMOTION_LABELS}}", emotionList)
    .replaceAll("{{FEWSHOT}}", fewShot);

  if (strict) rendered += STRICT_SUFFIX;
  return rendered;
}
