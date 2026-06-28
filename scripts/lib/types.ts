/**
 * Shared types + Zod schemas for the dataset-generation pipeline.
 * The emotion vocabulary is derived from src/layers/shared/types.ts — never
 * hardcoded here — so it stays in lockstep with the L2/L5 closed vocabulary.
 */

import { z } from "zod";
import { ALL_EMOTION_LABELS } from "../../src/layers/shared/types.js";

export const EMOTION_LABELS = [...ALL_EMOTION_LABELS] as [string, ...string[]];

export type GenerationMode = "standard" | "template_tone_only" | "off_register";

export const SESSION_PHASES = [
  "opening",
  "mid_session",
  "escalation",
  "de_escalation",
  "closing",
] as const;
export type SessionPhase = (typeof SESSION_PHASES)[number];

export const LANGS = ["en", "ru"] as const;
export type Lang = (typeof LANGS)[number];

/** A single tone entry from data/generation_plan.json `distribution`. */
export const ToneEntrySchema = z.object({
  count: z.number().int().nonnegative(),
  lang_split: z.object({ en: z.number().int().nonnegative(), ru: z.number().int().nonnegative() }),
  generation_mode: z.enum(["standard", "template_tone_only", "off_register"]),
  assigned_model: z.string().min(1),
});
export type ToneEntry = z.infer<typeof ToneEntrySchema>;

export const ModelEntrySchema = z.object({
  tones: z.array(z.string().min(1)),
  target: z.number().int().nonnegative(),
  reason: z.string(),
});

export const GenerationPlanSchema = z.object({
  total_target: z.number().int().positive(),
  existing: z.number().int().nonnegative(),
  to_generate: z.number().int().nonnegative(),
  b9_forbidden_substrings: z.array(z.string().min(1)),
  b9_approved_synonyms: z.record(z.string()).optional(),
  off_register_target: z.number().int().nonnegative(),
  crisis_generation_mode: z.literal("template_tone_only"),
  language_split: z.object({ en: z.number(), ru: z.number() }),
  batch_size: z.number().int().positive(),
  emotion_vocabulary_source: z.string().optional(),
  transitions_source: z.string().optional(),
  system_prompt_source: z.string().optional(),
  tone_emotion_map: z.record(z.array(z.string())),
  session_phase_by_tone: z.record(z.enum(SESSION_PHASES)),
  distribution: z.record(ToneEntrySchema),
  constraints_enforced: z.record(z.number()).optional(),
  models: z.record(ModelEntrySchema),
  checkpoint_file: z.string().min(1),
});
export type GenerationPlan = z.infer<typeof GenerationPlanSchema>;

/** One synthesized dialog as emitted by a worker model and written to JSONL. */
export const DialogSchema = z
  .object({
    messages: z
      .array(
        z.object({
          role: z.enum(["user", "assistant"]),
          content: z.string().min(1),
        }),
      )
      .min(2),
    tone: z.string().min(1),
    lang: z.enum(LANGS),
    generation_mode: z.enum(["standard", "template_tone_only", "off_register"]),
    source: z.string().min(1),
    emotion_label: z.string(),
    session_phase: z.enum(SESSION_PHASES),
  })
  .strict();
export type Dialog = z.infer<typeof DialogSchema>;

export function firstUserTurn(d: Dialog): string {
  const m = d.messages.find((x) => x.role === "user");
  return m ? m.content : "";
}

export function firstAssistantTurn(d: Dialog): string {
  const m = d.messages.find((x) => x.role === "assistant");
  return m ? m.content : "";
}

export interface BatchSpec {
  batchId: string;
  model: string;
  tone: string;
  lang: Lang;
  count: number;
  generationMode: GenerationMode;
  sessionPhase: SessionPhase;
  outputPath: string;
}

export interface CheckpointEntry {
  batchId: string;
  model: string;
  tone: string;
  lang: Lang;
  count: number;
  timestamp: string;
  outputPath: string;
}

export interface ValidationResult {
  id: string;
  passed: boolean;
  failed_checks: string[];
}
