/**
 * Parses raw worker-model output into validated Dialog records. Tolerates code
 * fences and stray prose; extracts one JSON object per line. Missing metadata
 * fields are backfilled from the batch spec so the worker can focus on content.
 */

import { type BatchSpec, type Dialog, DialogSchema } from "./types.js";

export interface ParseResult {
  dialogs: Dialog[];
  errors: string[];
}

function stripFences(text: string): string {
  return text.replace(/```(?:json|jsonl)?/gi, "").trim();
}

/** Extract candidate JSON-object substrings (one per top-level {...}). */
function extractObjects(text: string): string[] {
  const objs: string[] = [];
  let depth = 0;
  let start = -1;
  let inStr = false;
  let escape = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        objs.push(text.slice(start, i + 1));
        start = -1;
      }
    }
  }
  return objs;
}

export function parseDialogs(raw: string, spec: BatchSpec): ParseResult {
  const dialogs: Dialog[] = [];
  const errors: string[] = [];
  const objects = extractObjects(stripFences(raw));

  for (const objText of objects) {
    let candidate: Record<string, unknown>;
    try {
      candidate = JSON.parse(objText) as Record<string, unknown>;
    } catch (err) {
      errors.push(`JSON parse error: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }
    // Only treat objects that look like dialogs (have messages) as candidates.
    if (!("messages" in candidate)) continue;

    const backfilled = {
      messages: candidate.messages,
      tone: candidate.tone ?? spec.tone,
      lang: candidate.lang ?? spec.lang,
      generation_mode: candidate.generation_mode ?? spec.generationMode,
      source: candidate.source ?? spec.model,
      emotion_label: candidate.emotion_label ?? "none",
      session_phase: candidate.session_phase ?? spec.sessionPhase,
    };

    const result = DialogSchema.safeParse(backfilled);
    if (result.success) dialogs.push(result.data);
    else errors.push(`Schema error: ${result.error.issues.map((i) => i.message).join("; ")}`);
  }

  return { dialogs, errors };
}
