/**
 * Verbatim mirror of the Python repo's voice/safety constants and helpers.
 *
 * Sources (any change in either direction must be kept in sync):
 *   - inference/safety.py         → INJECTION_PATTERNS, CRISIS_TIER1, CRISIS_TIER2,
 *                                   META_WHO_CREATED, crisis_tier, is_meta_question
 *   - inference/voice_contract.py → BANNED_PHRASES, HOLLOW_CLOSINGS,
 *                                   PRECISION_VOCABULARY, STRUCTURAL_RULES, GLOBAL_RULES
 *   - inference/therapy_identity.py → OFF_TOPIC_DOMAINS (get_therapy_scope_guardrail scope list)
 *
 * This module exists so TS layers can import mirrored constants from the Python
 * inference stack alongside Layer 2 hard checks (`OFF_TOPIC_DOMAINS`).
 */

import type { DaisyState } from "./types.js";

export const INJECTION_PATTERNS: readonly RegExp[] = [
  /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
  /you\s+are\s+(now\s+)?(DAN|an?\s+(evil|unrestricted))/i,
  /игнорир(?:уй|овать)\s+(?:все\s+)?(?:предыдущие|прошлые)\s+инструкции/i,
  /забудь\s+(?:все|свои)\s+инструкции/i,
  /(?:системный|твой|ваш)\s+промпт/i,
];

export const CRISIS_TIER1: readonly string[] = [
  "suicide",
  "kill myself",
  "self-harm",
  "суицид",
  "самоубийств",
  "убить себя",
];

export const CRISIS_TIER2: readonly string[] = [
  "hopeless",
  "no reason to live",
  "безнадёжно",
  "никому не нужен",
];

export const META_WHO_CREATED: readonly string[] = [
  "who created you",
  "who made you",
  "кто тебя создал",
  "кто создал тебя",
  "сені кім жасады",
];

export function crisis_tier(text: string): 0 | 1 | 2 {
  const t = (text ?? "").toLowerCase();
  for (const p of CRISIS_TIER1) if (t.includes(p)) return 1;
  for (const p of CRISIS_TIER2) if (t.includes(p)) return 2;
  return 0;
}

export function is_meta_question(text: string): boolean {
  const t = (text ?? "").toLowerCase().trim();
  return META_WHO_CREATED.some((p) => t.includes(p));
}

/**
 * Off-topic domains declined by Daisy scope (`get_therapy_scope_guardrail` in
 * inference/therapy_identity.py). Layer 2 rule B9 matches `rawUserInput` against
 * these phrases (substring match with word-boundary guards).
 */
export const OFF_TOPIC_DOMAINS: readonly string[] = [
  "recipe",
  "recipes",
  "cooking",
  "video games",
  "walkthroughs",
  "sports trivia",
  "coding",
  "homework",
  "finance",
  "legal advice",
  "politics as debate",
];

export const BANNED_PHRASES: readonly string[] = [
  "That makes so much sense!",
  "Absolutely!",
  "I hear you!",
  "That's so valid.",
  "I completely understand.",
  "sort of",
  "kind of",
  "Does that make sense?",
  "It sounds like you're going through a lot.",
  "That must be really tough.",
  "generic silver linings",
  "unsolicited reframes during disclosure",
];

export const HOLLOW_CLOSINGS: readonly string[] = [
  "I'm here for you!",
  "Take care!",
];

export const PRECISION_VOCABULARY: Readonly<Record<string, readonly string[]>> = {
  sad: ["grieving", "deflated", "hollow", "heavy", "bleak", "numb"],
  anxious: ["bracing", "dreading", "hypervigilant", "unsettled"],
  angry: ["frustrated", "indignant", "resentful", "stung"],
  tired: ["depleted", "burned out", "running on fumes"],
  overwhelmed: ["flooded", "spinning", "at capacity"],
  hard: ["a bind", "impossible position", "weight to carry"],
  fine: ["holding together", "going through the motions"],
};

export type Phase = Exclude<DaisyState, "intake">;

export interface PhaseRules {
  min_sentences: number | null;
  max_sentences: number | null;
  max_steps: number | null;
}

export const STRUCTURAL_RULES: Readonly<Record<Phase, PhaseRules>> = {
  disclosure: { min_sentences: 2, max_sentences: 4, max_steps: null },
  psychoeducation: { min_sentences: 4, max_sentences: 8, max_steps: null },
  action_planning: { min_sentences: 3, max_sentences: 6, max_steps: 3 },
  crisis: { min_sentences: null, max_sentences: null, max_steps: null },
};

export const GLOBAL_RULES: readonly string[] = [
  "One question per response, always at the end, never stacked.",
  "No unsolicited reframing before the disclosure phase resolves.",
  "Never close with hollow phrases (e.g., \"I'm here for you!\", \"Take care!\").",
];
