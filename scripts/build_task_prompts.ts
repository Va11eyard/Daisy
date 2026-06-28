/**
 * Builds the four self-contained Cursor agent task prompts under
 * prompts/tasks/task_{opus,sonnet,gpt,gemini}.md.
 *
 * Few-shot examples are selected from data/train_v2.jsonl by lexical overlap
 * against each tone's tone_emotion_map entry (corpus.selectFewShot) and embedded
 * directly at build time, so the agent never has to re-search mid-task. The
 * banned-phrase and B9 lists are inlined from source (voiceContract + plan) so
 * the task prompt is fully standalone.
 *
 * Usage: ts-node scripts/build_task_prompts.ts
 */

import { writeFileSync } from "node:fs";
import { BANNED_PHRASES, HOLLOW_CLOSINGS } from "../src/prompts/voiceContract.js";
import { ensureDir } from "./lib/jsonl.js";
import { fromRoot, PATHS } from "./lib/paths.js";
import { allowedEmotionLabels, loadPlan } from "./lib/plan.js";
import { formatFewShot, selectFewShot } from "./lib/corpus.js";
import type { GenerationPlan } from "./lib/types.js";

const MODEL_TO_TASK: Record<string, string> = {
  "claude-opus-4-8": "opus",
  "claude-sonnet-4-6": "sonnet",
  "gpt-5.5": "gpt",
  "gemini-3-1-pro": "gemini",
};

// Hollow/banned phrases the user enumerated in the brief, on top of the source
// HOLLOW_CLOSINGS + BANNED_PHRASES, so the inline list is exhaustive.
const EXTRA_BANNED = [
  "That sounds really hard",
  "That must be difficult",
  "Of course!",
  "I hear you",
  "You're not alone",
];

function bannedListInline(): string {
  const all = [...HOLLOW_CLOSINGS, ...BANNED_PHRASES, ...EXTRA_BANNED];
  const seen = new Set<string>();
  const dedup = all.filter((p) => {
    const k = p.toLowerCase();
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return dedup.map((p) => `  - "${p}"`).join("\n");
}

function voiceContractBlock(includeOffRegister: boolean): string {
  const offRegisterRule = includeOffRegister
    ? "8. off_register dialogs: Daisy answers in 1–2 plain sentences — zero emotional scaffolding, no reflection, no warmth signals, no posed question unless strictly needed. `emotion_label` = \"none\"."
    : "8. off_register dialogs: (not assigned to this worker).";
  return [
    "1. Daisy NEVER uses hollow closings or banned phrases. Banned (any language):",
    bannedListInline(),
    "2. Daisy leads with reflection, not advice. Advice only if the user explicitly asks for it.",
    '3. No diagnosis language: never "you have", "you suffer from", "your disorder", "your condition", "you are diagnosed" (or RU "ты страдаешь", "у тебя диагноз").',
    '4. Use "Daisy noticed…" framing when referencing the user\'s patterns.',
    "5. Response length: 3–6 sentences (40–200 tokens). Never truncate mid-sentence.",
    '6. English register: GMAT/GRE precision — no hedging ("kind of", "sort of", "maybe"), no filler, active voice.',
    "7. Russian register: formal but warm (вы-form), no slang, no calques from English therapy-speak.",
    offRegisterRule,
    "9. B9 — the USER turn must NOT contain any forbidden substring (see list below). Use approved synonyms.",
  ].join("\n");
}

function b9Block(plan: GenerationPlan): string {
  const list = plan.b9_forbidden_substrings.map((s) => `  - "${s}"`).join("\n");
  const syn = Object.entries(plan.b9_approved_synonyms ?? {})
    .map(([bad, good]) => `  - "${bad}" → "${good}"`)
    .join("\n");
  return `Forbidden substrings (scan every USER turn, case-insensitive):\n${list}\n\nApproved synonyms:\n${syn}`;
}

function instructionsBlock(modelId: string, plan: GenerationPlan): string {
  return `
Execute this task start-to-finish with no human intervention.

1. Read \`data/generation_plan.json\`. Your assigned tones, per-tone counts,
   \`lang_split\`, \`generation_mode\`, \`session_phase_by_tone\`, and
   \`tone_emotion_map\` are the source of truth — derive everything from it.
2. Your voice rules and B9 list are inlined below (do not navigate elsewhere).
3. The closed 28-label emotion vocabulary is defined in
   \`src/layers/shared/types.ts\` (ALL_EMOTION_LABELS). \`emotion_label\` MUST be
   one of those labels, and for a given tone it MUST be one of that tone's
   entries in \`tone_emotion_map\` (or "none" for neutral/off_register).
4. Cross-session transitions: if a dialog spans more than one session with an
   emotional-state change, the (from → to) pair MUST exist in
   \`ALLOWED_TRANSITIONS\` (\`src/layers/shared/transitions.ts\`). Hard rules:
   suicidal_ideation → flourishing under 14 days is forbidden (do NOT generate);
   sadness → depression needs ≥168h + riskLevel; flooded → relief needs BSI.
   For single-exchange dialogs (the default here) there is no transition to check.

GENERATION LOOP (idempotent, resumable):
5. Determine resume point: read
   \`data/synthesized/${modelId}/progress.json\` if it exists
   ({ "completed", "skipped", "target", "last_batch" }). Resume from
   \`last_batch + 1\`; NEVER overwrite an existing
   \`data/synthesized/${modelId}/batch_*.jsonl\`.
6. Generate dialogs in batches of ${plan.batch_size}. For each batch N, write
   immediately to \`data/synthesized/${modelId}/batch_{N}.jsonl\` as you go —
   one JSON object per line, no array wrapper. Do NOT accumulate in memory.
   Distribute each tone's count across en/ru exactly per its \`lang_split\`.
7. BEFORE writing each dialog, enforce inline:
   a. B9: scan the USER turn for any forbidden substring → fail.
   b. Transition: if multi-session, validate (from → to) against
      ALLOWED_TRANSITIONS → fail if absent/timing-invalid.
   c. Length: assistant turn 40–200 tokens. With no tokenizer, estimate
      tokens ≈ word_count × 1.3 → fail if outside [40, 200]
      (off_register is exempt from the 40 floor; still cap at 200).
   d. Hollow closing: assistant turn must not contain any banned phrase → fail.
   On failure: regenerate that one dialog ONCE. If it fails again, SKIP it and
   append a line to \`data/synthesized/${modelId}/skipped.log\`
   (tone, lang, reason, the offending text).
8. After every batch, update \`data/synthesized/${modelId}/progress.json\`:
   { "completed": <running count>, "skipped": <running count>,
     "target": ${plan.models[modelId]!.target}, "last_batch": <N> }.
9. When the per-tone counts in the plan are all met (total =
   ${plan.models[modelId]!.target}), print a completion summary
   (completed / skipped / total) and STOP.
`.trim();
}

// Plain-answer style reference for register-collapse buckets. The gold corpus
// contains no neutral/logistical Q&A, so lexical few-shot would surface
// misleading therapeutic dialogs. These are explicitly labelled style
// references (NOT gold few-shot) so the agent does not copy them as register.
const OFF_REGISTER_STYLE = [
  'USER: "What does \'rumination\' mean?"',
  'DAISY: "Rumination is repetitive, looping thinking about the same worry or memory without reaching a resolution."',
  "",
  'USER: "Is our session still at 3pm tomorrow?"',
  'DAISY: "Yes, it\'s still scheduled for 3pm tomorrow."',
  "",
  'USER: "Can you repeat that last part?"',
  'DAISY: "I said the breathing step comes before checking your phone in the morning."',
].join("\n");

function toneSection(tone: string, plan: GenerationPlan): string {
  const entry = plan.distribution[tone]!;
  const phase = plan.session_phase_by_tone[tone] ?? "mid_session";
  const labels = allowedEmotionLabels(plan, tone);
  const labelStr = labels.length > 0 ? labels.join(", ") : 'none (use the literal "none")';

  const head = [
    `### Tone: ${tone}`,
    `- count: ${entry.count} (en ${entry.lang_split.en} / ru ${entry.lang_split.ru})`,
    `- generation_mode: ${entry.generation_mode}`,
    `- session_phase: ${phase}`,
    `- allowed emotion_label values: ${labelStr}`,
    "",
  ];

  // Empty emotion map ⇒ neutral / off_register: no gold few-shot exists.
  if (labels.length === 0) {
    return [
      ...head,
      "No matching gold examples exist for this plain/logistical register.",
      "Style reference (NOT gold — illustrative only; keep replies to 1–2 plain",
      "sentences, no reflection, no warmth, no posed question unless strictly needed):",
      "",
      "```",
      OFF_REGISTER_STYLE,
      "```",
    ].join("\n");
  }

  const fewShot = formatFewShot(selectFewShot(tone, labels, 3));
  return [
    ...head,
    "Few-shot reference (gold corpus — match register; never copy verbatim):",
    "",
    "```",
    fewShot,
    "```",
  ].join("\n");
}

function templateToneOnlyBlock(usesTemplate: boolean): string {
  if (!usesTemplate) return "";
  return `
## template_tone_only dialogs (crisis / suicidal_ideation)
- Daisy turn: calm grounding, no panic, no toxic reassurance, immediate
  validation of the stated feeling, reference a crisis resource exactly once,
  end with one gentle safety-check question.
- Do NOT generate free-form advice or multi-step coping plans.
- The USER turn must NOT contain tier-1 terms verbatim ("suicide", "kill myself",
  "self-harm", "суицид", "убить себя"). Use clinical paraphrase, e.g.
  "I don't want to be here anymore", "мне не хочется больше быть здесь".
`.trim();
}

function outputFormatBlock(modelId: string): string {
  return [
    "One JSON object per line (JSONL), no array wrapper, no code fences:",
    "",
    "```json",
    JSON.stringify({
      messages: [
        { role: "user", content: "..." },
        { role: "assistant", content: "..." },
      ],
      tone: "<assigned tone>",
      lang: "<en|ru>",
      generation_mode: "<standard|template_tone_only|off_register>",
      source: modelId,
      emotion_label: "<one of the tone's allowed labels, or 'none'>",
      session_phase: "<opening|mid_session|escalation|de_escalation|closing>",
    }),
    "```",
  ].join("\n");
}

function buildTaskPrompt(modelId: string, plan: GenerationPlan): string {
  const model = plan.models[modelId]!;
  const tones = model.tones;
  const usesTemplate = tones.some(
    (t) => plan.distribution[t]?.generation_mode === "template_tone_only",
  );
  const usesOffRegister = tones.some(
    (t) => plan.distribution[t]?.generation_mode === "off_register",
  );

  return `# Cursor Agent Task — Daisy Dialog Synthesis (${modelId})

## Header
- **Model:** ${modelId} (assign this model manually in the Cursor model selector)
- **Assigned tones:** ${tones.join(", ")}
- **Target:** ${model.target} dialogs
- **Output dir:** \`data/synthesized/${modelId}/\`
- **Rationale:** ${model.reason}

You are a clinical dialog synthesizer for Daisy, a therapeutic AI assistant.
Your output trains the tone and voice of the model, not its factual knowledge.

## Instructions
${instructionsBlock(modelId, plan)}

## Voice Contract (inline — non-negotiable, applies to every dialog)
${voiceContractBlock(usesOffRegister)}

## B9 Forbidden List (inline)
${b9Block(plan)}

## Assigned tones, counts, and few-shot examples
${tones.map((t) => toneSection(t, plan)).join("\n\n")}

${templateToneOnlyBlock(usesTemplate)}

## Output format
${outputFormatBlock(modelId)}

## Completion
When all assigned per-tone counts are met (total ${model.target}), print:
\`completed=<N> skipped=<N> total=<N>\` and stop.
`.replace(/\n{3,}/g, "\n\n");
}

function main(): void {
  const plan = loadPlan();
  const tasksDir = fromRoot("prompts", "tasks");
  for (const modelId of Object.keys(plan.models)) {
    const taskName = MODEL_TO_TASK[modelId];
    if (!taskName) {
      console.warn(`No task filename mapped for ${modelId}; skipping`);
      continue;
    }
    const out = `${tasksDir}/task_${taskName}.md`;
    ensureDir(out);
    writeFileSync(out, buildTaskPrompt(modelId, plan), "utf-8");
    console.log(`wrote ${out}`);
  }
  // Silence unused-path lint when PATHS is otherwise unreferenced here.
  void PATHS.workersDir;
}

main();
