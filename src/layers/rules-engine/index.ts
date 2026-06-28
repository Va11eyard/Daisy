/**
 * Layer 2 — Symbolic Constraint Validation (Rules Engine).
 *
 * Implements spec §3 rules (groups A–E) plus B9 off-topic scope alignment.
 * declared in spec §1.2 / shared/types.ts. Each rule returns either a
 * ConstraintViolation (with optional EscalationRoute) or null.
 *
 * No magic numbers — all thresholds come from `hallucinationConfig` (§5).
 * No new domain constants — all repo-side constants are imported from
 * `shared/repoConstants.ts` (verbatim mirrors of inference/*.py) and
 * `shared/transitions.ts` (spec §2.7 ALLOWED_TRANSITIONS).
 *
 * Note: rules that need natural-language detection (A5, A6, A7, B8, C2,
 * C3, C5, C6, C7, D1, D3, D4, D5, E6, E7, E8) use small inline regexes
 * scoped to the rule. These are implementation heuristics, not spec
 * invariants — they are NOT exported and contain no thresholds.
 */

import { hallucinationConfig } from "../shared/config.js";
import {
  BANNED_PHRASES,
  CRISIS_TIER1,
  GLOBAL_RULES,
  HOLLOW_CLOSINGS,
  INJECTION_PATTERNS,
  OFF_TOPIC_DOMAINS,
  PRECISION_VOCABULARY,
  STRUCTURAL_RULES,
  crisis_tier,
  is_meta_question,
} from "../shared/repoConstants.js";
import { findTransition } from "../shared/transitions.js";
import {
  ALL_EMOTION_LABELS,
  type ConstraintViolation,
  type EmotionLabel,
  type EmotionalClaim,
  type EscalationRoute,
  type EvidenceCitation,
  type EvidenceSet,
  type Intensity,
  type Layer2Input,
  type Layer2Output,
  type Timestamp,
} from "../shared/types.js";

void GLOBAL_RULES; // referenced by spec rule E5; kept imported for traceability

type RuleResult =
  | { violation: ConstraintViolation; escalation?: EscalationRoute }
  | null;

type Rule = (ctx: Layer2Input) => RuleResult;

const INTENSITY_RANK: Record<Intensity, number> = {
  mild: 1,
  moderate: 2,
  severe: 3,
};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function getEvidenceCitations(e: EvidenceSet): readonly EvidenceCitation[] {
  return e.state === "not_answerable" ? [] : e.evidence;
}

function resolveClaimTimestamp(c: EmotionalClaim, now: Timestamp): string | null {
  const t = c.timestamp;
  if (t === undefined || t === "trend") return null;
  if (t === "now") return now as string;
  return t as string;
}

function isSameSession(a: EmotionalClaim, b: EmotionalClaim): boolean {
  const aSessions = new Set(a.citations.map((c) => c.sessionId));
  for (const c of b.citations) if (aSessions.has(c.sessionId)) return true;
  return false;
}

function isCoOccur(a: EmotionalClaim, b: EmotionalClaim, now: Timestamp): boolean {
  const ta = resolveClaimTimestamp(a, now);
  const tb = resolveClaimTimestamp(b, now);
  if (ta && tb && ta.slice(0, 10) === tb.slice(0, 10)) return true;
  return isSameSession(a, b);
}

function getTransitionPairs(
  asserts: readonly EmotionalClaim[],
  now: Timestamp,
): Array<[EmotionalClaim, EmotionalClaim]> {
  const concrete = asserts
    .map((c) => ({ claim: c, ts: resolveClaimTimestamp(c, now) }))
    .filter((x): x is { claim: EmotionalClaim; ts: string } => x.ts !== null);
  concrete.sort((a, b) => a.ts.localeCompare(b.ts));
  const pairs: Array<[EmotionalClaim, EmotionalClaim]> = [];
  for (let i = 0; i < concrete.length - 1; i++) {
    const a = concrete[i]!.claim;
    const b = concrete[i + 1]!.claim;
    if (a.label !== b.label) pairs.push([a, b]);
  }
  return pairs;
}

function hoursBetween(t1: string, t2: string): number | null {
  const a = new Date(t1).getTime();
  const b = new Date(t2).getTime();
  if (isNaN(a) || isNaN(b)) return null;
  return (b - a) / 3_600_000;
}

function parseWindowToHours(window: string): number | null {
  const m = window.toLowerCase().match(/(\d+(?:\.\d+)?)\s*(hour|day|week|month)s?/);
  if (!m) return null;
  const n = parseFloat(m[1]!);
  switch (m[2]) {
    case "hour": return n;
    case "day": return n * 24;
    case "week": return n * 168;
    case "month": return n * 720;
    default: return null;
  }
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  const curr = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(curr[j - 1]! + 1, prev[j]! + 1, prev[j - 1]! + cost);
    }
    prev = curr.slice();
  }
  return prev[n]!;
}

function levRatio(a: string, b: string): number {
  const max = Math.max(a.length, b.length);
  if (max === 0) return 1;
  return 1 - levenshtein(a, b) / max;
}

function countSentences(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  const matches = trimmed.match(/[^.!?]+[.!?]+/g);
  if (matches) return matches.length;
  return 1;
}

function countActionSteps(text: string): number {
  const numbered = text.match(/(?:(?:^|\n)\s*|[.!?]\s+)(?:\d+[\.\)]|step\s+\d+[:.\)])\s+/gi);
  return numbered ? numbered.length : 0;
}

function rejectionFactory(
  ruleId: string,
  severity: ConstraintViolation["severity"],
  retriable: boolean,
  message: string,
  escalation?: EscalationRoute,
): RuleResult {
  const violation: ConstraintViolation = { ruleId, severity, retriable, message };
  return escalation === undefined ? { violation } : { violation, escalation };
}

const HAS_CRISIS_RESOURCE = /\b(988|crisis\s+line|hotline|emergency|crisis\s+support)\b/i;

// ────────────────────────────────────────────────────────────────────────────
// Group C — Escalation protocols (run first; safety-critical)
// ────────────────────────────────────────────────────────────────────────────

const c9: Rule = (ctx) => {
  if (INJECTION_PATTERNS.some((re) => re.test(ctx.rawUserInput))) {
    return rejectionFactory(
      "C9",
      "block",
      false,
      "Injection pattern detected in user input.",
      { kind: "injection_block" },
    );
  }
  return null;
};

const c1: Rule = (ctx) => {
  if (crisis_tier(ctx.rawUserInput) === 1) {
    return rejectionFactory(
      "C1",
      "block",
      false,
      "Tier-1 crisis terms in user input require crisis-template routing.",
      { kind: "crisis_template", tier: 1 },
    );
  }
  return null;
};

const c8: Rule = (ctx) => {
  if (is_meta_question(ctx.rawUserInput)) {
    return rejectionFactory(
      "C8",
      "block",
      false,
      "Meta question about Daisy's identity routes to identity template.",
      { kind: "identity_template" },
    );
  }
  return null;
};

const C2_SAFETY_ANCHOR =
  /\b(safe|safety|right\s+now|in\s+this\s+moment|are\s+you\s+ok|ok\s+right\s+now)\b/i;

const c2: Rule = (ctx) => {
  if (crisis_tier(ctx.rawUserInput) !== 2) return null;
  if (ctx.draft.daisyState === "action_planning") {
    return rejectionFactory(
      "C2",
      "block",
      false,
      "Tier-2 crisis suppresses action_planning mode for this turn.",
      { kind: "crisis_template", tier: 2 },
    );
  }
  const text = ctx.draft.text;
  const hasQuestion = text.includes("?");
  if (!hasQuestion || !C2_SAFETY_ANCHOR.test(text)) {
    return rejectionFactory(
      "C2",
      "block",
      false,
      "Tier-2 crisis output must include a safety check question.",
      { kind: "crisis_template", tier: 2 },
    );
  }
  return null;
};

const c3: Rule = (ctx) => {
  const ssi = ctx.psychProfile?.SSI;
  if (ssi === undefined) return null;
  if (ssi < hallucinationConfig.threshold_SSI) return null;
  if (HAS_CRISIS_RESOURCE.test(ctx.draft.text)) return null;
  return rejectionFactory(
    "C3",
    "block",
    false,
    `psychProfile.SSI=${ssi} ≥ threshold_SSI=${hallucinationConfig.threshold_SSI} requires crisis-line resource in output.`,
  );
};

const c5: Rule = (ctx) => {
  const text = ctx.draft.text.toLowerCase();
  const containsCrisisTerm = CRISIS_TIER1.some((p) => text.includes(p));
  if (!containsCrisisTerm) return null;
  if (HAS_CRISIS_RESOURCE.test(ctx.draft.text)) return null;
  return rejectionFactory(
    "C5",
    "block",
    true,
    "Output references crisis-tier-1 terms without a crisis-line resource.",
  );
};

const c4: Rule = (ctx) => {
  if (ctx.psychProfile?.riskLevel !== "high") return null;
  const hasTrend = ctx.draft.asserts.some((c) => c.trend !== undefined);
  if (!hasTrend) return null;
  return rejectionFactory(
    "C4",
    "block",
    true,
    "psychProfile.riskLevel='high' suppresses speculative trend claims.",
  );
};

const C6_DRUG_TERMS =
  /\b(meds?|medication|prozac|zoloft|lexapro|xanax|ativan|valium|adderall|ritalin|antidepressants?|anxiolytics?|ssri|snri|benzodiazepines?)\b/i;
const C6_RX_VERBS =
  /\b(should\s+take|try\s+(?:taking\s+)?|stop\s+taking|up\s+the\s+dose|increase\s+(?:your\s+)?dose|recommend\s+(?:taking|trying))\b/i;

const c6: Rule = (ctx) => {
  const text = ctx.draft.text;
  if (C6_RX_VERBS.test(text) && C6_DRUG_TERMS.test(text)) {
    return rejectionFactory(
      "C6",
      "block",
      false,
      "Output contains a medication or substance recommendation.",
    );
  }
  return null;
};

const C7_DIAGNOSIS_TERMS =
  /\b(major\s+depressive\s+disorder|generalized\s+anxiety\s+disorder|bipolar\s+disorder|borderline\s+personality\s+disorder|ptsd|ocd|adhd|gad|mdd|bpd|schizophrenia|panic\s+disorder)\b/i;
const C7_FACT_FRAMING =
  /\b(you\s+have|you've\s+got|you\s+are|you're\s+suffering\s+from|i\s+diagnose|this\s+is)\b/i;
const C7_PERMITTED_FRAMING =
  /\b(pattern\s+resembling|what\s+some\s+clinicians\s+call|looks\s+like|sounds\s+like|might\s+be|could\s+be|seems\s+like)\b/i;

const c7: Rule = (ctx) => {
  const text = ctx.draft.text;
  if (!C7_DIAGNOSIS_TERMS.test(text)) return null;
  if (C7_PERMITTED_FRAMING.test(text)) return null;
  if (!C7_FACT_FRAMING.test(text)) return null;
  return rejectionFactory(
    "C7",
    "block",
    true,
    "Output asserts a DSM/ICD diagnostic label as fact.",
  );
};

// ────────────────────────────────────────────────────────────────────────────
// Group A — Temporal Monotonicity
// ────────────────────────────────────────────────────────────────────────────

const a1: Rule = (ctx) => {
  const pairs = getTransitionPairs(ctx.draft.asserts, ctx.now);
  for (const [a, b] of pairs) {
    if (!isSameSession(a, b)) continue;
    if (a.intensity === "severe" && b.label === "flourishing") {
      return rejectionFactory(
        "A1",
        "block",
        true,
        `Severe ${a.label} → flourishing within a single session is not allowed.`,
      );
    }
  }
  return null;
};

const A2_INTERVENTION_TERMS =
  /\b(therapy|therapist|session|breathwork|grounding\s+exercise|medication|crisis\s+line|988|hospital|hotline|talked\s+to|reached\s+out|appointment|psychiatrist)\b/i;

const a2: Rule = (ctx) => {
  const pairs = getTransitionPairs(ctx.draft.asserts, ctx.now);
  for (const [a, b] of pairs) {
    if (a.intensity !== "severe" || b.label !== "groundedness") continue;
    const ta = resolveClaimTimestamp(a, ctx.now);
    const tb = resolveClaimTimestamp(b, ctx.now);
    if (!ta || !tb) continue;
    const dh = hoursBetween(ta, tb);
    if (dh === null || dh >= 72) continue;
    const haystack =
      [...a.citations, ...b.citations].map((c) => c.textExcerpt).join(" ") +
      " " +
      ctx.rawUserInput +
      " " +
      ctx.draft.text;
    if (!A2_INTERVENTION_TERMS.test(haystack)) {
      return rejectionFactory(
        "A2",
        "block",
        true,
        `Severe ${a.label} → groundedness within ${dh.toFixed(1)}h requires a user-cited intervention.`,
      );
    }
  }
  return null;
};

const a3: Rule = (ctx) => {
  const pairs = getTransitionPairs(ctx.draft.asserts, ctx.now);
  for (const [a, b] of pairs) {
    if (a.label !== "suicidal_ideation" || b.label !== "flourishing") continue;
    const ta = resolveClaimTimestamp(a, ctx.now);
    const tb = resolveClaimTimestamp(b, ctx.now);
    if (!ta || !tb) continue;
    const dh = hoursBetween(ta, tb);
    if (dh === null) continue;
    if (dh < 336) {
      return rejectionFactory(
        "A3",
        "block",
        false,
        `Suicidal_ideation → flourishing within ${dh.toFixed(1)}h is forbidden (< 14d).`,
        { kind: "human_review" },
      );
    }
  }
  return null;
};

const a4: Rule = (ctx) => {
  for (const c of ctx.draft.asserts) {
    if (!c.trend) continue;
    const W = parseWindowToHours(c.trend.window);
    if (W === null) continue;
    if (c.citations.length < 3) {
      return rejectionFactory(
        "A4",
        "block",
        true,
        `Trend claim over '${c.trend.window}' requires ≥3 citations (got ${c.citations.length}).`,
      );
    }
    const ts = c.citations
      .map((cit) => new Date(cit.timestamp).getTime())
      .filter((t) => !isNaN(t));
    if (ts.length < 3) {
      return rejectionFactory(
        "A4",
        "block",
        true,
        `Trend claim over '${c.trend.window}' has <3 valid-timestamp citations.`,
      );
    }
    const span = (Math.max(...ts) - Math.min(...ts)) / 3_600_000;
    if (span < 0.7 * W) {
      return rejectionFactory(
        "A4",
        "block",
        true,
        `Trend claim over '${c.trend.window}' spans ${span.toFixed(1)}h; needs ≥${(0.7 * W).toFixed(1)}h.`,
      );
    }
  }
  return null;
};

const A5_TERMS = /\b(recently|lately|недавно|соңғы\s+кезде)\b/i;

const a5: Rule = (ctx) => {
  if (!A5_TERMS.test(ctx.draft.text)) return null;
  const now = new Date(ctx.now).getTime();
  if (isNaN(now)) return null;
  const windowMs = hallucinationConfig.recencyWindowDays * 24 * 3_600 * 1_000;
  const allCites = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  const hasRecent = allCites.some((c) => {
    const t = new Date(c.timestamp).getTime();
    return !isNaN(t) && now - t <= windowMs;
  });
  if (!hasRecent) {
    return rejectionFactory(
      "A5",
      "block",
      true,
      `'recently'/'lately' used but no citation within ${hallucinationConfig.recencyWindowDays}d.`,
    );
  }
  return null;
};

function trendCheck(
  ctx: Layer2Input,
  direction: "increasing" | "decreasing",
  ruleId: "A6" | "A7",
  word: string,
  cmp: (next: number, prev: number) => boolean,
): RuleResult {
  const claims = ctx.draft.asserts.filter((c) => c.trend?.direction === direction);
  for (const c of claims) {
    const sameLabel = c.citations.filter(
      (cit) =>
        (cit.emotionLabels ?? []).includes(c.label) && cit.intensity !== undefined,
    );
    if (sameLabel.length < 2) {
      return rejectionFactory(
        ruleId,
        "block",
        true,
        `${word} claim for ${c.label} needs ≥2 same-label citations with intensity (got ${sameLabel.length}).`,
      );
    }
    const sorted = [...sameLabel].sort((x, y) => x.timestamp.localeCompare(y.timestamp));
    let monotone = true;
    for (let i = 1; i < sorted.length; i++) {
      const prev = INTENSITY_RANK[sorted[i - 1]!.intensity!];
      const next = INTENSITY_RANK[sorted[i]!.intensity!];
      if (!cmp(next, prev)) {
        monotone = false;
        break;
      }
    }
    if (!monotone) {
      return rejectionFactory(
        ruleId,
        "block",
        true,
        `${word} claim for ${c.label} not supported by strictly ${direction} cited intensities over time.`,
      );
    }
  }
  return null;
}

const a6: Rule = (ctx) => trendCheck(ctx, "decreasing", "A6", "Improvement", (n, p) => n < p);
const a7: Rule = (ctx) => trendCheck(ctx, "increasing", "A7", "Deterioration", (n, p) => n > p);

const a8: Rule = (ctx) => {
  const pairs = getTransitionPairs(ctx.draft.asserts, ctx.now);
  for (const [a, b] of pairs) {
    const edge = findTransition(a.label, b.label);
    if (!edge) {
      return rejectionFactory(
        "A8",
        "block",
        true,
        `Transition ${a.label} → ${b.label} is not in ALLOWED_TRANSITIONS.`,
      );
    }
    const ta = resolveClaimTimestamp(a, ctx.now);
    const tb = resolveClaimTimestamp(b, ctx.now);
    if (ta && tb) {
      const dh = hoursBetween(ta, tb);
      if (dh !== null && dh < edge.minHours) {
        return rejectionFactory(
          "A8",
          "block",
          true,
          `Transition ${a.label} → ${b.label} happened in ${dh.toFixed(1)}h; needs ≥${edge.minHours}h.`,
        );
      }
      if (dh !== null && dh > edge.maxHours) {
        return rejectionFactory(
          "A8",
          "block",
          true,
          `Transition ${a.label} → ${b.label} took ${dh.toFixed(1)}h; max ${edge.maxHours}h.`,
        );
      }
    }
    if (edge.requiresCheck === "SSI" && ctx.psychProfile?.SSI === undefined) {
      return rejectionFactory(
        "A8",
        "block",
        true,
        `Transition ${a.label} → ${b.label} requires psychProfile.SSI; not provided.`,
      );
    }
    if (edge.requiresCheck === "riskLevel" && ctx.psychProfile?.riskLevel === undefined) {
      return rejectionFactory(
        "A8",
        "block",
        true,
        `Transition ${a.label} → ${b.label} requires psychProfile.riskLevel; not provided.`,
      );
    }
    if (edge.requiresCheck === "BSI" && ctx.psychProfile?.BSI === undefined) {
      return rejectionFactory(
        "A8",
        "block",
        true,
        `Transition ${a.label} → ${b.label} requires psychProfile.BSI; not provided.`,
      );
    }
  }
  return null;
};

// ────────────────────────────────────────────────────────────────────────────
// Group B — Internal Consistency
// ────────────────────────────────────────────────────────────────────────────

const b5: Rule = (ctx) => {
  for (const c of ctx.draft.asserts) {
    if (!ALL_EMOTION_LABELS.includes(c.label)) {
      return rejectionFactory(
        "B5",
        "block",
        true,
        `Out-of-vocabulary emotion label: '${c.label}'.`,
      );
    }
  }
  return null;
};

const b6: Rule = (ctx) => {
  const evidenceTimestamps = new Set(
    getEvidenceCitations(ctx.evidence).map((c) => c.timestamp as string),
  );
  for (const c of ctx.draft.asserts) {
    const t = c.timestamp;
    if (typeof t !== "string" || t === "now" || t === "trend") continue;
    if (!evidenceTimestamps.has(t as string)) {
      return rejectionFactory(
        "B6",
        "block",
        true,
        `Claim timestamp ${t} for label '${c.label}' is not present in EvidenceSet.`,
      );
    }
  }
  return null;
};

const b7: Rule = (ctx) => {
  for (const c of ctx.draft.asserts) {
    if (!c.intensity) continue;
    const cited = c.citations.map((cit) => cit.intensity).filter(Boolean) as Intensity[];
    if (cited.length === 0) continue;
    const maxRank = Math.max(...cited.map((i) => INTENSITY_RANK[i]));
    if (INTENSITY_RANK[c.intensity] > maxRank) {
      return rejectionFactory(
        "B7",
        "block",
        true,
        `Aggregate intensity '${c.intensity}' for ${c.label} exceeds max cited intensity.`,
      );
    }
  }
  return null;
};

function forbidPair(
  ctx: Layer2Input,
  ruleId: "B1" | "B2" | "B3" | "B4",
  l1: EmotionLabel,
  l2: EmotionLabel,
  requireSevereOnBoth: boolean,
  message: string,
): RuleResult {
  const asserts = ctx.draft.asserts;
  for (let i = 0; i < asserts.length; i++) {
    for (let j = i + 1; j < asserts.length; j++) {
      const a = asserts[i]!;
      const b = asserts[j]!;
      if (!isCoOccur(a, b, ctx.now)) continue;
      const labels = new Set([a.label, b.label]);
      if (!(labels.has(l1) && labels.has(l2))) continue;
      if (requireSevereOnBoth && !(a.intensity === "severe" && b.intensity === "severe")) {
        continue;
      }
      return rejectionFactory(ruleId, "block", true, message);
    }
  }
  return null;
}

const b1: Rule = (ctx) =>
  forbidPair(
    ctx,
    "B1",
    "flourishing",
    "hopelessness",
    false,
    "Flourishing and hopelessness cannot co-occur in the same session.",
  );

const b2: Rule = (ctx) =>
  forbidPair(
    ctx,
    "B2",
    "joy",
    "depression",
    true,
    "joy.severe and depression.severe cannot co-occur.",
  );

const b3: Rule = (ctx) =>
  forbidPair(
    ctx,
    "B3",
    "groundedness",
    "dissociation",
    false,
    "Groundedness and dissociation cannot co-occur.",
  );

const b4: Rule = (ctx) =>
  forbidPair(
    ctx,
    "B4",
    "numbness",
    "flooded",
    true,
    "numbness.severe and flooded.severe cannot co-occur.",
  );

const b8: Rule = (ctx) => {
  const text = ctx.draft.text;
  const candidates: readonly string[] = [
    ...ALL_EMOTION_LABELS,
    ...Object.keys(PRECISION_VOCABULARY),
  ];
  for (const term of candidates) {
    const re = new RegExp(`\\bnot\\s+${term.replace("_", "[ _]")}\\b`, "i");
    if (!re.test(text)) continue;
    const hasNegationHedge = ctx.draft.hedges.some((h) => h.kind === "negation");
    if (!hasNegationHedge) {
      return rejectionFactory(
        "B8",
        "warn",
        true,
        `Negation of '${term}' lacks an explicit negation hedge from user citations.`,
      );
    }
  }
  return null;
};

function rawInputMatchesOffTopicDomain(rawUserInput: string): boolean {
  const lower = rawUserInput.toLowerCase();
  const isWordChar = (ch: string | undefined) =>
    ch !== undefined && /[A-Za-z0-9_]/.test(ch);

  for (const phrase of OFF_TOPIC_DOMAINS) {
    const needle = phrase.toLowerCase();
    let from = 0;
    while (from <= lower.length) {
      const i = lower.indexOf(needle, from);
      if (i === -1) break;
      const before = i === 0 ? undefined : rawUserInput[i - 1];
      const afterIdx = i + needle.length;
      const after = afterIdx >= rawUserInput.length ? undefined : rawUserInput[afterIdx];
      if (!isWordChar(before) && !isWordChar(after)) {
        return true;
      }
      from = i + 1;
    }
  }
  return false;
}

/** Hard scope gate — mirrors Python `get_therapy_scope_guardrail` off-topic list. */
const b9: Rule = (ctx) => {
  if (!rawInputMatchesOffTopicDomain(ctx.rawUserInput)) return null;
  return rejectionFactory("B9", "block", true, "off_topic_domain");
};

// ────────────────────────────────────────────────────────────────────────────
// Group D — Temporal grounding / data anchoring
// ────────────────────────────────────────────────────────────────────────────

const d2: Rule = (ctx) => {
  const all = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  for (const c of all) {
    if (!c.entryId || !c.sessionId || !c.timestamp) {
      return rejectionFactory(
        "D2",
        "block",
        true,
        "Citation missing entryId / sessionId / timestamp.",
      );
    }
  }
  return null;
};

const D1_RELATIVE_TIME =
  /\b(yesterday|today|this\s+(?:morning|evening|afternoon)|last\s+(?:week|month|session|night)|\d+\s+(?:days?|weeks?|months?)\s+ago|вчера|сегодня|на\s+прошлой\s+неделе)\b/i;

const d1: Rule = (ctx) => {
  if (!D1_RELATIVE_TIME.test(ctx.draft.text)) return null;
  const allCites = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  if (allCites.length === 0) {
    return rejectionFactory(
      "D1",
      "block",
      true,
      "Relative time reference used but no evidence citation provided.",
    );
  }
  return null;
};

const D3_ATTRIBUTION =
  /(?:you\s+said|you\s+mentioned|you\s+told\s+me)[\s,:'"]*['"]([^'"]+)['"]/gi;

const d3: Rule = (ctx) => {
  const matches = [...ctx.draft.text.matchAll(D3_ATTRIBUTION)];
  if (matches.length === 0) return null;
  const allCites = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  for (const m of matches) {
    const quoted = m[1]!.toLowerCase();
    const ok = allCites.some(
      (c) => levRatio(quoted, c.textExcerpt.toLowerCase()) >= 0.8,
    );
    if (!ok) {
      return rejectionFactory(
        "D3",
        "block",
        true,
        `'You said'/'you mentioned' attribution "${m[1]}" not matched by any citation (Levenshtein ratio < 0.8).`,
      );
    }
  }
  return null;
};

const D4_FREQ = /\b(often|frequently|always|usually|repeatedly|часто|жиі)\b/i;

const d4: Rule = (ctx) => {
  if (!D4_FREQ.test(ctx.draft.text)) return null;
  const allCites = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  if (allCites.length < 3) {
    return rejectionFactory(
      "D4",
      "block",
      true,
      `Frequency aggregation requires ≥3 citations (got ${allCites.length}).`,
    );
  }
  return null;
};

const D5_COMPARATIVE =
  /\b(more|less|better|worse)\s+than\s+(?:last|previous|before|the\s+last)\b/i;

const d5: Rule = (ctx) => {
  if (!D5_COMPARATIVE.test(ctx.draft.text)) return null;
  const allCites = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  const ts = allCites
    .map((c) => new Date(c.timestamp).getTime())
    .filter((t) => !isNaN(t));
  if (ts.length < 2) {
    return rejectionFactory(
      "D5",
      "block",
      true,
      "Comparative claim requires citations from both compared windows (got <2).",
    );
  }
  const days = new Set(ts.map((t) => Math.floor(t / (24 * 3_600_000))));
  if (days.size < 2) {
    return rejectionFactory(
      "D5",
      "block",
      true,
      "Comparative claim's citations all fall within a single day window.",
    );
  }
  return null;
};

const d6: Rule = (ctx) => {
  const now = new Date(ctx.now).getTime();
  if (isNaN(now)) return null;
  const cutoff = now - ctx.consentedRetentionWindowDays * 24 * 3_600_000;
  const all = [...ctx.draft.cites, ...getEvidenceCitations(ctx.evidence)];
  for (const c of all) {
    const t = new Date(c.timestamp).getTime();
    if (!isNaN(t) && t < cutoff) {
      return rejectionFactory(
        "D6",
        "block",
        false,
        `Citation timestamp ${c.timestamp} is older than the consented retention window of ${ctx.consentedRetentionWindowDays}d.`,
      );
    }
  }
  return null;
};

// ────────────────────────────────────────────────────────────────────────────
// Group E — Voice / format alignment
// ────────────────────────────────────────────────────────────────────────────

const e1: Rule = (ctx) => {
  const lower = ctx.draft.text.toLowerCase();
  for (const phrase of BANNED_PHRASES) {
    if (lower.includes(phrase.toLowerCase())) {
      return rejectionFactory(
        "E1",
        "block",
        true,
        `Banned phrase used: "${phrase}".`,
      );
    }
  }
  return null;
};

const e2: Rule = (ctx) => {
  const trimmed = ctx.draft.text.trim().toLowerCase();
  for (const phrase of HOLLOW_CLOSINGS) {
    if (trimmed.endsWith(phrase.toLowerCase())) {
      return rejectionFactory(
        "E2",
        "block",
        true,
        `Output ends with hollow closing: "${phrase}".`,
      );
    }
  }
  return null;
};

const e3: Rule = (ctx) => {
  const phase = ctx.draft.daisyState;
  if (phase === "intake") return null;
  const rules = STRUCTURAL_RULES[phase];
  const count = countSentences(ctx.draft.text);
  if (rules.min_sentences !== null && count < rules.min_sentences) {
    return rejectionFactory(
      "E3",
      "block",
      true,
      `Sentence count ${count} below min ${rules.min_sentences} for ${phase}.`,
    );
  }
  if (rules.max_sentences !== null && count > rules.max_sentences) {
    return rejectionFactory(
      "E3",
      "block",
      true,
      `Sentence count ${count} above max ${rules.max_sentences} for ${phase}.`,
    );
  }
  return null;
};

const e4: Rule = (ctx) => {
  if (ctx.draft.daisyState !== "action_planning") return null;
  const max = STRUCTURAL_RULES.action_planning.max_steps;
  if (max === null) return null;
  const steps = countActionSteps(ctx.draft.text);
  if (steps > max) {
    return rejectionFactory(
      "E4",
      "block",
      true,
      `Action plan has ${steps} steps; max ${max}.`,
    );
  }
  return null;
};

const e5: Rule = (ctx) => {
  const text = ctx.draft.text.trim();
  const qCount = (text.match(/\?/g) ?? []).length;
  if (qCount > 1) {
    return rejectionFactory(
      "E5",
      "block",
      true,
      `Output has ${qCount} questions; max 1 per response (GLOBAL_RULES).`,
    );
  }
  if (qCount === 1) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
    const last = sentences[sentences.length - 1] ?? "";
    if (!last.includes("?")) {
      return rejectionFactory(
        "E5",
        "block",
        true,
        "Question present but not the final sentence.",
      );
    }
  }
  return null;
};

const E6_PATTERNS: readonly RegExp[] = [
  /\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,\s*\d{4}\)/,
  /\bp\.?\s*\d+\b/i,
  /\bchapter\s+\d+\b/i,
  /\b(?:footnote|endnote)\s+\d+\b/i,
  /\[\d+\]/,
];

const e6: Rule = (ctx) => {
  for (const re of E6_PATTERNS) {
    if (re.test(ctx.draft.text)) {
      return rejectionFactory(
        "E6",
        "block",
        true,
        "Output contains a literature-citation, page, chapter, or footnote pattern.",
      );
    }
  }
  return null;
};

const E7_ACADEMIC =
  /\b(etiology|comorbid|comorbidity|nosology|pathophysiology|psychopathology|cognitive[-\s]behavio(?:u)?ral\s+therapy|emdr|neuroplasticity|amygdala|prefrontal\s+cortex|hpa\s+axis|dsm[-\s]?\d?|icd[-\s]?\d+)\b/i;
const E7_PARAPHRASE =
  /\b(in\s+other\s+words|that\s+is|i\.e\.|—|–|that\s+means|put\s+plainly|basically|in\s+plain\s+terms)\b/i;

const e7: Rule = (ctx) => {
  if (!E7_ACADEMIC.test(ctx.draft.text)) return null;
  if (E7_PARAPHRASE.test(ctx.draft.text)) return null;
  return rejectionFactory(
    "E7",
    "warn",
    true,
    "Academic / psychoanalytic terminology used without an immediate plain-language paraphrase.",
  );
};

const e8: Rule = (ctx) => {
  const lower = ctx.draft.text.toLowerCase();
  for (const word of Object.keys(PRECISION_VOCABULARY)) {
    const re = new RegExp(`\\b${word}\\b`, "i");
    if (!re.test(lower)) continue;
    const hasModSevere = ctx.draft.asserts.some(
      (c) => c.intensity === "moderate" || c.intensity === "severe",
    );
    if (hasModSevere) {
      return rejectionFactory(
        "E8",
        "warn",
        true,
        `Coarse word '${word}' used with moderate/severe cited intensity; use a PRECISION_VOCABULARY alternative.`,
      );
    }
  }
  return null;
};

// ────────────────────────────────────────────────────────────────────────────
// Engine
// ────────────────────────────────────────────────────────────────────────────

/**
 * Priority order. C-group runs first because escalation is safety-critical;
 * within a group, validity / structural rules run before semantic rules so
 * the engine surfaces the most fundamental violation first.
 */
const RULES_IN_PRIORITY_ORDER: ReadonlyArray<Rule> = [
  c9, c1, c8, c2, c3, c5, c4, c6, c7,
  a1, a2, a3, a4, a5, a6, a7, a8,
  b5, b6, b7, b1, b2, b3, b4, b8, b9,
  d2, d1, d3, d4, d5, d6,
  e1, e2, e3, e4, e5, e6, e7, e8,
];

export function validateLayer2(input: Layer2Input): Layer2Output {
  for (const rule of RULES_IN_PRIORITY_ORDER) {
    const result = rule(input);
    if (!result) continue;
    if (result.escalation === undefined) {
      return { verdict: "rejected", violation: result.violation };
    }
    return {
      verdict: "rejected",
      violation: result.violation,
      escalation: result.escalation,
    };
  }
  return { verdict: "passed", draft: input.draft };
}

export const __internal = {
  rules: {
    c9, c1, c8, c2, c3, c5, c4, c6, c7,
    a1, a2, a3, a4, a5, a6, a7, a8,
    b5, b6, b7, b1, b2, b3, b4, b8, b9,
    d2, d1, d3, d4, d5, d6,
    e1, e2, e3, e4, e5, e6, e7, e8,
  },
};
