/**
 * Layer 5 — Autoformalization + logical consistency check (spec §1.5).
 *
 * Translates each EmotionalClaim into propositional logic via the injected
 * `formalizeFn`, then checks for label-level negation contradictions against
 * the user's knownFacts. After consistency passes, performs a VeriTrans-style
 * round-trip: re-formalize the joined formula and abort if the result drifts
 * (Levenshtein ratio < 0.7).
 *
 * Caller MUST pass a confident Layer4Output. An abstain input is a contract
 * violation.
 */

import {
  ALL_EMOTION_LABELS,
  type EmotionalClaim,
  type EmotionalStateLabel,
  type FormalVocabulary,
  type IntensityLevel,
  type Layer5Input,
  type Layer5Output,
  type PropositionalFact,
  type PropositionalFormula,
  type TemporalRelation,
} from "../shared/types.js";
import { Layer5OutputSchema } from "../shared/layerSchemas.zod.js";

export type { Layer5Input, Layer5Output };

const INTENSITY_LEVELS: IntensityLevel[] = ["mild", "moderate", "severe"];

const TEMPORAL_RELATIONS: TemporalRelation[] = [
  "precedes",
  "co_occurs",
  "persists",
  "recurs",
  "escalates",
  "de_escalates",
  "transitions",
  "oscillates",
];

const LOGICAL_OPERATORS = ["AND", "OR", "NOT", "IMPLIES", "IFF"];

const ROUND_TRIP_RATIO_FLOOR = 0.7;

const EMOTION_LABEL_SET: ReadonlySet<string> = new Set<string>(ALL_EMOTION_LABELS);
const OPERATOR_SET: ReadonlySet<string> = new Set<string>(LOGICAL_OPERATORS);

export function buildFormalVocabulary(): FormalVocabulary {
  return {
    stateLabels: [...ALL_EMOTION_LABELS] as EmotionalStateLabel[],
    intensityLevels: [...INTENSITY_LEVELS],
    temporalRelations: [...TEMPORAL_RELATIONS],
    logicalOperators: [...LOGICAL_OPERATORS],
  };
}

function claimToNL(claim: EmotionalClaim): string {
  const intensity = claim.intensity ? `${claim.intensity} ` : "";
  let text = `User reports ${intensity}${claim.label}`;
  if (claim.trend) {
    text += ` (${claim.trend.direction} over ${claim.trend.window})`;
  } else if (claim.timestamp && claim.timestamp !== "now" && claim.timestamp !== "trend") {
    text += ` at ${claim.timestamp}`;
  }
  return text;
}

function stripArgs(token: string): string {
  const idx = token.indexOf("(");
  return idx >= 0 ? token.slice(0, idx) : token;
}

interface AtomSets {
  positive: Set<string>;
  negative: Set<string>;
}

/**
 * Tokenizes a `formal` propositional string into asserted vs negated label
 * atoms. Args (intensity, etc.) are stripped — contradiction detection is
 * label-level only.
 */
function extractAtoms(formal: string): AtomSets {
  const tokens = formal
    .replace(/[()]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  const positive = new Set<string>();
  const negative = new Set<string>();
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i]!;
    if (tok === "NOT") {
      const next = tokens[i + 1];
      if (next) {
        const label = stripArgs(next);
        if (EMOTION_LABEL_SET.has(label)) negative.add(label);
        i++;
      }
      continue;
    }
    if (OPERATOR_SET.has(tok)) continue;
    const label = stripArgs(tok);
    if (EMOTION_LABEL_SET.has(label)) positive.add(label);
  }
  return { positive, negative };
}

function isContradiction(a: AtomSets, b: AtomSets): boolean {
  for (const x of a.positive) if (b.negative.has(x)) return true;
  for (const x of a.negative) if (b.positive.has(x)) return true;
  return false;
}

function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const m = a.length;
  const n = b.length;
  let prev = new Array<number>(n + 1);
  let curr = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(curr[j - 1]! + 1, prev[j]! + 1, prev[j - 1]! + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}

function levenshteinRatio(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(a, b) / maxLen;
}

function combineFormulas(formulas: PropositionalFormula[]): PropositionalFormula {
  const labelSet = new Set<EmotionalStateLabel>();
  const relationSet = new Set<TemporalRelation>();
  for (const f of formulas) {
    for (const l of f.labels) labelSet.add(l);
    for (const r of f.relations) relationSet.add(r);
  }
  return {
    raw: formulas.map((f) => f.raw).join(" AND "),
    formal: formulas.map((f) => f.formal).filter((s) => s.length > 0).join(" AND "),
    labels: [...labelSet],
    relations: [...relationSet],
  };
}

function ok(formula: PropositionalFormula): Layer5Output {
  return Layer5OutputSchema.parse({ verdict: "consistent", formula }) as Layer5Output;
}

function bad(
  formula: PropositionalFormula,
  conflictingFacts: PropositionalFact[],
): Layer5Output {
  return Layer5OutputSchema.parse({
    verdict: "inconsistent",
    conflictingFacts,
    formula,
  }) as Layer5Output;
}

export async function runAutoformalization(input: Layer5Input): Promise<Layer5Output> {
  if (input.layer4Output.verdict !== "confident") {
    throw new Error("Layer5 must not receive abstain Layer4Output");
  }

  const vocabulary = buildFormalVocabulary();

  const formulas: PropositionalFormula[] = [];
  let formalizeFailures = 0;
  for (const claim of input.claims) {
    try {
      const f = await input.formalizeFn(claimToNL(claim), vocabulary);
      formulas.push(f);
    } catch (err) {
      formalizeFailures++;
      console.warn(
        `[layer5] formalizeFn failed for claim ${claim.label}: ${(err as Error).message}`,
      );
    }
  }

  if (formulas.length === 0 && input.claims.length > 0) {
    return bad(
      {
        raw: `All ${formalizeFailures} claims failed to formalize`,
        formal: "",
        labels: [],
        relations: [],
      },
      [],
    );
  }

  const combined = combineFormulas(formulas);

  const formulaAtoms = extractAtoms(combined.formal);
  const conflictingFacts: PropositionalFact[] = [];
  for (const fact of input.knownFacts) {
    const factAtoms = extractAtoms(fact.formula);
    if (isContradiction(formulaAtoms, factAtoms)) {
      conflictingFacts.push(fact);
    }
  }

  if (conflictingFacts.length > 0) {
    return bad(combined, conflictingFacts);
  }

  let roundTrip: PropositionalFormula;
  try {
    roundTrip = await input.formalizeFn(combined.formal, vocabulary);
  } catch (err) {
    console.warn(`[layer5] round-trip formalize failed: ${(err as Error).message}`);
    return bad(combined, []);
  }

  if (levenshteinRatio(combined.formal, roundTrip.formal) < ROUND_TRIP_RATIO_FLOOR) {
    return bad(combined, []);
  }

  return ok(combined);
}
