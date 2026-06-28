import { describe, expect, it, vi } from "vitest";

import {
  ALL_EMOTION_LABELS,
  type EmotionalClaim,
  type EntryId,
  type FormalVocabulary,
  type Layer4Output,
  type PropositionalFact,
  type PropositionalFormula,
  type SessionId,
  type Timestamp,
} from "../shared/types.js";
import { Layer5OutputSchema } from "../shared/layerSchemas.zod.js";
import { buildFormalVocabulary, runAutoformalization } from "./index.js";

const NOW = "2026-04-27T12:00:00.000Z" as Timestamp;

const CONFIDENT: Layer4Output = {
  verdict: "confident",
  answer: "You've been cycling between joy and exhaustion this week.",
  confidence: 0.85,
};

const ABSTAIN: Layer4Output = { verdict: "abstain_low_confidence", template: "T4" };

function claim(
  label: EmotionalClaim["label"],
  intensity?: EmotionalClaim["intensity"],
): EmotionalClaim {
  return {
    subject: "user",
    label,
    ...(intensity ? { intensity } : {}),
    citations: [
      {
        entryId: "e1" as EntryId,
        sessionId: "s1" as SessionId,
        timestamp: NOW,
        textExcerpt: "x",
        similarity: 0.9 as never,
      },
    ],
  };
}

function fact(id: string, formula: string): PropositionalFact {
  return {
    id,
    formula,
    sourceEntryId: "e0",
    sessionId: "s0",
    timestamp: "2026-04-20T10:00:00.000Z",
  };
}

function pf(formal: string, labels: string[] = [], raw = "x"): PropositionalFormula {
  return { raw, formal, labels: labels as never, relations: [] };
}

/**
 * Stateful formalizer mock. Each entry corresponds to one call (claims first,
 * then the round-trip call). `null` means "throw".
 */
function makeFormalizer(plan: Array<PropositionalFormula | null>) {
  let i = 0;
  return vi.fn(async (text: string, _vocab: FormalVocabulary): Promise<PropositionalFormula> => {
    const next = plan[i++];
    if (next === null || next === undefined) {
      throw new Error("formalizer plan exhausted");
    }
    return { ...next, raw: text };
  });
}

describe("runAutoformalization", () => {
  it("returns consistent when all claims have no contradictions in knownFacts", async () => {
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy", "moderate")],
      knownFacts: [fact("f1", "groundedness(any)")],
      formalizeFn: makeFormalizer([
        pf("joy(moderate)", ["joy"]),
        pf("joy(moderate)"),
      ]),
    });
    expect(result.verdict).toBe("consistent");
    if (result.verdict === "consistent") {
      expect(result.formula.formal).toBe("joy(moderate)");
      expect(result.formula.labels).toContain("joy");
    }
  });

  it("flags one conflicting fact when a claim contradicts a known NOT-fact", async () => {
    const conflict = fact("f1", "NOT joy(any)");
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy")],
      knownFacts: [fact("f0", "groundedness(any)"), conflict],
      formalizeFn: makeFormalizer([
        pf("joy(moderate)", ["joy"]),
        pf("joy(moderate)"),
      ]),
    });
    expect(result.verdict).toBe("inconsistent");
    if (result.verdict === "inconsistent") {
      expect(result.conflictingFacts).toEqual([conflict]);
    }
  });

  it("returns all conflicting facts when multiple contradictions exist", async () => {
    const c1 = fact("f1", "NOT joy(any)");
    const c2 = fact("f2", "NOT anger(any)");
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy"), claim("anger")],
      knownFacts: [c1, c2, fact("f3", "groundedness(any)")],
      formalizeFn: makeFormalizer([
        pf("joy(moderate)", ["joy"]),
        pf("anger(moderate)", ["anger"]),
        pf("joy(moderate) AND anger(moderate)"),
      ]),
    });
    expect(result.verdict).toBe("inconsistent");
    if (result.verdict === "inconsistent") {
      expect(result.conflictingFacts.map((f) => f.id).sort()).toEqual(["f1", "f2"]);
    }
  });

  it("skips claims whose formalizeFn throws and proceeds with the rest", async () => {
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy"), claim("relief")],
      knownFacts: [fact("f1", "groundedness(any)")],
      formalizeFn: makeFormalizer([
        null,
        pf("relief(moderate)", ["relief"]),
        pf("relief(moderate)"),
      ]),
    });
    expect(result.verdict).toBe("consistent");
    if (result.verdict === "consistent") {
      expect(result.formula.formal).toBe("relief(moderate)");
      expect(result.formula.labels).toEqual(["relief"]);
    }
  });

  it("returns inconsistent with conflictingFacts: [] when ALL claims fail to formalize", async () => {
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy"), claim("relief")],
      knownFacts: [fact("f1", "NOT joy(any)")],
      formalizeFn: makeFormalizer([null, null]),
    });
    expect(result.verdict).toBe("inconsistent");
    if (result.verdict === "inconsistent") {
      expect(result.conflictingFacts).toEqual([]);
      expect(result.formula.raw).toMatch(/failed to formalize/i);
      expect(result.formula.formal).toBe("");
    }
  });

  it("returns inconsistent when round-trip Levenshtein ratio < 0.7", async () => {
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy", "moderate")],
      knownFacts: [],
      formalizeFn: makeFormalizer([
        pf("joy(moderate)", ["joy"]),
        pf("totally_different_output_xyz"),
      ]),
    });
    expect(result.verdict).toBe("inconsistent");
    if (result.verdict === "inconsistent") {
      expect(result.conflictingFacts).toEqual([]);
    }
  });

  it("returns consistent when round-trip Levenshtein ratio >= 0.7", async () => {
    const result = await runAutoformalization({
      layer4Output: CONFIDENT,
      claims: [claim("joy", "moderate")],
      knownFacts: [],
      formalizeFn: makeFormalizer([
        pf("joy(moderate)", ["joy"]),
        pf("joy(moderate)"),
      ]),
    });
    expect(result.verdict).toBe("consistent");
  });

  it("throws when caller passes an abstain Layer4Output", async () => {
    await expect(
      runAutoformalization({
        layer4Output: ABSTAIN,
        claims: [claim("joy")],
        knownFacts: [],
        formalizeFn: makeFormalizer([pf("joy(moderate)", ["joy"])]),
      }),
    ).rejects.toThrowError("Layer5 must not receive abstain Layer4Output");
  });
});

describe("buildFormalVocabulary", () => {
  it("contains exactly the 28 closed-set state labels from §2.1", () => {
    const vocab = buildFormalVocabulary();
    expect(vocab.stateLabels).toHaveLength(ALL_EMOTION_LABELS.length);
    expect(vocab.stateLabels).toHaveLength(28);
    expect(new Set(vocab.stateLabels)).toEqual(new Set(ALL_EMOTION_LABELS));
  });

  it("contains 3 intensity levels and 8 temporal relations from §2.2/§2.3", () => {
    const vocab = buildFormalVocabulary();
    expect(vocab.intensityLevels).toEqual(["mild", "moderate", "severe"]);
    expect(vocab.temporalRelations).toHaveLength(8);
    expect(vocab.logicalOperators).toEqual(["AND", "OR", "NOT", "IMPLIES", "IFF"]);
  });
});

describe("Layer5OutputSchema", () => {
  it("rejects a consistent Layer5Output missing the formula field", () => {
    const r = Layer5OutputSchema.safeParse({ verdict: "consistent" });
    expect(r.success).toBe(false);
  });

  it("rejects an inconsistent Layer5Output missing the formula field", () => {
    const r = Layer5OutputSchema.safeParse({
      verdict: "inconsistent",
      conflictingFacts: [],
    });
    expect(r.success).toBe(false);
  });
});
