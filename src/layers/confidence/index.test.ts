import { describe, expect, it } from "vitest";

import { hallucinationConfig } from "../shared/config.js";
import { Layer4OutputSchema } from "../shared/layerSchemas.zod.js";
import type { Layer3Output } from "../shared/types.js";
import { runConfidenceGate } from "./index.js";

function converged(consensusScore = 0.9, candidate = "answer"): Layer3Output {
  return { verdict: "converged", candidate, consensusScore };
}

function divergent(): Layer3Output {
  return { verdict: "divergent", reason: "sample_divergence", samples: [] };
}

const CEILING = hallucinationConfig.sequenceEntropyMax;
const CONSENSUS_FLOOR = hallucinationConfig.semanticEntropyThreshold;

describe("runConfidenceGate", () => {
  it("returns confident when both gates pass (confidence in (0, 1])", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9, "the answer"),
      getTokenLogProbs: async () => [-0.1, -0.1, -0.1],
    });
    expect(result.verdict).toBe("confident");
    if (result.verdict === "confident") {
      expect(result.answer).toBe("the answer");
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    }
  });

  it("abstains when mean entropy is above sequenceEntropyMax", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9),
      getTokenLogProbs: async () => [-(CEILING + 1), -(CEILING + 1)],
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });

  it("abstains on consensus gate alone (low score, low entropy)", async () => {
    const lowConsensus = CONSENSUS_FLOOR - 0.01;
    const result = await runConfidenceGate({
      layer3Output: converged(lowConsensus),
      getTokenLogProbs: async () => [-0.05],
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });

  it("returns single abstain (not a double) when both gates fail", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(CONSENSUS_FLOOR - 0.1),
      getTokenLogProbs: async () => [-(CEILING + 1), -(CEILING + 1)],
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });

  it("abstains (no exception) when getTokenLogProbs rejects", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9),
      getTokenLogProbs: async () => {
        throw new Error("logprob API down");
      },
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });

  it("abstains when getTokenLogProbs returns []", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9),
      getTokenLogProbs: async () => [],
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });

  it("throws when caller passes a divergent Layer3Output", async () => {
    await expect(
      runConfidenceGate({
        layer3Output: divergent(),
        getTokenLogProbs: async () => [-0.1],
      }),
    ).rejects.toThrowError("Layer4 must not receive divergent Layer3Output");
  });

  it("clamps confidence to exactly 1.0 when entropy is 0", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9),
      getTokenLogProbs: async () => [0, 0, 0],
    });
    expect(result.verdict).toBe("confident");
    if (result.verdict === "confident") {
      expect(result.confidence).toBe(1);
    }
  });

  it("abstains rather than returning confident with 0 when entropy = ceiling", async () => {
    const result = await runConfidenceGate({
      layer3Output: converged(0.9),
      getTokenLogProbs: async () => [-CEILING, -CEILING, -CEILING],
    });
    expect(result).toEqual({ verdict: "abstain_low_confidence", template: "T4" });
  });
});

describe("Layer4OutputSchema", () => {
  it("rejects a confident Layer4Output missing the confidence field", () => {
    const r = Layer4OutputSchema.safeParse({
      verdict: "confident",
      answer: "x",
    });
    expect(r.success).toBe(false);
  });

  it("rejects an abstain Layer4Output with the wrong template literal", () => {
    const r = Layer4OutputSchema.safeParse({
      verdict: "abstain_low_confidence",
      template: "T7",
    });
    expect(r.success).toBe(false);
  });
});
