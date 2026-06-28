import { describe, expect, it } from "vitest";

import { hallucinationConfig } from "../shared/config.js";
import { Layer3OutputSchema } from "../shared/layerSchemas.zod.js";
import type { EvidenceSet, Layer3Input, Similarity, Timestamp } from "../shared/types.js";
import { runSelfConsistency } from "./index.js";

const NOW = "2026-04-27T12:00:00.000Z" as Timestamp;

const FULL_EVIDENCE: EvidenceSet = {
  state: "fully_answerable",
  evidence: [
    {
      entryId: "e1" as never,
      sessionId: "s1" as never,
      timestamp: NOW,
      textExcerpt: "x",
      similarity: 0.9 as Similarity,
    },
  ],
  aggregateSimilarity: 0.9 as Similarity,
  retrievedAt: NOW,
};

const NOT_ANSWERABLE: EvidenceSet = {
  state: "not_answerable",
  reason: "below_threshold",
  retrievedAt: NOW,
};

const K = hallucinationConfig.K;

function baseInput(overrides: Partial<Layer3Input> = {}): Layer3Input {
  return {
    query: "How am I feeling lately?",
    evidenceSet: FULL_EVIDENCE,
    generateFn: async () => "answer",
    embedFn: async () => [1, 0, 0, 0, 0],
    ...overrides,
  };
}

describe("runSelfConsistency", () => {
  it("converges when K samples are semantically similar (centroid is index 0 on tie)", async () => {
    let i = 0;
    const result = await runSelfConsistency(
      baseInput({
        generateFn: async () => `answer-${i++}`,
        embedFn: async () => [1, 0, 0, 0, 0],
      }),
    );
    expect(result.verdict).toBe("converged");
    if (result.verdict === "converged") {
      expect(result.consensusScore).toBeCloseTo(1, 6);
      expect(result.candidate).toBe("answer-0");
    }
  });

  it("returns divergent when K samples are semantically far apart", async () => {
    let i = 0;
    const orthonormal = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
    ];
    const result = await runSelfConsistency(
      baseInput({
        generateFn: async () => `answer-${i++}`,
        embedFn: async (text: string) => {
          const id = Number(text.slice("answer-".length));
          return orthonormal[id % orthonormal.length]!;
        },
      }),
    );
    expect(result.verdict).toBe("divergent");
    if (result.verdict === "divergent") {
      expect(result.reason).toBe("sample_divergence");
      expect(result.samples.length).toBe(K);
    }
  });

  it("converges when one generateFn rejects but ≥ ceil(K/2) succeed", async () => {
    let calls = 0;
    const result = await runSelfConsistency(
      baseInput({
        generateFn: async () => {
          const id = calls++;
          if (id === 0) throw new Error("first sample fails");
          return `answer-${id}`;
        },
        embedFn: async () => [1, 0, 0, 0, 0],
      }),
    );
    expect(result.verdict).toBe("converged");
    if (result.verdict === "converged") {
      expect(result.consensusScore).toBeCloseTo(1, 6);
      expect(result.candidate).toBe("answer-1");
    }
  });

  it("returns divergent when fewer than ceil(K/2) succeed", async () => {
    const minSuccess = Math.ceil(K / 2);
    let calls = 0;
    const result = await runSelfConsistency(
      baseInput({
        generateFn: async () => {
          const id = calls++;
          if (id < minSuccess) throw new Error("forced failure");
          return `answer-${id}`;
        },
        embedFn: async () => [1, 0, 0, 0, 0],
      }),
    );
    expect(result.verdict).toBe("divergent");
    if (result.verdict === "divergent") {
      expect(result.reason).toBe("sample_divergence");
      expect(result.samples.length).toBeLessThan(minSuccess);
    }
  });

  it("throws synchronously-rejecting Error when EvidenceSet.state is not_answerable", async () => {
    await expect(
      runSelfConsistency(baseInput({ evidenceSet: NOT_ANSWERABLE })),
    ).rejects.toThrowError("Layer3 must not receive not_answerable EvidenceSet");
  });

  it("invokes generateFn exactly K times in parallel (Promise.all semantics)", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    let totalCalls = 0;

    const result = await runSelfConsistency(
      baseInput({
        generateFn: async () => {
          totalCalls++;
          inFlight++;
          if (inFlight > maxInFlight) maxInFlight = inFlight;
          await new Promise<void>((resolve) => setTimeout(resolve, 5));
          inFlight--;
          return "answer";
        },
        embedFn: async () => [1, 0, 0, 0, 0],
      }),
    );

    expect(totalCalls).toBe(K);
    expect(maxInFlight).toBe(K);
    expect(result.verdict).toBe("converged");
  });
});

describe("Layer3OutputSchema", () => {
  it("rejects a converged Layer3Output missing consensusScore", () => {
    const r = Layer3OutputSchema.safeParse({
      verdict: "converged",
      candidate: "x",
    });
    expect(r.success).toBe(false);
  });

  it("rejects a divergent Layer3Output with the wrong reason literal", () => {
    const r = Layer3OutputSchema.safeParse({
      verdict: "divergent",
      reason: "something_else",
      samples: [],
    });
    expect(r.success).toBe(false);
  });
});
