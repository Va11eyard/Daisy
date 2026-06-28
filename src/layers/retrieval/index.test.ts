import { describe, expect, it } from "vitest";

import { EvidenceCitationSchema, EvidenceSetSchema } from "../shared/layerSchemas.zod.js";
import { hallucinationConfig } from "../shared/config.js";
import type { JournalEntry, Layer1Input } from "../shared/types.js";
import { retrieveGroundedEvidence } from "./index.js";

const NOW = "2026-04-27T12:00:00.000Z";

function entry(
  id: string,
  session: string,
  timestamp: string,
  text: string,
): JournalEntry {
  return {
    entryId: id as JournalEntry["entryId"],
    sessionId: session as JournalEntry["sessionId"],
    timestamp: timestamp as JournalEntry["timestamp"],
    text,
  };
}

/** Unit vectors on axes so cosine equals the non-zero component. */
function embedFactory(vectors: Map<string, number[]>): Layer1Input["embeddingFn"] {
  return async (text: string) => {
    const v = vectors.get(text);
    if (!v) throw new Error(`unknown text for mock embed: ${text}`);
    return v;
  };
}

describe("retrieveGroundedEvidence", () => {
  const baseInput = (overrides: Partial<Layer1Input> = {}): Layer1Input => ({
    query: "q",
    uploadedEntries: [],
    consentedRetentionWindowDays: 365,
    now: NOW as Layer1Input["now"],
    embeddingFn: async () => [1, 0, 0],
    ...overrides,
  });

  it("returns fully_answerable when top score ≥ threshold_full", async () => {
    const q = "my query";
    const t = "my entry text";
    const vec = [1, 0, 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        uploadedEntries: [entry("e1", "s1", "2026-04-26T10:00:00.000Z", t)],
        embeddingFn: embedFactory(
          new Map([
            [q, vec],
            [t, vec],
          ]),
        ),
      }),
    );
    expect(result.state).toBe("fully_answerable");
    if (result.state === "fully_answerable") {
      expect(result.evidence.length).toBeGreaterThanOrEqual(1);
      expect(result.evidence[0]!.similarity).toBeGreaterThanOrEqual(hallucinationConfig.threshold_full);
      expect(result.aggregateSimilarity).toBeGreaterThanOrEqual(hallucinationConfig.threshold_full);
    }
  });

  it("returns partially_answerable when best score is between thresholds", async () => {
    const q = "my query";
    const t = "entry body";
    const qv = [1, 0, 0];
    const ev = [0.65, Math.sqrt(1 - 0.65 * 0.65), 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        uploadedEntries: [entry("e1", "s1", "2026-04-26T10:00:00.000Z", t)],
        embeddingFn: embedFactory(
          new Map([
            [q, qv],
            [t, ev],
          ]),
        ),
      }),
    );
    expect(result.state).toBe("partially_answerable");
    if (result.state === "partially_answerable") {
      expect(result.evidence.length).toBeGreaterThanOrEqual(1);
      const sims = result.evidence.map((c) => c.similarity);
      expect(Math.max(...sims)).toBeGreaterThanOrEqual(hallucinationConfig.threshold_partial);
      expect(Math.max(...sims)).toBeLessThan(hallucinationConfig.threshold_full);
      expect(result.gaps.length).toBeGreaterThan(0);
    }
  });

  it("returns not_answerable when all scores < threshold_partial", async () => {
    const q = "my query";
    const t = "entry body";
    const qv = [1, 0, 0];
    const ev = [0.5, Math.sqrt(0.75), 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        uploadedEntries: [entry("e1", "s1", "2026-04-26T10:00:00.000Z", t)],
        embeddingFn: embedFactory(
          new Map([
            [q, qv],
            [t, ev],
          ]),
        ),
      }),
    );
    expect(result.state).toBe("not_answerable");
    if (result.state === "not_answerable") {
      expect(result.reason).toBe("below_threshold");
    }
  });

  it("returns not_answerable with embedding_failure when embeddingFn rejects", async () => {
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: "q",
        uploadedEntries: [entry("e1", "s1", "2026-04-26T10:00:00.000Z", "t")],
        embeddingFn: async () => {
          throw new Error("upstream embed failed");
        },
      }),
    );
    expect(result.state).toBe("not_answerable");
    if (result.state === "not_answerable") {
      expect(result.reason).toBe("embedding_failure");
    }
  });

  it("returns not_answerable with data_excluded_by_consent when uploads exist but every entry is outside consentedRetentionWindowDays", async () => {
    const result = await retrieveGroundedEvidence(
      baseInput({
        consentedRetentionWindowDays: 7,
        uploadedEntries: [
          entry("e-old", "s1", "2026-04-10T10:00:00.000Z", "too old for consent"),
        ],
        embeddingFn: async () => {
          throw new Error("embed must not run");
        },
      }),
    );
    expect(result.state).toBe("not_answerable");
    if (result.state === "not_answerable") {
      expect(result.reason).toBe("data_excluded_by_consent");
    }
  });

  it("returns no_data_in_window when entries pass consent but none fall within recencyWindowDays", async () => {
    const result = await retrieveGroundedEvidence(
      baseInput({
        consentedRetentionWindowDays: 365,
        uploadedEntries: [
          entry("e-old", "s1", "2026-03-01T10:00:00.000Z", "outside recency only"),
        ],
        embeddingFn: async () => {
          throw new Error("embed must not run");
        },
      }),
    );
    expect(result.state).toBe("not_answerable");
    if (result.state === "not_answerable") {
      expect(result.reason).toBe("no_data_in_window");
    }
  });

  it("excludes entries outside recencyWindowDays before scoring", async () => {
    const q = "q";
    const oldText = "old";
    const newText = "new";
    const vec = [1, 0, 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        consentedRetentionWindowDays: 365,
        uploadedEntries: [
          entry("old", "s1", "2026-03-01T10:00:00.000Z", oldText),
          entry("new", "s1", "2026-04-26T10:00:00.000Z", newText),
        ],
        embeddingFn: embedFactory(
          new Map([
            [q, vec],
            [oldText, [0, 1, 0]],
            [newText, vec],
          ]),
        ),
      }),
    );
    expect(result.state).toBe("fully_answerable");
    if (result.state === "fully_answerable") {
      expect(result.evidence.every((c) => c.entryId === ("new" as string))).toBe(true);
    }
  });

  it("uses the smaller of recencyWindowDays and consentedRetentionWindowDays", async () => {
    const q = "q";
    const t = "inside-seven-days";
    const vec = [1, 0, 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        consentedRetentionWindowDays: 7,
        uploadedEntries: [
          entry("e10", "s1", "2026-04-16T10:00:00.000Z", "ten days ago"),
          entry("e5", "s1", "2026-04-22T10:00:00.000Z", t),
        ],
        embeddingFn: embedFactory(
          new Map([
            [q, vec],
            [t, vec],
            ["ten days ago", [0, 1, 0]],
          ]),
        ),
      }),
    );
    expect(result.state).toBe("fully_answerable");
    if (result.state === "fully_answerable") {
      expect(result.evidence.some((c) => c.entryId === ("e5" as string))).toBe(true);
      expect(result.evidence.some((c) => c.entryId === ("e10" as string))).toBe(false);
    }
  });

  it("every returned EvidenceCitation includes D2 fields (entryId, sessionId, timestamp, textExcerpt)", async () => {
    const q = "q";
    const t = "body";
    const vec = [1, 0, 0];
    const result = await retrieveGroundedEvidence(
      baseInput({
        query: q,
        uploadedEntries: [entry("e1", "sess-a", "2026-04-26T10:00:00.000Z", t)],
        embeddingFn: embedFactory(
          new Map([
            [q, vec],
            [t, vec],
          ]),
        ),
      }),
    );
    expect(result.state).not.toBe("not_answerable");
    if (result.state === "fully_answerable" || result.state === "partially_answerable") {
      for (const c of result.evidence) {
        expect(c.entryId).toBeTruthy();
        expect(c.sessionId).toBeTruthy();
        expect(c.timestamp).toBeTruthy();
        expect(c.textExcerpt).toBeTruthy();
        expect(typeof c.similarity).toBe("number");
      }
    }
  });
});

describe("EvidenceSetSchema / EvidenceCitationSchema", () => {
  it("rejects a citation missing timestamp", () => {
    const bad = {
      entryId: "e1",
      sessionId: "s1",
      textExcerpt: "x",
      similarity: 0.9,
    };
    const r = EvidenceCitationSchema.safeParse(bad);
    expect(r.success).toBe(false);
  });

  it("rejects fully_answerable EvidenceSet when a citation omits timestamp", () => {
    const bad = {
      state: "fully_answerable" as const,
      aggregateSimilarity: 0.9,
      retrievedAt: NOW,
      evidence: [
        {
          entryId: "e1",
          sessionId: "s1",
          textExcerpt: "hello",
          similarity: 0.9,
        },
      ],
    };
    const r = EvidenceSetSchema.safeParse(bad);
    expect(r.success).toBe(false);
  });
});
