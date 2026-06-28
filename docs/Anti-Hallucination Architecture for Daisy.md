# Anti-Hallucination Architecture for Daisy
## Adapting VMI-Inspired Principles to Emotional AI

***

## Executive Summary

Daisy operates in a high-stakes emotional/psychological domain where hallucinations are not merely incorrect — they can cause direct user harm. This report outlines a layered anti-hallucination architecture inspired by the Desargues AI VMI approach (formal verification via Lean 4) but adapted to Daisy's specific constraints: natural language emotional inference, user-uploaded data grounding, and conversational continuity. The architecture consists of five stacked layers, ranging from retrieval grounding to symbolic constraint validation and confidence-aware abstention.

***

## Why Standard LLM Outputs Are Unsafe for Daisy

Large language models hallucinate by design — they are trained to produce fluent, statistically plausible completions, not verified truths. OpenAI's own research confirmed that hallucinations are mathematically inevitable at the model level, driven by fundamental constraints including cross-entropy loss minimization and extrapolation beyond the training data convex hull. For Daisy, which uses Qwen as the base model and draws conclusions about a user's emotional state and evolution over time, this creates a specific failure mode: the model may confidently assert emotional patterns that contradict or are entirely absent from the user's actual data.[^1]

Neuro-symbolic AI is now recognized as the most appropriate architecture for mental health and emotional AI contexts specifically because it reduces hallucinations while enforcing verifiable constraints. A Forbes analysis from April 2026 identified neuro-symbolic systems as demonstrably safer than standalone LLMs for mental health guidance, precisely because the symbolic layer can enforce escalation policies, clinical constraints, and traceability. Neurosymbolic deployments have shown up to 40% reduction in hallucination rates compared to pure LLM baselines.[^2][^3][^4]

***

## The Core Architecture: Five Layers

### Layer 1 — Grounded Retrieval (RAG with Evidence Sufficiency Gating)

Every claim Daisy makes about a user's emotional state must be traceable to a specific data point from their uploaded history. A naive RAG pipeline (embed → retrieve → generate) is insufficient for production and collapses under imperfect evidence. Daisy requires an **evidence-sufficiency model** that classifies each query into three states before generation:[^5][^6]

| State | Condition | Action |
|---|---|---|
| Fully answerable | Sufficient high-similarity evidence exists | Generate grounded response |
| Partially answerable | Some evidence, bounded reasoning possible | Generate with explicit uncertainty marker |
| Not safely answerable | Evidence below threshold | Abstain or ask clarifying question |

A 2026 production RAG architecture paper formalizes this pattern: each pipeline stage emits a structured schema-validated object (`QuerySpec`, `EvidenceSet`, `AnswerDraft`) — no stage reads raw unvalidated text from a prior stage. This prevents hallucinated "chains" where a fabricated claim in one step becomes the input for the next. The system must **expand partial-answerability** while strictly **protecting the not-safely-answerable state** — naive systems collapse state (3) into (1), hallucinating confidently.[^5]

Real-time verification of full-context retrieval (up to 32K tokens) during generation is now feasible: a 2026 paper demonstrated production-grade RAG verification under latency constraints that substantially improves detection of unsupported responses.[^7]

### Layer 2 — Symbolic Constraint Validation (Domain Rules Engine)

This is the direct adaptation of VMI's principle. VMI uses Lean 4 to formally verify that mathematical conclusions follow from stated premises. For Daisy's emotional domain, full Lean 4 formalization is impractical — emotional states are not formalizable as mathematical propositions. However, a **domain-specific rules engine** implements the same verification principle at a lower formalization level.[^8][^2]

Concretely, this means encoding constraints such as:
- **Temporal monotonicity**: Emotional state transitions must be plausible given the time delta between sessions (e.g., severe distress → flourishing cannot occur in one interaction without intermediate states)
- **Internal consistency**: Contradictory emotional labels within a single session window (e.g., "hopelessness" + "enthusiasm") must be flagged before surface
- **Escalation protocols**: Specific emotional markers (suicidal ideation signals, crisis language patterns) must route to a hardcoded escalation path, bypassing the generative layer entirely
- **Temporal grounding**: Any reference to "recently" or "lately" must be anchored to a specific timestamped data point from the user's upload

If an LLM output violates any encoded rule, it is rejected and regenerated (up to N retries) or replaced with an abstention response. This is precisely the pattern Forbes identifies as the key advantage of neuro-symbolic systems over standalone LLMs for mental health: "encoding clinical constraints and escalation policies as explicit logic".[^3][^2]

### Layer 3 — Self-Consistency Sampling with Semantic Voting

For claims that pass grounding and constraint checks, self-consistency sampling provides an additional confidence signal. The mechanism: generate K responses (temperature > 0), cluster by semantic similarity, and select the majority cluster.[^9][^10]

The key insight from recent research is that **disagreement across samples is a reliable hallucination signal** — high semantic divergence across K samples indicates the model is operating in an uncertain region, even if any individual output sounds confident. SelfCheckGPT implements this without requiring access to token log-probabilities, making it model-agnostic and compatible with Qwen via API.[^10][^9]

A more powerful variant, **Consortium Consistency**, extends this across multiple LLMs. By running the same prompt through two or more models and computing `consortium entropy` (divergence across model outputs), it achieves better hallucination detection than single-model consistency while reducing per-query costs through shared sample budgets. For Daisy, a two-model setup (Qwen as primary + a lightweight verifier model) is a practical implementation.[^11]

The formal hallucination risk bound decomposes into data-driven and reasoning-driven components:[^10]

\[
\text{HR} = \alpha \cdot \text{DataMismatch} + \beta \cdot \text{ReasoningInstability}
\]

Self-consistency sampling directly targets \( \beta \) (reasoning instability), while RAG grounding targets \( \alpha \) (data mismatch).[^10]

### Layer 4 — Confidence-Aware Abstention

Calibrated abstention is the cleanest production-grade mitigation: refuse to answer when the expected value of answering (given confidence) is negative. This requires a confidence signal that correlates with actual correctness — not verbalized confidence (which is systematically overconfident post-RLHF).[^12][^9]

Two practical implementations for Daisy:

**Entropy-based abstention**: Token-level output entropy exceeding a threshold triggers abstention or uncertainty marking. Research shows entropy-based early stopping achieves 25–50% compute savings while maintaining accuracy, with the threshold tunable per risk tolerance. For emotional AI, the threshold should be set conservatively — high abstention rate is preferable to confident fabrication.[^13][^9]

**Activation-based confidence probing**: A lightweight LSTM probe over intermediate transformer activations computes a scalar confidence from mid-layer representations, achieving over 40% latency reduction compared to full generation + post-hoc check. This is the architecture used in a 2026 paper on high-stakes healthcare and finance domains — the probe combines cross-entropy loss with Huber calibration regularization to balance precision and coverage.[^12]

A deployed production example from Gusto: routing token log-probability low-confidence outputs to human review (rather than auto-answering) raised product-level accuracy from 45% to 76% for high-confidence outputs.[^9]

### Layer 5 — Autoformalization for Logical Consistency Checks

This is the direct technical bridge to VMI's architecture. Autoformalization — translating natural language LLM outputs into formal logic representations — allows consistency checking without full Lean 4 proof.[^14][^15]

A 2025 arXiv paper demonstrated two practical experiments:[^14]
1. The autoformalizer correctly identified that two differently-worded requirements were logically equivalent
2. The autoformalizer identified a logical inconsistency between a stated requirement and an LLM-generated output

For Daisy, this translates to: formalize Daisy's output claims into propositional logic, then run consistency checks against the user's known data facts and prior session conclusions. VeriTrans (April 2026) implements exactly this — NL → propositional logic → CNF → SAT solver validation, with validator overhead under 15% of end-to-end runtime. The round-trip consistency validator (PL → NL → check) provides a conservative acceptance gate before any output is surfaced.[^16]

The autoformalization bottleneck (translating emotional NL claims into formal logic is itself an LLM task and can introduce errors) is mitigated by scoping the vocabulary: Daisy's domain is bounded, so the formal vocabulary can be pre-defined (emotional state labels, temporal relations, intensity levels) rather than open-ended.[^15]

***

## Architecture Diagram

```
User Input + Uploaded Data
          │
          ▼
┌─────────────────────────────────┐
│  Layer 1: RAG + Evidence Gate   │  ← cosine similarity threshold,
│  (Grounded Retrieval)           │    state classification (full/partial/abstain)
└────────────────┬────────────────┘
                 │ EvidenceSet (schema-validated)
                 ▼
┌─────────────────────────────────┐
│  Layer 2: Rules Engine          │  ← temporal constraints, label consistency,
│  (Symbolic Constraint Check)    │    escalation routing, data anchoring
└────────────────┬────────────────┘
                 │ Validated AnswerDraft
                 ▼
┌─────────────────────────────────┐
│  Layer 3: Self-Consistency      │  ← K samples, semantic clustering,
│  (Voting / Consortium Entropy)  │    majority vote or abstain on divergence
└────────────────┬────────────────┘
                 │ Candidate Answer
                 ▼
┌─────────────────────────────────┐
│  Layer 4: Confidence Abstention │  ← entropy threshold, LSTM confidence
│  (Calibrated Gating)            │    probe, route to human review if c < τ
└────────────────┬────────────────┘
                 │ Confidence-Gated Answer
                 ▼
┌─────────────────────────────────┐
│  Layer 5: Autoformalization     │  ← NL → PL → SAT consistency check,
│  (Logical Consistency Check)    │    round-trip validation
└────────────────┬────────────────┘
                 │
                 ▼
        Surface to User
        (or Abstain + Ask Clarifying Q)
```

***

## Key Differences from VMI (Lean 4)

| Dimension | VMI / Desargues AI | Daisy Architecture |
|---|---|---|
| Verification target | Mathematical theorems | Emotional state claims |
| Formalization layer | Lean 4 proof assistant | Propositional logic + SAT solver [^16] |
| Domain | Formally formalizable (math, logic) | Semi-formalizable (bounded emotional vocabulary) |
| Verification verdict | Binary (proven / fails type-check) [^8] | Probabilistic + rule-based |
| Failure mode | Proof doesn't compile | Constraint violation + abstention |
| Human loop | Not required for math | Required for escalation cases |

The principle is identical: **never surface output that hasn't passed an independent deterministic check**. The implementation differs because emotional inference is not fully formalizable, requiring a probabilistic + rule hybrid rather than pure formal proof.[^2][^8]

***

## Implementation Priority for Daisy

Given Daisy's current stage, the recommended build order prioritizes impact-to-effort ratio:

1. **Layer 2 first (Rules Engine)** — Highest safety impact, lowest ML complexity. Define the constraint set, encode escalation routing. Can be shipped as a pure code layer on top of the existing Qwen pipeline.
2. **Layer 1 (RAG grounding)** — Ground all emotional claims to user's uploaded data. Implement cosine similarity threshold; abstain if no evidence exceeds threshold.
3. **Layer 4 (Entropy abstention)** — Add token-level entropy monitoring to Qwen outputs. Set a conservative threshold, route low-confidence to "Daisy isn't sure — does this resonate?" framing.
4. **Layer 3 (Self-consistency)** — Add K-sample voting for high-stakes outputs (emotional diagnoses, trend conclusions). Run 3–5 samples; high divergence → abstain or mark as uncertain.
5. **Layer 5 (Autoformalization)** — Most complex, greatest overlap with VMI. Scope the formal vocabulary to Daisy's emotional label set. Implement as a post-hoc consistency checker, not a pre-generation blocker.

***

## Critical Constraint: Human-in-the-Loop

All five layers reduce but cannot eliminate hallucination. For Daisy's domain, the irreducible residual must be handled by **surfacing outputs as hypotheses, not conclusions**. Framing matters:[^1]

- Instead of: *"You've been experiencing chronic anxiety for 3 weeks."*
- Use: *"Daisy noticed a pattern of high-tension entries over the past 3 weeks — does this match how you've been feeling?"*

This is not a UX preference — it is a safety requirement. The user becomes the final verification layer, which is appropriate given that emotional inference is genuinely ambiguous even for trained clinicians.[^3][^2]

---

## References

1. [OpenAI admits AI hallucinations are mathematically inevitable, not ...](https://www.computerworld.com/article/4059383/openai-admits-ai-hallucinations-are-mathematically-inevitable-not-just-engineering-flaws.html) - Vempala, provided a comprehensive mathematical framework explaining why AI systems must generate pla...

2. [Neuro-Symbolic AI Enhances Mental Health Advice Quality](https://letsdatascience.com/news/neuro-symbolic-ai-enhances-mental-health-advice-quality-1676f44f) - By pairing LLM capabilities for natural language understanding with a rules-based expert system, neu...

3. [Using Neuro-Symbolic AI For Mental Health Advice Is Better Than ...](https://www.forbes.com/sites/lanceeliot/2026/04/22/using-neuro-symbolic-ai-for-mental-health-advice-is-better-than-conventional-ai-for-these-crucial-reasons/) - Neuro-symbolic AI is now being used to provide mental health guidance. Turns out this is better than...

4. [Why Neurosymbolic AI Is the Next Big Leap Beyond LLMs - LinkedIn](https://www.linkedin.com/pulse/why-neurosymbolic-ai-next-big-leap-beyond-llms-gary-ramah-uyqic) - Studies and deployments show neurosymbolic systems can reduce hallucination rates by up to forty per...

5. [Designing High-Precision LLM RAG Systems: An Enterprise-Grade ...](https://dev.to/optyxstack/designing-high-precision-llm-rag-systems-an-enterprise-grade-architecture-blueprint-1ldo) - A production RAG platform should follow this layered pipeline: Query Understanding; Retrieval Planni...

6. [RAG Architecture Explained: How Retrieval-Augmented Generation ...](https://bigdataboutique.com/blog/rag-architecture-explained-how-retrieval-augmented-generation-works) - RAG is a system design in which an LLM's generation step is preceded by a retrieval step that fetche...

7. [Real-Time Verification for Long-Document Retrieval-Augmented ...](https://arxiv.org/html/2603.23508v1) - Retrieval-augmented generation (RAG) has become a standard approach for grounding large language mod...

8. [Lean4: How the theorem prover works and why it's the new ...](https://venturebeat.com/ai/lean4-how-the-theorem-prover-works-and-why-its-the-new-competitive-edge-in) - Lean4 is both a programming language and a proof assistant designed for formal verification. Every t...

9. [LLM Confidence Calibration in Production: Measuring and Fixing ...](https://tianpan.co/blog/2026-04-16-llm-confidence-calibration-production) - LLMs that say 'I'm highly confident' are often wrong at that exact rate. How to measure calibration ...

10. [(PDF) Lightweight Self-Consistency Voting Framework for Reliable ...](https://www.academia.edu/165520853/Lightweight_Self_Consistency_Voting_Framework_for_Reliable_Hallucination_Detection_in_LLMs) - The primary goals of this work are to: • Develop a high-performance self-consistency voting mechanis...

11. [4 Results](https://arxiv.org/html/2510.19507v1)

12. [Confidence-Aware Abstention in ML - Emergent Mind](https://www.emergentmind.com/topics/confidence-aware-abstention) - Discover a machine learning framework that empowers models to abstain from low-confidence prediction...

13. [Sequence-Level Entropy as a Confidence Signal for LLM Reasoning](https://arxiv.org/html/2510.08146v1)

14. [[2511.11829] Towards Autoformalization of LLM-generated Outputs ...](https://arxiv.org/abs/2511.11829) - Autoformalization, the process of translating informal statements into formal logic, has gained rene...

15. [Autoformalization with Large Language Models - Emergent Mind](https://www.emergentmind.com/topics/autoformalization-with-large-language-models) - Autoformalization with Large Language Models translates informal expressions into formal proofs and ...

16. [VeriTrans: Fine-Tuned LLM-Assisted NL-to-PL Translation ... - arXiv](https://arxiv.org/html/2604.10341v1) - Instead, we study how LLM-generated logical formulas can be safely integrated into verification pipe...

