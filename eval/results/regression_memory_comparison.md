
========================================================================
  DAISY MEMORY REGRESSION REPORT COMPARISON
  Generated: 2026-07-02T08:06:32Z
  Build: INFERENCE_BUILD=2026-07-qwen3-v16-memory
========================================================================

------------------------------------------------------------------------
  SINGLE-TURN (56 cases, cross_topic_regression.jsonl)
------------------------------------------------------------------------
  Metric                               v15 base   v16-memory      Delta
------------------------------------------------------------------------
  Overall Pass Rate                     57.1%        50.0%        -7.1%
  Total Passed                             32           28           -4
  Total Failed                             24           28           +4
  Gate (≥60%)                              —           FAIL          —
------------------------------------------------------------------------

  PER-CLUSTER (single-turn)
------------------------------------------------------------------------
  Cluster             v15 base   v16-memory      Delta    Direction
------------------------------------------------------------------------
  anxiety              50.0%       75.0%      +25.0%   improved
  breakup              75.0%       62.5%      -12.5%  regressed
  clarity              37.5%       25.0%      -12.5%  regressed
  grief                62.5%       12.5%      -50.0%  regressed
  somatic              87.5%       62.5%      -25.0%  regressed
  stress               50.0%       62.5%      +12.5%   improved
  work                 37.5%       50.0%      +12.5%   improved
------------------------------------------------------------------------

  PER-LOCALE (single-turn)
------------------------------------------------------------------------
  Locale              v15 base   v16-memory      Delta    Direction
------------------------------------------------------------------------
  en                   50.0%       32.1%      -17.9%  regressed
  ru                   64.3%       67.9%       +3.6%   improved
------------------------------------------------------------------------

  FAILURE ANALYSIS (single-turn)
------------------------------------------------------------------------
  All failures: keyword_mismatch (28)
  structural_leak: 0 | script_leak: 0 | canned_greeting: 0
------------------------------------------------------------------------

------------------------------------------------------------------------
  MULTI-TURN (12 cases, multi_turn_regression.jsonl) — NEW GATE
------------------------------------------------------------------------
  Overall Pass Rate                      n/a         0.0%
  Gate (≥75%, 9/12)                      —           FAIL
  Primary failure mode: prior_topic_mismatch (11/12 cases)
  keyword_mismatch also present: 7/12 cases
------------------------------------------------------------------------

  INTERPRETATION
------------------------------------------------------------------------
  Pipeline fixes deployed (CBT history, history summary in prompt,
  register lock, anti-generic rules, user_context injection) but model
  still fails to echo prior-topic stems in multi-turn replies.
  Single-turn regressed vs v15 baseline — EN cluster especially weak.
  v16 LoRA training remains blocked per gated plan.
------------------------------------------------------------------------
