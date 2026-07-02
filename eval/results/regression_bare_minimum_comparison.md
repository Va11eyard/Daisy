
========================================================================
  DAISY BARE-MINIMUM REGRESSION COMPARISON
  Generated: 2026-07-02T09:33:44Z
  Build: INFERENCE_BUILD=2026-07-qwen3-v17-bare-minimum
========================================================================

------------------------------------------------------------------------
  SINGLE-TURN (56 cases)
------------------------------------------------------------------------
  Metric                          v16-memory   v17-bare      Delta
------------------------------------------------------------------------
  Overall Pass Rate                  50.0%       46.4%        -3.6%
  Total Passed                          28          26           -2
  Gate (≥60%)                          FAIL        FAIL          —
  structural_leak                         0           0           0
  script_leak                             0           0           0
------------------------------------------------------------------------

  PER-LOCALE (single-turn)
------------------------------------------------------------------------
  Locale                          v16-memory   v17-bare      Delta
------------------------------------------------------------------------
  en                                 32.1%       14.3%       -17.8%
  ru                                 67.9%       78.6%       +10.7%
------------------------------------------------------------------------

------------------------------------------------------------------------
  MULTI-TURN (12 cases)
------------------------------------------------------------------------
  Overall Pass Rate                   0.0%        0.0%         +0.0%
  Gate (≥50%, 6/12)                   FAIL        FAIL          —
  Primary failure: prior_topic_mismatch
------------------------------------------------------------------------

  INTERPRETATION
------------------------------------------------------------------------
  Bare-minimum strip eliminated pipeline corruption (0 leaks) and
  improved RU single-turn (+10.7pp). Keyword-match regression gate
  still fails — model does not echo eval stems reliably, especially EN
  and multi-turn prior-topic linkage. v16 LoRA training remains blocked.
------------------------------------------------------------------------
