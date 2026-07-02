
========================================================================
  DAISY REGRESSION REPORT COMPARISON
  Generated: 2026-07-01T21:17:58.514908+00:00
========================================================================

------------------------------------------------------------------------
  Metric                               Before        After        Delta
------------------------------------------------------------------------
  Overall Pass Rate                     28.6%         0.0%       -28.6%
  Total Passed                             16            0          -16
  Total Failed                             40           56          +16
  Total Cases                              56           56           +0
------------------------------------------------------------------------

  PER-CLUSTER BREAKDOWN
------------------------------------------------------------------------
  Cluster             Before      After      Delta    Direction
------------------------------------------------------------------------
  anxiety              12.5%       0.0%     -12.5%  regressed тЖУ
  breakup              75.0%       0.0%     -75.0%  regressed тЖУ
  clarity              50.0%       0.0%     -50.0%  regressed тЖУ
  grief                25.0%       0.0%     -25.0%  regressed тЖУ
  somatic              12.5%       0.0%     -12.5%  regressed тЖУ
  stress               12.5%       0.0%     -12.5%  regressed тЖУ
  work                 12.5%       0.0%     -12.5%  regressed тЖУ
------------------------------------------------------------------------

  PER-LOCALE BREAKDOWN
------------------------------------------------------------------------
  Locale              Before      After      Delta    Direction
------------------------------------------------------------------------
  en                   39.3%       0.0%     -39.3%  regressed тЖУ
  ru                   17.9%       0.0%     -17.9%  regressed тЖУ
------------------------------------------------------------------------

  FAILURE ANALYSIS
------------------------------------------------------------------------
  Resolved (count reduced):
    тЬЕ hollow: -1
  New (count increased):
    тЭМ script_leak: +28
    тЭМ locale_incorrect: +28
    тЭМ keyword_mismatch: +16
  Remaining (still present):
      keyword_mismatch: 56
      script_leak: 28
      locale_incorrect: 28
------------------------------------------------------------------------

  REGRESSED CASES (16): passed before, failed after
    тЭМ anxiety_en_1 (anxiety/en) тАФ keyword_mismatch
    тЭМ breakup_en_1 (breakup/en) тАФ keyword_mismatch
    тЭМ breakup_en_3 (breakup/en) тАФ keyword_mismatch
    тЭМ breakup_en_4 (breakup/en) тАФ keyword_mismatch
    тЭМ breakup_ru_1 (breakup/ru) тАФ script_leak, keyword_mismatch, locale_incorrect
    тЭМ breakup_ru_2 (breakup/ru) тАФ script_leak, keyword_mismatch, locale_incorrect
    тЭМ breakup_ru_4 (breakup/ru) тАФ script_leak, keyword_mismatch, locale_incorrect
    тЭМ clarity_en_1 (clarity/en) тАФ keyword_mismatch
    тЭМ clarity_en_3 (clarity/en) тАФ keyword_mismatch
    тЭМ clarity_en_4 (clarity/en) тАФ keyword_mismatch
    тЭМ clarity_ru_3 (clarity/ru) тАФ script_leak, keyword_mismatch, locale_incorrect
    тЭМ grief_en_2 (grief/en) тАФ keyword_mismatch
    тЭМ grief_en_4 (grief/en) тАФ keyword_mismatch
    тЭМ somatic_en_1 (somatic/en) тАФ keyword_mismatch
    тЭМ stress_en_2 (stress/en) тАФ keyword_mismatch
    тЭМ work_ru_2 (work/ru) тАФ script_leak, keyword_mismatch, locale_incorrect

========================================================================
  RELEASE GATES
========================================================================
    тЭМ Overall pass rate >= 90%
    тЭМ Per-cluster pass rate >= 85%
    тЬЕ Zero structural_leak failures
    тЭМ Zero script_leak failures

  тЭМ DO NOT RELEASE
========================================================================

