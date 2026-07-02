
========================================================================
  DAISY REGRESSION REPORT COMPARISON
  Generated: 2026-07-02T07:04:31.264498+00:00
========================================================================

------------------------------------------------------------------------
  Metric                               Before        After        Delta
------------------------------------------------------------------------
  Overall Pass Rate                     57.1%        55.4%        -1.8%
  Total Passed                             32           31           -1
  Total Failed                             24           25           +1
  Total Cases                              56           56           +0
------------------------------------------------------------------------

  PER-CLUSTER BREAKDOWN
------------------------------------------------------------------------
  Cluster             Before      After      Delta    Direction
------------------------------------------------------------------------
  anxiety              50.0%      75.0%     +25.0%   improved тЖС
  breakup              75.0%     100.0%     +25.0%   improved тЖС
  clarity              37.5%      25.0%     -12.5%  regressed тЖУ
  grief                62.5%      25.0%     -37.5%  regressed тЖУ
  somatic              87.5%      75.0%     -12.5%  regressed тЖУ
  stress               50.0%      62.5%     +12.5%   improved тЖС
  work                 37.5%      25.0%     -12.5%  regressed тЖУ
------------------------------------------------------------------------

  PER-LOCALE BREAKDOWN
------------------------------------------------------------------------
  Locale              Before      After      Delta    Direction
------------------------------------------------------------------------
  en                   50.0%      39.3%     -10.7%  regressed тЖУ
  ru                   64.3%      71.4%      +7.1%   improved тЖС
------------------------------------------------------------------------

  FAILURE ANALYSIS
------------------------------------------------------------------------
  Resolved: (none)
  New (count increased):
    тЭМ keyword_mismatch: +1
  Remaining (still present):
      keyword_mismatch: 25
------------------------------------------------------------------------

  REGRESSED CASES (11): passed before, failed after
    тЭМ clarity_ru_2 (clarity/ru) тАФ keyword_mismatch
    тЭМ clarity_ru_4 (clarity/ru) тАФ keyword_mismatch
    тЭМ grief_en_2 (grief/en) тАФ keyword_mismatch
    тЭМ grief_en_3 (grief/en) тАФ keyword_mismatch
    тЭМ grief_en_4 (grief/en) тАФ keyword_mismatch
    тЭМ grief_ru_2 (grief/ru) тАФ keyword_mismatch
    тЭМ somatic_en_1 (somatic/en) тАФ keyword_mismatch
    тЭМ somatic_en_4 (somatic/en) тАФ keyword_mismatch
    тЭМ stress_en_2 (stress/en) тАФ keyword_mismatch
    тЭМ work_en_3 (work/en) тАФ keyword_mismatch
    тЭМ work_ru_3 (work/ru) тАФ keyword_mismatch

  IMPROVED CASES (10): failed before, passed after
    тЬЕ anxiety_en_2 (anxiety/en)
    тЬЕ anxiety_en_4 (anxiety/en)
    тЬЕ breakup_en_3 (breakup/en)
    тЬЕ breakup_ru_1 (breakup/ru)
    тЬЕ clarity_ru_1 (clarity/ru)
    тЬЕ grief_ru_1 (grief/ru)
    тЬЕ somatic_en_2 (somatic/en)
    тЬЕ stress_ru_2 (stress/ru)
    тЬЕ stress_ru_3 (stress/ru)
    тЬЕ work_ru_1 (work/ru)

========================================================================
  RELEASE GATES
========================================================================
    тЭМ Overall pass rate >= 90%
    тЭМ Per-cluster pass rate >= 85%
    тЬЕ Zero structural_leak failures
    тЬЕ Zero script_leak failures

  тЭМ DO NOT RELEASE
========================================================================

