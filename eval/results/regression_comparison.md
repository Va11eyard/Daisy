
========================================================================
  DAISY REGRESSION REPORT COMPARISON
  Generated: 2026-07-02T02:47:45.140597+00:00
========================================================================

------------------------------------------------------------------------
  Metric                               Before        After        Delta
------------------------------------------------------------------------
  Overall Pass Rate                     28.6%        57.1%       +28.6%
  Total Passed                             16           32          +16
  Total Failed                             40           24          -16
  Total Cases                              56           56           +0
------------------------------------------------------------------------

  PER-CLUSTER BREAKDOWN
------------------------------------------------------------------------
  Cluster             Before      After      Delta    Direction
------------------------------------------------------------------------
  anxiety              12.5%      50.0%     +37.5%   improved тЖС
  breakup              75.0%      75.0%      +0.0%  unchanged тЖТ
  clarity              50.0%      37.5%     -12.5%  regressed тЖУ
  grief                25.0%      62.5%     +37.5%   improved тЖС
  somatic              12.5%      87.5%     +75.0%   improved тЖС
  stress               12.5%      50.0%     +37.5%   improved тЖС
  work                 12.5%      37.5%     +25.0%   improved тЖС
------------------------------------------------------------------------

  PER-LOCALE BREAKDOWN
------------------------------------------------------------------------
  Locale              Before      After      Delta    Direction
------------------------------------------------------------------------
  en                   39.3%      50.0%     +10.7%   improved тЖС
  ru                   17.9%      64.3%     +46.4%   improved тЖС
------------------------------------------------------------------------

  FAILURE ANALYSIS
------------------------------------------------------------------------
  Resolved (count reduced):
    тЬЕ keyword_mismatch: -16
    тЬЕ hollow: -1
  New: (none)
  Remaining (still present):
      keyword_mismatch: 24
------------------------------------------------------------------------

  REGRESSED CASES (7): passed before, failed after
    тЭМ anxiety_en_1 (anxiety/en) тАФ keyword_mismatch
    тЭМ breakup_en_3 (breakup/en) тАФ keyword_mismatch
    тЭМ breakup_ru_1 (breakup/ru) тАФ keyword_mismatch
    тЭМ clarity_en_1 (clarity/en) тАФ keyword_mismatch
    тЭМ clarity_en_3 (clarity/en) тАФ keyword_mismatch
    тЭМ clarity_ru_3 (clarity/ru) тАФ keyword_mismatch
    тЭМ work_ru_2 (work/ru) тАФ keyword_mismatch

  IMPROVED CASES (23): failed before, passed after
    тЬЕ anxiety_ru_1 (anxiety/ru)
    тЬЕ anxiety_ru_2 (anxiety/ru)
    тЬЕ anxiety_ru_3 (anxiety/ru)
    тЬЕ anxiety_ru_4 (anxiety/ru)
    тЬЕ breakup_en_2 (breakup/en)
    тЬЕ breakup_ru_3 (breakup/ru)
    тЬЕ clarity_ru_2 (clarity/ru)
    тЬЕ clarity_ru_4 (clarity/ru)
    тЬЕ grief_en_3 (grief/en)
    тЬЕ grief_ru_2 (grief/ru)
    тЬЕ grief_ru_4 (grief/ru)
    тЬЕ somatic_en_3 (somatic/en)
    тЬЕ somatic_en_4 (somatic/en)
    тЬЕ somatic_ru_1 (somatic/ru)
    тЬЕ somatic_ru_2 (somatic/ru)
    тЬЕ somatic_ru_3 (somatic/ru)
    тЬЕ somatic_ru_4 (somatic/ru)
    тЬЕ stress_en_1 (stress/en)
    тЬЕ stress_en_4 (stress/en)
    тЬЕ stress_ru_4 (stress/ru)
    тЬЕ work_en_3 (work/en)
    тЬЕ work_ru_3 (work/ru)
    тЬЕ work_ru_4 (work/ru)

========================================================================
  RELEASE GATES
========================================================================
    тЭМ Overall pass rate >= 90%
    тЭМ Per-cluster pass rate >= 85%
    тЬЕ Zero structural_leak failures
    тЬЕ Zero script_leak failures

  тЭМ DO NOT RELEASE
========================================================================

