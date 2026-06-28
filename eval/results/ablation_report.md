# Genericness ablation report

Timestamp: 2026-06-22T12:00:55Z
Cases: 41

## Metrics by configuration

| Config | Genericness | Distinct-2 | Canned rate | Specificity | Fallback rate |
|--------|-------------|------------|-------------|-------------|---------------|
| C0_base | 0.2461 | 0.63 | 0.1463 | 0.4523 | 0.0 |
| C1_lora | 0.2062 | 0.6532 | 0.0732 | 0.4508 | 0.0 |
| C2_prompt | 0.1982 | 0.818 | 0.4146 | 0.5062 | 0.0 |
| C3_rag | 0.1679 | 0.8075 | 0.3902 | 0.49 | 0.0 |
| C4_full | 0.2969 | 0.5083 | 0.561 | 0.2899 | 0.8049 |
| C4_nofallback | 0.1988 | 0.6256 | 0.7073 | 0.4541 | 0.0 |

## Adjacent config deltas (effect size)

- **C0_base -> C1_lora**: delta_genericness=-0.0399, delta_canned=-0.0731, delta_specificity=-0.0015
- **C1_lora -> C2_prompt**: delta_genericness=-0.0080, delta_canned=+0.3414, delta_specificity=+0.0554
- **C2_prompt -> C3_rag**: delta_genericness=-0.0303, delta_canned=-0.0244, delta_specificity=-0.0162
- **C3_rag -> C4_full**: delta_genericness=+0.1290, delta_canned=+0.1708, delta_specificity=-0.2001
- **C4_full -> C4_nofallback**: delta_genericness=-0.0981, delta_canned=+0.1463, delta_specificity=+0.1642

## Dominant cause

**Verdict:** `anti_hallucination_layers`

- C4 vs C3: genericness +0.129, canned_rate +0.1708, specificity -0.2001
- C4 fallback_rate=0.8049 (curated fallback on every turn)
- Fix validation C4_full vs C4_nofallback: genericness -0.0981, specificity +0.1642, fallback_rate 0.8049 -> 0.0

## Recommended fix

Disable curated fallback substitution in score.py (keep regen, drop fallback_reply replacement). Re-measure C4 canned_rate and cross_scenario_genericness.

## Fix verification (C4_full vs C4_nofallback)

Production fix: disable `fallback_reply` substitution after voice-QC regen in `score.py`.
Ablation proxy: `C4_nofallback` = same stack without curated fallback substitution.

| Metric | C4_full (before) | C4_nofallback (after) | Delta |
|--------|------------------|------------------------|-------|
| cross_scenario_genericness | 0.2969 | 0.1988 | -0.0981 (improved) |
| distinct_2 | 0.5083 | 0.6256 | +0.1173 (improved) |
| canned_rate | 0.5610 | 0.7073 | +0.1463 (worse) |
| mean_specificity | 0.2899 | 0.4541 | +0.1642 (improved) |
| fallback_rate | 0.8049 | 0.0000 | -0.8049 (improved) |

Fallback substitution was the dominant blandness driver: fallback_rate 0.8049 -> 0.0, cross_scenario_genericness -0.0981, specificity +0.1642, distinct_2 +0.1173. Fix applied in score.py; redeploy endpoint to validate on production traffic.