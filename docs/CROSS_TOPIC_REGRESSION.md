# Cross-topic regression eval

56 live endpoint cases across **8 clusters** × **EN/RU** (not anxiety-only).

## Clusters

| Cluster | EN cases | RU cases | What it catches |
|---------|----------|----------|-----------------|
| breakup | 4 | 4 | Hollow breakup template, grammar, emptiness |
| work | 4 | 4 | CEO/boss stress, canned greeting |
| anxiety | 4 | 4 | Script leaks, error bubble scenarios |
| stress | 4 | 4 | Formulaic multi-question templates |
| grief | 4 | 4 | Loss language, inappropriate cheer |
| clarity | 4 | 4 | Thought-sorting requests |
| somatic | 4 | 4 | Body-focused disclosure |

## Files

- Cases: [`eval/cross_topic_regression.jsonl`](../eval/cross_topic_regression.jsonl)
- Runner: [`scripts/run_cross_topic_regression.py`](../scripts/run_cross_topic_regression.py)
- Generator: [`scripts/generate_cross_topic_eval.py`](../scripts/generate_cross_topic_eval.py)
- Report output: `eval/results/cross_topic_regression_report.json`

## Run

```powershell
cd E:\WebstormProjects\Daisy-1
$env:DAISY_ENDPOINT_KEY = (az ml online-endpoint get-credentials --name daisy-therapy -g Daisy_group -w Daisy -o tsv --query primaryKey)

# Production (default gpu-deployment-finetuned)
python scripts/run_cross_topic_regression.py

# A/B deployment
python scripts/run_cross_topic_regression.py --deployment gpu-deployment-ru-translate

# Smoke subset
python scripts/run_cross_topic_regression.py --limit 8 --delay 3
```

## Pass criteria (per case)

- Response length ≥ 25 chars
- No canned greeting (`Hey — I'm glad you're here...`)
- No structural leak (`Assistant:`, rubric tokens, `.,.,.`)
- RU/KK: no script leak (`generation_has_script_leak`)
- At least one topic keyword from case definition
- No hollow one-liner openers on short replies

## Success gate (release)

- **≥ 90%** pass rate overall
- **≥ 85%** per cluster (both locales)
- **0** `script_leak` or `structural_leak` failures in full 56-case run
