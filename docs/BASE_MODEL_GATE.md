# Base model swap — decision gate

Do **not** change `BASE_MODEL` until v13 + inference v9 fail these gates on live probes.

## Quality gates (must pass on v13 + v9)

| Metric | Target | How to measure |
|--------|--------|----------------|
| `cross_scenario_genericness` | < 0.18 | `eval/ablation_runner.py` or `eval/endpoint_baseline.py` |
| RU anxiety / meta-feedback | pass | `scripts/verify_deploy.py` |
| `reflect_plus_question_shape_rate` (train) | < 0.70 | `eval/analyze_offline.py --train data/train_v13.jsonl` |

## Latency gates

| Metric | Target | How to measure |
|--------|--------|----------------|
| P50 latency | < 12s | `scripts/latency_probe.py` |
| P90 `total_regen_count` | <= 1 | `debug_context` in endpoint responses |

## If gates fail

1. Expand curated data (`scripts/build_v13_shape_synth.py`, RU distill).
2. Re-run v13 training.
3. Only then evaluate base swap:

| Candidate | When to try |
|-----------|-------------|
| `Qwen/Qwen2.5-3B-Instruct` | Latency still bad, quality acceptable on small probe |
| `meta-llama/Llama-3.1-8B-Instruct` | EN quality plateau, RU/KK validated on holdout |
| `mistralai/Mistral-7B-Instruct-v0.3` | EN-only traffic acceptable |

Any swap requires re-render of full dataset + new LoRA + full `verify_deploy` regression.

## Current status

- **Base model:** `Qwen/Qwen2.5-7B-Instruct` (unchanged)
- **Gate evaluation:** run after v13 deploy; document results in `eval/results/v13_gate_report.json`

```powershell
python eval/analyze_offline.py --train data/train_v13.jsonl
python scripts/verify_deploy.py
python scripts/latency_probe.py
python eval/endpoint_baseline.py --limit 10
```
