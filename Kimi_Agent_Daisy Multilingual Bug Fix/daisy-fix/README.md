# Daisy Fix — Production Implementation Package

**Date:** 2026-07-01
**Status:** Ready for execution
**Architecture Decision:** Option A (Qwen3 rebuild) + Option B (RU translate routing as short-term)

---

## What This Is

A complete implementation package to fix the Daisy therapy chatbot (talktodaisy.com) — a multilingual (EN/RU/KK) AI emotional-support system that is currently broken in production due to train/serve misalignment, 99.9%-English training data, 23-module inference debt, and disabled quality controls.

## Root Causes Fixed

| Root Cause | Fix |
|-----------|-----|
| 99.9% EN training data, 0 RU rows | `build_balanced_dataset.py` + `synthesize_ru_kk_dialogues.py` → ≥35% RU, ≥25% KK |
| "Daisy noticed" Latin leaks in Cyrillic turns | `strip_latin_leaks.py` + `audit_training_locale.py` — cleans 26% leak rate to 0% |
| 23-module inference, dead env vars, QC disabled | `score_qwen3.py` — 3-layer pipeline, 8 functional env vars, voice QC re-enabled |
| Canned hollow therapy templates | VoiceQC banned patterns + no fallback substitution + higher temperature (0.6) |
| Structural/script leaks in production | Layer-2 QC with stop strings, script leak detection, one regen on failure |
| Hardcoded Russian error for all locales | `route.ts` — locale-aware EN/RU/KK error messages |
| No deployment routing for RU/KK | `route.ts` + `deployment-lora-v11-ru-translate-v2.yaml` — locale-split deployment headers |
| No automated quality gate | `run_cross_topic_regression.py` — 56-case cross-topic regression with per-cluster reporting |

## Package Contents

```
daisy-fix/
├── SPEC.md                              # Architecture & interface specification
├── README.md                            # This file
│
├── stage1-immediate/                    # DAY 1-2: Baseline + Short-term RU fix
│   ├── run_cross_topic_regression.py    # 56-case regression runner (baseline production)
│   ├── audit_training_locale.py         # Training data audit + Latin leak fix
│   ├── deployment-lora-v11-ru-translate-v2.yaml  # RU translate A/B with post-translate QC
│   ├── route.ts                         # Locale-aware frontend (EN/RU/KK errors + routing)
│   ├── cutover_traffic.ps1              # Safe Azure traffic management
│   └── eval/cross_topic_regression.jsonl # 56 test cases (8 clusters × EN/RU)
│
├── stage2-qwen3-migration/              # DAY 3-7: Architecture rebuild
│   ├── score_qwen3.py                   # 3-layer inference (safety → generate → QC)
│   ├── system_prompt_qwen3.py           # Clean EN/RU/KK prompts (no dead vars)
│   ├── voice_qc_lightweight.py          # Re-enabled voice quality control
│   ├── deployment-qwen3-lora-v15.yaml   # Qwen3-8B production deployment
│   ├── build_balanced_dataset.py        # Balanced EN/RU/KK dataset builder
│   ├── synthesize_ru_kk_dialogues.py    # RU/KK therapy dialogue generator
│   └── strip_latin_leaks.py             # Latin leak removal utility
│
└── stage3-verify/                       # DAY 8-10: Verification & cutover
    ├── compare_regression_reports.py    # Before/after comparison + release gating
    └── cutover_checklist.md             # Production cutover procedures
```

**Total: 15 files, ~8,000 lines**

---

## Quick Start — What To Run First

### 1. Baseline Production (Day 1, 30 minutes)

```powershell
# Set up
cd E:\WebstormProjects\Daisy-1
copy daisy-fix\stage1-immediate\run_cross_topic_regression.py scripts\
copy daisy-fix\stage1-immediate\eval\cross_topic_regression.jsonl eval\

# Get endpoint key
$env:DAISY_ENDPOINT_KEY = (az ml online-endpoint get-credentials `
    --name daisy-therapy -g Daisy_group -w Daisy `
    -o tsv --query primaryKey)

# Run baseline against current production
python scripts/run_cross_topic_regression.py `
    --deployment gpu-deployment-finetuned `
    --output eval/results/baseline_regression.json

# View report
cat eval/results/baseline_regression.json
```

**Expected:** Current production likely scores 60-75% overall (based on evidence in docs). RU clusters likely <50%.

### 2. Fix Training Data (Day 1-2)

```powershell
cd E:\WebstormProjects\Daisy-Model
copy daisy-fix\stage1-immediate\audit_training_locale.py scripts\
copy daisy-fix\stage2-qwen3-migration\strip_latin_leaks.py scripts\
copy daisy-fix\stage2-qwen3-migration\synthesize_ru_kk_dialogues.py scripts\

# Audit current data
python scripts/audit_training_locale.py --report eval/results/audit_before.json

# Fix Latin leaks
python scripts/audit_training_locale.py --fix --output-dir data/cleaned

# Generate RU/KK dialogues
python scripts/synthesize_ru_kk_dialogues.py --output-dir data/synthesized

# Build balanced dataset v15
python scripts/build_balanced_dataset.py `
    --en-sources data/cleaned/train_v13.jsonl `
    --ru-sources data/synthesized/ru,data/cleaned/legacy_ru `
    --kk-sources data/synthesized/kk `
    --output data/train_v15.jsonl `
    --target-mix '{"en":0.40,"ru":0.35,"kk":0.25}'
```

### 3. Short-Term RU Routing (Day 2, optional)

```powershell
cd E:\WebstormProjects\Daisy-1

# Deploy updated RU translate A/B
az ml online-deployment create `
    --file daisy-fix/stage1-immediate/deployment-lora-v11-ru-translate-v2.yaml `
    -g Daisy_group -w Daisy

# Route RU traffic (10% test)
.\daisy-fix\stage1-immediate\cutover_traffic.ps1 -Action cutover `
    -Deployment gpu-deployment-ru-translate -Percent 10

# Update frontend
Copy-Item daisy-fix\stage1-immediate\route.ts `
    C:\Users\Valleyard\Daisy\src\app\api\chat\route.ts
```

### 4. Qwen3 Migration (Day 3-7)

```powershell
cd E:\WebstormProjects\Daisy-1

# Train new LoRA v15 on balanced data (Azure ML training job)
python scripts/submit_training_job.py `
    --train-file data/train_v15.jsonl `
    --base-model Qwen/Qwen3-8B `
    --output-name daisy-finetuned-lora:15

# Deploy Qwen3 inference
az ml online-deployment create `
    --file daisy-fix/stage2-qwen3-migration/deployment-qwen3-lora-v15.yaml `
    -g Daisy_group -w Daisy

# Run regression against Qwen3 A/B
python scripts/run_cross_topic_regression.py `
    --deployment gpu-deployment-v14 `
    --output eval/results/qwen3_regression.json

# Compare
python daisy-fix/stage3-verify/compare_regression_reports.py `
    eval/results/baseline_regression.json `
    eval/results/qwen3_regression.json `
    --format markdown
```

### 5. Production Cutover (Day 8-10)

Follow `stage3-verify/cutover_checklist.md` for the complete cutover procedure.

```powershell
# Quick cutover (after checklist validation)
.\daisy-fix\stage1-immediate\cutover_traffic.ps1 -Action cutover `
    -Deployment gpu-deployment-v14 -Percent 100 -Confirm
```

---

## Architecture Changes Summary

### Before (v11 — current production)
```
Qwen2.5-7B-Instruct + LoRA v11 (99.9% EN training)
  → 23-module inference (score.py)
  → simple mode (voice QC OFF)
  → 15+ regex cleaners
  → dead env vars: DAISY_RAG, DAISY_CONFIDENCE_GATE, etc.
  → gpu-deployment-finetuned (100% traffic)
```

### After (Qwen3 v15 — target)
```
Qwen3-8B + LoRA v15 (40% EN / 35% RU / 25% KK training)
  → 3-layer inference (score_qwen3.py)
  → voice QC ON (re-enabled)
  → 8 functional env vars only
  → gpu-deployment-v14 (100% traffic after cutover)
```

### Short-Term (RU translate — parallel)
```
Same LoRA v11
  → EN generate → Azure translate → RU
  → post-translate QC (min length, script guard, fallback)
  → gpu-deployment-ru-translate (RU traffic only)
```

---

## Success Criteria

| Criterion | Threshold | How to Verify |
|-----------|-----------|---------------|
| Overall pass rate | ≥90% | `run_cross_topic_regression.py` → overall.pass_rate |
| Per-cluster pass rate | ≥85% | by_cluster.*.pass_rate |
| Structural leaks | 0 | failure_breakdown.structural_leak == 0 |
| Script leaks | 0 | failure_breakdown.script_leak == 0 |
| Canned greeting | 0 | failure_breakdown.canned_greeting == 0 |
| RU informal | ты (not вы) | Manual spot-check + keyword_match on RU cases |
| EN template variety | No identical on paraphrase | case-level comparison |
| P50 latency | <15s | metadata.latency_ms from regression report |
| Locale errors | EN/RU/KK | route.ts ERROR_MESSAGES |

---

## Key Design Decisions

1. **Qwen3-8B over Qwen2.5-7B**: Stronger multilingual (RU/KK) capability, Apache-2.0 license, better instruction following.
2. **3-layer inference over 23-module**: Removes dead code, simplifies debugging, enables streaming.
3. **Temperature 0.6 (up from 0.55)**: More natural variation, reduces template repetition.
4. **Max tokens 120 (up from 90)**: More complete responses without the garbage that caused the 768-token floor bug.
5. **Voice QC re-enabled**: Was disabled in `simple` mode — primary cause of hollow one-liners shipping.
6. **No RAG in v1**: RAG added significant complexity and was OFF in production anyway. Can be re-added cleanly after base quality is solid.
7. **No confidence gate**: Was never wired. QC layer handles quality instead.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Qwen3 training takes longer than expected | Short-term RU translate routing keeps RU users served |
| Qwen3 base (no LoRA) worse than v11 | A/B at 5% first; only scale if regression passes |
| RU translate adds too much latency | P50 budget <20s acceptable per ARCHITECTURE_DECISION.md |
| KK quality still poor even with Qwen3 | Qwen3 has strong Turkic support; synthetic data adds 300+ KK turns |
| Training data synthesis quality | All synthetic turns validated: no leaks, informal ты, ≥50 chars |

---

## Azure Resources

| Resource | Value |
|----------|-------|
| Subscription | `9239bc75-105c-486e-8957-da8e49309c55` |
| Resource Group | `Daisy_group` |
| Workspace | `Daisy` |
| Region | `westus2` |
| Endpoint | `daisy-therapy` |
| GPU | `Standard_NC4as_T4_v3` (T4 16GB) |

### Deployment Slots

| Slot | Current | After Cutover |
|------|---------|---------------|
| `gpu-deployment-finetuned` | 100% (LoRA v11) | 0% → retired |
| `gpu-deployment-ru-translate` | 0% (A/B) | RU traffic (optional) |
| `gpu-deployment-v14` | 0% (Qwen3 base) | 100% (LoRA v15) |

---

## Files Map (Repo → Package)

| Production File | Package File | Action |
|-----------------|--------------|--------|
| `E:\WebstormProjects\Daisy-1\inference\score.py` | `stage2-qwen3-migration\score_qwen3.py` | Replace |
| `E:\WebstormProjects\Daisy-1\inference\system_prompt.py` | `stage2-qwen3-migration\system_prompt_qwen3.py` | Replace |
| `E:\WebstormProjects\Daisy-1\inference\voice_qc.py` | `stage2-qwen3-migration\voice_qc_lightweight.py` | Replace |
| `E:\WebstormProjects\Daisy-1\azureml\deployment-lora-v11-natural2.yaml` | `stage1-immediate\deployment-lora-v11-ru-translate-v2.yaml` | Update for A/B |
| `E:\WebstormProjects\Daisy-1\azureml\deployment-v14.yaml` | `stage2-qwen3-migration\deployment-qwen3-lora-v15.yaml` | Replace |
| `C:\Users\Valleyard\Daisy\src\app\api\chat\route.ts` | `stage1-immediate\route.ts` | Replace |
| `E:\WebstormProjects\Daisy-1\scripts\run_cross_topic_regression.py` | `stage1-immediate\run_cross_topic_regression.py` | Create new |
| `E:\WebstormProjects\Daisy-Model\scripts\audit_training_locale.py` | `stage1-immediate\audit_training_locale.py` | Create new |
| `E:\WebstormProjects\Daisy-1\scripts\cutover_traffic.ps1` | `stage1-immediate\cutover_traffic.ps1` | Create new |

---

## Support

For issues during execution, refer to:
- `SPEC.md` — detailed interface specifications
- `stage3-verify/cutover_checklist.md` — step-by-step procedures with rollback
- Azure ML docs: https://docs.microsoft.com/azure/machine-learning/

---

*Generated by Kimi Agent Swarm, 2026-07-01*
