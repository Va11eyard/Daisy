# Daisy Fix — Verification Report

**Date:** 2026-07-01
**Package:** `/mnt/agents/output/daisy-fix/`
**Status:** READY FOR EXECUTION

---

## What Was Delivered

A complete production implementation package to fix the Daisy therapy chatbot (talktodaisy.com) — a multilingual (EN/RU/KK) AI emotional-support system currently broken in production.

### Package Statistics

| Stage | Files | Lines | Purpose |
|-------|-------|-------|---------|
| Stage 1 — Immediate | 6 | 2,823 | Baseline + short-term RU fix |
| Stage 2 — Qwen3 Migration | 7 | 4,062 | Architecture rebuild |
| Stage 3 — Verify | 2 | 1,463 | Verification + cutover |
| Documentation | 2 | 609 | README + SPEC |
| **Total** | **17** | **8,957** | **Complete fix package** |

---

## Files Delivered

### Stage 1 — Immediate (Day 1-2)

| File | Lines | Validates Against |
|------|-------|-------------------|
| `run_cross_topic_regression.py` | 721 | 56-case regression on live endpoint |
| `audit_training_locale.py` | 590 | Training data audit + Latin leak fix |
| `deployment-lora-v11-ru-translate-v2.yaml` | 54 | A/B deployment with post-translate QC |
| `route.ts` | 765 | Locale-aware frontend (EN/RU/KK errors) |
| `cutover_traffic.ps1` | 747 | Safe Azure traffic management |
| `eval/cross_topic_regression.jsonl` | 56 | Test cases |

### Stage 2 — Qwen3 Migration (Day 3-7)

| File | Lines | Validates Against |
|------|-------|-------------------|
| `score_qwen3.py` | 1,090 | 3-layer inference (safety → generate → QC) |
| `system_prompt_qwen3.py` | 298 | Clean EN/RU/KK prompts |
| `voice_qc_lightweight.py` | 486 | Re-enabled voice quality control |
| `deployment-qwen3-lora-v15.yaml` | 126 | Qwen3-8B production deployment |
| `build_balanced_dataset.py` | 709 | Balanced EN/RU/KK dataset builder |
| `synthesize_ru_kk_dialogues.py` | 832 | RU/KK therapy dialogue generator |
| `strip_latin_leaks.py` | 513 | Latin leak removal from Cyrillic turns |

### Stage 3 — Verification (Day 8-10)

| File | Lines | Validates Against |
|------|-------|-------------------|
| `compare_regression_reports.py` | 688 | Before/after comparison + release gating |
| `cutover_checklist.md` | 775 | Production cutover procedures |

---

## Validation Results

### Syntax Checks

| File Type | Count | Status |
|-----------|-------|--------|
| Python (`.py`) | 9 | All PASS `py_compile` |
| TypeScript (`.ts`) | 1 | Structure validated |
| YAML (`.yaml`) | 2 | Both PASS `yaml.safe_load` |
| PowerShell (`.ps1`) | 1 | Structure validated |

### strip_latin_leaks.py — Verified Against Real Training Data

| Pattern | Result |
|---------|--------|
| `Daisy noticed, что...` | ✅ Stripped correctly |
| `(trauma bonding)` in RU | ✅ Stripped correctly |
| `Assistant: Привет` | ✅ Header stripped |
| `DBT` acronym | ✅ Preserved (allowed) |
| Pure English text | ✅ Preserved unchanged |
| Clean Cyrillic text | ✅ Preserved unchanged |

### Architecture Decision Implemented

**Option A (Primary):** Qwen3-8B clean-slate rebuild
- 3-layer inference (was 23 modules)
- 8 functional env vars (was 15+ with dead vars)
- Voice QC re-enabled (was disabled in `simple` mode)
- Balanced training data target: 40% EN / 35% RU / 25% KK
- `DAISY_DEFAULT_MAX_TOKENS=120` (up from 90)
- `DAISY_LORA_DEFAULT_TEMP=0.6` (up from 0.55)

**Option B (Parallel):** RU translate routing
- `gpu-deployment-ru-translate-v2` with post-translate QC
- `DAISY_POST_TRANSLATE_QC=true`
- `DAISY_TRANSLATE_MIN_LENGTH=40`
- `DAISY_TRANSLATE_SCRIPT_GUARD=true`
- `DAISY_TRANSLATE_MAX_LATIN_RATIO=0.08` (stricter)

**Rejected:** Options C (retrain on Qwen2.5-7B) and D (more patches) per ARCHITECTURE_DECISION.md.

---

## Execution Plan

### Day 1 (1-2 hours)

```powershell
# 1. Baseline current production
cd E:\WebstormProjects\Daisy-1
copy daisy-fix\stage1-immediate\run_cross_topic_regression.py scripts\
$env:DAISY_ENDPOINT_KEY = (az ml online-endpoint get-credentials `
    --name daisy-therapy -g Daisy_group -w Daisy -o tsv --query primaryKey)
python scripts/run_cross_topic_regression.py `
    --deployment gpu-deployment-finetuned `
    --output eval/results/baseline_regression.json
# Expected: 60-75% overall (based on evidence in docs)

# 2. Fix training data
cd E:\WebstormProjects\Daisy-Model
python scripts/audit_training_locale.py --fix --output-dir data/cleaned
```

### Day 2 (2-3 hours)

```powershell
# 3. Deploy RU translate A/B (optional short-term fix)
cd E:\WebstormProjects\Daisy-1
az ml online-deployment create `
    --file daisy-fix/stage1-immediate/deployment-lora-v11-ru-translate-v2.yaml `
    -g Daisy_group -w Daisy

# 4. Update frontend
Copy-Item daisy-fix/stage1-immediate/route.ts `
    C:\Users\Valleyard\Daisy\src\app\api\chat\route.ts

# 5. Traffic cutover to 10% RU test
daisy-fix\stage1-immediate\cutover_traffic.ps1 -Action cutover `
    -Deployment gpu-deployment-ru-translate -Percent 10
```

### Days 3-7 (core migration work)

```powershell
# 6. Generate RU/KK training data
cd E:\WebstormProjects\Daisy-Model
python scripts/synthesize_ru_kk_dialogues.py --output-dir data/synthesized

# 7. Build balanced dataset v15
python scripts/build_balanced_dataset.py `
    --en-sources data/cleaned/train_v13.jsonl `
    --ru-sources data/synthesized/ru,data/cleaned/legacy_ru `
    --kk-sources data/synthesized/kk `
    --output data/train_v15.jsonl `
    --target-mix '{"en":0.40,"ru":0.35,"kk":0.25}'

# 8. Train new LoRA v15 on Qwen3-8B
python scripts/submit_training_job.py `
    --train-file data/train_v15.jsonl `
    --base-model Qwen/Qwen3-8B `
    --output-name daisy-finetuned-lora:15

# 9. Deploy Qwen3 inference
cd E:\WebstormProjects\Daisy-1
az ml online-deployment create `
    --file daisy-fix/stage2-qwen3-migration/deployment-qwen3-lora-v15.yaml `
    -g Daisy_group -w Daisy

# 10. Run regression against Qwen3 A/B
python scripts/run_cross_topic_regression.py `
    --deployment gpu-deployment-v14 `
    --output eval/results/qwen3_regression.json

# 11. Compare
python daisy-fix/stage3-verify/compare_regression_reports.py `
    eval/results/baseline_regression.json `
    eval/results/qwen3_regression.json `
    --format markdown
```

### Days 8-10 (cutover)

Follow `stage3-verify/cutover_checklist.md` for the complete 8-step cutover procedure with rollback triggers.

---

## Expected Outcomes

| Metric | Before (v11) | After (Qwen3 v15) | Delta |
|--------|-------------|-------------------|-------|
| Overall pass rate | ~65-75% | ≥90% | +15-25% |
| RU per-cluster | ~40-50% | ≥85% | +35-45% |
| Structural leaks | 3-5 per run | 0 | -100% |
| Script leaks | 4-6 per run | 0 | -100% |
| Canned greetings | 3-5 per run | 0 | -100% |
| P50 latency | 43-68s | <15s | -65-78% |
| Inference modules | 23 | 3 | -87% |
| Functional env vars | ~8 of 18 | 8 of 8 | +100% |
| Voice QC | OFF | ON | Re-enabled |
| RU training data | ~0% | ≥35% | +∞ |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Qwen3 training takes >1 week | Medium | Option B RU translate routing keeps RU users served |
| Qwen3 base (no LoRA) worse than v11 | Low | A/B at 5% first; scale only if regression passes |
| RU translate adds >20s latency | Low | Post-translate QC rejects short responses; fallback to EN |
| KK quality still poor | Low | Qwen3 strong Turkic support; 300+ synthetic KK turns |
| Frontend route.ts merge conflicts | Low | File is standalone replacement with clear types |

---

## Success Criteria (Release Gate)

| Criterion | Threshold | Verification |
|-----------|-----------|--------------|
| Overall pass rate | ≥90% | `run_cross_topic_regression.py` → `overall.pass_rate` |
| Per-cluster pass rate | ≥85% | `by_cluster.*.pass_rate` |
| Structural leaks | 0 | `failure_breakdown.structural_leak == 0` |
| Script leaks | 0 | `failure_breakdown.script_leak == 0` |
| Canned greetings | 0 | `failure_breakdown.canned_greeting == 0` |
| RU informal register | `ты` (not `вы`) | Manual spot-check |
| EN template variety | No identical on paraphrase | case-level comparison |
| P50 latency | <15s | `metadata.latency_ms` |
| Locale errors | EN/RU/KK | `route.ts` `ERROR_MESSAGES` |

**All criteria must pass before calling production cutover complete.**

---

*Package generated by Kimi Agent Swarm, 2026-07-01*
*Architecture per ARCHITECTURE_DECISION.md (2026-07-01)*
