# Daisy Fix Plan — Production Implementation

## Context Summary
- **Product**: talktodaisy.com — multilingual (EN/RU/KK) AI emotional-support chatbot
- **Root cause**: 99.9% EN training data → LoRA v11 cannot produce natural RU; 23-module inference with dead env vars and QC disabled
- **Current prod**: gpu-deployment-finetuned 100%, LoRA v11, simple mode, natural4-ru build
- **A/B slots**: gpu-deployment-ru-translate (0%), gpu-deployment-v14 Qwen3-8B (0%)

## Decision
**Primary: Option A (Clean-slate Qwen3 rebuild)** — ARCHITECTURE_DECISION.md recommended path.
**Short-term parallel: Option B (RU translate routing)** — immediate RU stabilization.
**Reject**: Option C (retrain on Qwen2.5-7B) — still English-centric base; Option D (patches) — already exhausted.

## Stages

### Stage 1 — Immediate Baseline & Short-Term RU Fix (Day 1-2)
**Agents**: Regression_Runner, RU_Translate_Router
1. Create `run_cross_topic_regression.py` — complete 56-case runner with per-cluster reporting
2. Create updated `deployment-lora-v11-ru-translate.yaml` — production-ready with post-translate QC
3. Create `route.ts` frontend patch — locale-aware deployment header + error messages
4. Create `audit_training_locale.py` — strip Latin leaks from Cyrillic rows
5. Produce traffic cutover script (`scripts/cutover_traffic.ps1`)

### Stage 2 — Qwen3 Migration Package (Day 3-7)
**Agents**: Qwen3_Migrator, Training_Data_Builder, Inference_Simplifier
1. New deployment YAML: `deployment-qwen3-lora-v15.yaml` for gpu-deployment-v14
2. Simplified inference: `score_qwen3.py` — 3-layer pipeline (safety → generate → QC)
3. Training data builder: `build_balanced_dataset.py` — ≥20% RU, ≥15% KK, strip leaks
4. Dataset synthesis: `synthesize_ru_kk_dialogues.py` — 500+ RU, 300+ KK therapy turns
5. Voice QC re-enabled with lightweight checks

### Stage 3 — Verification & Cutover (Day 8-10)
**Agents**: Verification_Engineer
1. Run regression on A/B deployment
2. Compare before/after per cluster
3. Produce cutover decision matrix
4. Final deployment package with INFERENCE_BUILD bump

## Deliverables
All code written to `/mnt/agents/output/daisy-fix/`:
```
daisy-fix/
├── stage1-immediate/
│   ├── run_cross_topic_regression.py
│   ├── deployment-lora-v11-ru-translate-v2.yaml
│   ├── route.ts.patch
│   ├── audit_training_locale.py
│   └── cutover_traffic.ps1
├── stage2-qwen3-migration/
│   ├── deployment-qwen3-lora-v15.yaml
│   ├── score_qwen3.py
│   ├── build_balanced_dataset.py
│   ├── synthesize_ru_kk_dialogues.py
│   └── voice_qc_lightweight.py
├── stage3-verify/
│   ├── compare_regression_reports.py
│   └── cutover_checklist.md
└── README.md
```

## Success Criteria
- ≥90% pass overall, ≥85% per cluster, 0 structural/script leaks
- RU: informal ты, no EN/PL/DE mid-sentence
- EN: no identical template on paraphrased inputs
- P50 latency <15s on T4
- Locale-aware error messages

## Skill Loading
- Stage 1: vibecoding-general-swarm (Python scripts, Azure YAML, TypeScript patch)
- Stage 2: vibecoding-general-swarm (Python inference pipeline, dataset building)
- Stage 3: report-writing (verification report)
