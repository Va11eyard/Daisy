# Architecture decision: fixing Daisy model quality

**Date:** 2026-07-01  
**Status:** Recommended path for Kimi Agent Swarm  
**Context:** Production still fails on EN/RU across all topics despite 12+ LoRA versions and inference patches. See [issue brief](AZURE_LIVE_STATE.md) and [`eval/results/training_locale_audit.json`](../eval/results/training_locale_audit.json).

---

## Problem statement

Daisy is not a single bug. It is **train/serve misalignment** plus **disabled quality gates** plus **English-dominant training**:

| Layer | Issue |
|-------|--------|
| Training | v11/v12/v13 LoRA: **~99.9% EN rows**; production LoRA `:11` from `train_v3` |
| Training | Cyrillic assistant turns often contain English (`Daisy noticed` in 26% of v12 rows) |
| Inference | Prod uses `DAISY_INFERENCE_MODE=simple` → **voice QC off** |
| Inference | `confidence.py` **unwired**; many YAML env vars are dead |
| Inference | 15+ regex post-processors fight symptoms, not root cause |
| Frontend | No deployment routing; hardcoded RU error on AML failure |

Incremental patches (slim RU prompt, script-leak regen, token cap) improved probes marginally; **user-facing screenshots still show corruption**.

---

## Options evaluated

### Option A — Clean-slate architecture (RECOMMENDED)

**What:** New inference package + `Qwen/Qwen3-8B` (or Qwen3-4B for latency) + new LoRA on balanced EN/RU/KK data + optional RAG from `data/synthesized/*/batch_*.jsonl`.

| Pros | Cons |
|------|------|
| Removes 23-module debt and dead env vars | Highest engineering cost (2–4 weeks) |
| Qwen3 strong RU/KK/EN vs Qwen2.5-7B | Requires full regression suite before traffic |
| Can enable streaming for perceived latency | New LoRA training job + GPU cost |
| RAG on clean per-turn JSON, not ChatML dumps | Team must maintain one code path |

**Deploy path:** `gpu-deployment-v14` already exists at 0% traffic. Train new LoRA on Qwen3, or use base + RAG + tight QC first.

**When to choose:** CEO needs reliable quality on **all topics**; willing to replace stack rather than patch.

---

### Option B — Locale-split routing (RU translate path)

**What:** Keep LoRA v11 for EN; route `locale=ru|kk` to `gpu-deployment-ru-translate` (`DAISY_DIRECT_MULTILINGUAL=false` → EN generate → Azure OpenAI translate back).

| Pros | Cons |
|------|------|
| Uses model's best language (EN) for generation | +latency (translate ×2) |
| Gender-aware RU via translator | Translate path has no post-QC today |
| Deployment already live at 0% traffic | Probes: 6/8 pass vs 7/8 direct; `too_short` failures |
| Frontend change: pass deployment header or split endpoints | Does not fix EN hollow templates |

**When to choose:** Short-term RU stabilization while Option A trains; acceptable if P50 < 20s.

**Implementation:**
1. Set `AML_DEPLOYMENT_NAME=gpu-deployment-ru-translate` for RU in frontend (done: env-driven header).
2. Add post-translate script-leak + min-length check in `score.py`.
3. Traffic rule: RU → translate deployment, EN → finetuned.

---

### Option C — RU-heavy retrain on current stack

**What:** Expand `build_v12_ru_seed.py` + `md_distilled_ru.jsonl`; train `daisy-finetuned-lora:15` on ≥20% RU; serve with **full mode + voice QC on** (not simple).

| Pros | Cons |
|------|------|
| Keeps Qwen2.5-7B + existing Azure setup | v12 hollow-fix v14 showed **no clear win** over v11 |
| Targets root training gap | Still 7B English-centric base |
| Lower risk than full rewrite | Months of inference patches remain |

**When to choose:** Option A blocked by timeline; must stay on Qwen2.5-7B.

**Requirements:**
- Audit/fix Latin-in-Cyrillic rows before training ([`scripts/audit_training_locale.py`](../scripts/audit_training_locale.py))
- `DAISY_INFERENCE_MODE` ≠ `simple` in production
- Run [`scripts/run_cross_topic_regression.py`](../scripts/run_cross_topic_regression.py) before promote

---

### Option D — Inference-only (STATUS: exhausted)

**What:** More regex, slimmer prompts, regen loops — current approach.

| Pros | Cons |
|------|------|
| Fast to ship | **Failed** per user screenshots after `natural4-ru` |
| No retrain cost | Cannot teach Russian grammar or stop all multilingual drift |

**Verdict:** Do not continue as primary strategy.

---

## Decision matrix

| Criterion | A: Qwen3 rebuild | B: RU translate | C: RU retrain | D: Patches only |
|-----------|------------------|-----------------|---------------|-----------------|
| EN quality | High | Medium (unchanged) | Medium | Low |
| RU quality | High (if data balanced) | Medium–High | Medium | Low |
| KK quality | High (Qwen3) | Medium | Low (3 KK rows in v13) | Low |
| Time to prod | 3–4 weeks | 1 week | 2–3 weeks | Days (already done) |
| Latency | Medium | Higher | Medium | Lowest |
| Technical debt | Low (reset) | Medium | High | Very high |

---

## Recommended sequence

1. **Immediate (week 1)**
   - Run full [`cross_topic_regression`](../docs/CROSS_TOPIC_REGRESSION.md) on prod; publish pass rate to stakeholders.
   - Enable locale-aware frontend errors (done).
   - Optional: route RU traffic to `gpu-deployment-ru-translate` for A/B with real users (5–10%).

2. **Short-term (weeks 2–3)**
   - Start **Option A** spike: Qwen3-8B on `gpu-deployment-v14` with simple pipeline + cross-topic eval.
   - Parallel: build balanced EN/RU/KK dataset (target 35/35/30% assistant turns minimum).

3. **Production cutover (week 4+)**
   - Promote only if regression ≥ 90% overall, 0 structural/script leaks.
   - Retire dead modules (`confidence` wire-up or delete; remove unused YAML vars).

4. **Do not**
   - Deploy v12/v13 LoRA to 100% traffic without new RU data ([`V11_PROVENANCE.md`](V11_PROVENANCE.md) + audit).
   - Add more regex cleaners without training or architecture change.

---

## Open product questions

1. Is Kazakh (KK) launch-critical? (v13 has 3 KK training rows.)
2. Acceptable P50 latency for RU translate path?
3. Budget for second GPU during Qwen3 migration?
4. Can CEO/stakeholders accept 2–3 week quality freeze for rebuild?

---

## References

- Live Azure state: [`docs/AZURE_LIVE_STATE.md`](AZURE_LIVE_STATE.md)
- Training audit: [`eval/results/training_locale_audit.json`](../eval/results/training_locale_audit.json)
- Ablation (fallback caused 80% blandness): [`eval/results/ablation_report.md`](../eval/results/ablation_report.md)
- v11 pin: [`docs/V11_PROVENANCE.md`](V11_PROVENANCE.md)
