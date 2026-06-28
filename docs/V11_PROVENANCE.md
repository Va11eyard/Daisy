# daisy-finetuned-lora:11 provenance

## Production pin (2026-06)

**Current prod stack** (after pre-June restore):

| Component | Value |
|-----------|--------|
| LoRA | `azureml:daisy-finetuned-lora:11` |
| Inference | Pre-June restore (`INFERENCE_BUILD=2026-04-pre-june-restore`) |
| Base | `Qwen/Qwen2.5-7B-Instruct` + LoRA adapter |
| Endpoint | `daisy-therapy` / `gpu-deployment-finetuned` |

**Deploy:** `scripts/deploy_pre_june.ps1` (patch: `azureml/deployment-pre-june-patch.yaml`)

**Parked until pre-June stack is stable:**

- `daisy-finetuned-lora:12` and `:13` — do not deploy
- `data/archive/train.jsonl.retired` — bad 14k dataset; use `train_v13.jsonl` only after retrain decision
- June inference layers: book RAG, rubric judge, voice-QC regen loops, `ensure_open_question` canned tails

## Registry

- **Model:** `azureml:daisy-finetuned-lora:11`
- **Training job:** train_v3 (`hungry_kiwi`)

## Training data (expected)

`scripts/submit_training_job.py` defaults to:

1. `TRAIN_FILE=train_v3.jsonl` when present (else `train_v2.jsonl`)
2. `VAL_FILE=val_v3.jsonl` when present (else `val_v2.jsonl`)

`train_v3.jsonl` / `train_v2.jsonl` use the **full voice contract** system prompt:

- CRITICAL OUTPUT RULES
- NEVER USE / NEVER CLOSE WITH
- PREFER PRECISE LANGUAGE
- CURRENT INTERACTION MODE + REGISTER REFERENCE

**Not** the compact `DAISY_PROMPT_MODE=aligned` overlay.

## Inference match

Serve v11 with:

```yaml
DAISY_PROMPT_MODE: "full"
DAISY_DIRECT_MULTILINGUAL: "true"
DAISY_BOOK_KNOWLEDGE: "false"
INFERENCE_BUILD: "2026-04-pre-june-restore"
```

Using `aligned` at inference caused dry generic replies (train/serve skew).

June 2026 inference (book RAG, multi-regen, canned `_OPEN_QUESTIONS` tails) caused worse UX than v11 weights alone — prod rolled back to **v11 + pre-June `inference/` tree** with only:

- Direct multilingual generation (`reply_language.py`)
- `final_sanitize_reply()` degenerate-output guard

## Verify job metadata

```powershell
az ml model show --name daisy-finetuned-lora --version 11 `
  -g Daisy_group -w Daisy --subscription <sub-id>

az ml job list -g Daisy_group -w Daisy --max-results 30 -o table
```

Check the completed job env for `TRAIN_FILE` / `VAL_FILE`.

## Verify live endpoint

```powershell
python scripts/verify_deploy.py
```

Includes RU finance-anxiety 4-turn thread from user screenshots.
