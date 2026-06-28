# Daisy-Model

Fine-tuning (LoRA) and Azure Machine Learning **inference** for the Daisy assistant. Training data uses `tokenizer.apply_chat_template` so the format matches inference.

## Layout

- `training/` — `train.py`, `conda.yaml` for GPU Command jobs.
- `inference/` — `score.py` (Azure ML `init`/`run`) runs a single linear pipeline with five anti-hallucination layers (crisis override → phase classification → RAG injection → one generation pass → confidence gate → voice validation). No separate coordinator model. Optional compact **`user_image`** (`inference/user_image.py`) in the JSON request. See the "Inference pipeline" section below.
- `data/` — curated `raw/` JSON + generated `train.jsonl` / `val.jsonl` (see [docs/DATASET.md](docs/DATASET.md)).
- `scripts/` — `prepare_dataset.py`, `dataset_prompts.py` (training system = inference `build_system_prompt`), `submit_training_job.py`, `register_model.py`.
- `azureml/` — `command_job.yaml`, `deployment.yaml` templates.
- `config/crisis_resources.yaml` — crisis hotline hints for scoring.
- `sample_data/` — tiny example for the data prep script.
- Docs: [docs/DATASET.md](docs/DATASET.md), [docs/GPU_TRAINING_AND_INFERENCE.md](docs/GPU_TRAINING_AND_INFERENCE.md), [docs/EVAL.md](docs/EVAL.md).

## Base model (default)

The repository defaults to **`Qwen/Qwen3-8B`** (Apache-2.0, commercial-safe): strong **multilingual** instruction following (Russian + English strong, Kazakh usable), 119-language coverage, and it fits a single **T4 16GB** in 4-bit. It replaces the previous `Qwen/Qwen2.5-7B-Instruct`.

Documented alternatives (set `BASE_MODEL` to switch):

| Model | License | When |
|-------|---------|------|
| `Qwen/Qwen3-8B` (default) | Apache-2.0 | Commercial-safe, best balance of RU/EN/KK on T4. |
| `Qwen/Qwen3-4B-Instruct` | Apache-2.0 | Latency fallback if 8B misses the <3s T4 target. |
| `inceptionai/Llama-3.1-Sherkala-8B-Chat` | CC-BY-NC-SA-4.0 (**non-commercial**) | Maximize Kazakh quality for research/non-commercial builds. |

Qwen3 runs in "thinking" mode by default; `score.py` calls `apply_chat_template(..., enable_thinking=False)` so no `<think>` blocks are emitted. Training and inference must use the **same** `BASE_MODEL`; LoRA (phase 2) data must be re-rendered with the chosen base's chat template. Override anytime: `set BASE_MODEL=org/model-name`.

## Inference pipeline (5 anti-hallucination layers)

`score.run()` executes one linear pass — no nested retry loops:

1. **Layer 5 — Crisis hard override** (`safety.crisis_tier`): fires first; crisis replies bypass all other layers.
2. **Layer 1 — Input classification** (`router.detect_phase` → `state_detector`): one deterministic phase decision (intake / disclosure / psychoeducation / action_planning / crisis). No second model call.
3. **Layer 2 — RAG context injection** (`rag.py`): embeds the user message (+ recent history), retrieves the top-3 exemplary assistant replies from the dialog corpus (FAISS, cosine), and injects them as a `[RETRIEVED CONTEXT]` block for tone/vocabulary grounding only.
4. **Single generation pass** (`generation.generate_reply`, logprobs captured).
5. **Layer 4 — Token confidence gate** (`confidence.py`): if mean token log-probability < `DAISY_CONFIDENCE_THRESHOLD` (default -2.5), skip straight to a safe fallback (no regeneration).
6. **Layer 3 — Structural validation** (`voice_qc.violates_voice_contract`): runs once; on violation, exactly one regeneration with `voice_regen_suffix`; if still violating, a fallback reply.

Build the RAG index before deploying: `python scripts/build_rag_index.py` (writes `rag_index.faiss` + `rag_meta.json` + `rag_vectors.npy` into `inference/knowledge/`). The corpus is `data/synthesized/*/batch_*.jsonl` (EN + RU; Kazakh queries fall back to RU/EN grounding). Set `DEBUG_MODE=true` to get a `debug_context.layer_trace` showing which layers fired and their outcomes.

**Streaming / SSE:** `generation.generate_reply_stream()` provides internal token streaming for early-stop/latency. Azure ML managed online endpoints return a single string from `run()`, so true client-side SSE requires different hosting (e.g. a custom container); the HTTP contract here stays request/response.

## LoRA vs full fine-tuning (`USE_LORA`)

- **`USE_LORA=true` (default)** — only adapter weights are trained (8-bit base + LoRA). Cheapest on a single **T4**; good specialization on your therapy data.
- **`USE_LORA=false`** — **all** pretrained weights are updated on your data (full fine-tuning). This is **not** “training from scratch”: the model still starts from Qwen’s pretrained checkpoint; your run **cannot** erase general knowledge or leave “only therapy + languages” in a strict sense — it **shifts** behavior toward your corpus. To literally train only from random init you would need a huge budget and dataset (not supported here).

Full fine-tuning **7B** requires **much more VRAM** than LoRA (often **40–80GB+** or multi-GPU depending on batch/sequence length). For a single small GPU, prefer LoRA or a **smaller** base model (e.g. 3B) with `USE_LORA=false`.

Output folders: default `./outputs/daisy-lora` vs `./outputs/daisy-full`. For deployment, **register the output folder** as usual. Inference loads a **full checkpoint** if it finds `config.json` + weight files under `AZUREML_MODEL_DIR` (see `inference/model_loader.py`); otherwise it loads `BASE_MODEL` + LoRA adapter.

Set before submit: `set USE_LORA=false` and tune `LEARNING_RATE` (default **2e-5** for full, **2e-4** for LoRA), `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`.

## Prerequisites

- Hugging Face token if your chosen `BASE_MODEL` is gated (Qwen3 weights are usually public; Sherkala requires accepting its license; still log in when the Hub requires it).
- Azure ML workspace, GPU compute cluster, and CLI (`az ml`) or Python SDK.

## Training on Azure ML

Step-by-step (env vars, `az ml job`, register model, deploy): **[docs/AZURE_PUSH_AND_TRAIN.md](docs/AZURE_PUSH_AND_TRAIN.md)**.

1. Build `train.jsonl` / `val.jsonl` (see `scripts/prepare_dataset.py`, [docs/DATASET.md](docs/DATASET.md), and `data/raw/daisy_curated.json`).
2. Keep them as **`data/train.jsonl`** and **`data/val.jsonl`** — **`submit_training_job.py` copies them into `training/`** before upload (or copy manually and use `azureml/command_job.yaml`).
3. Set `HF_TOKEN`, Azure ML env vars, optionally `BASE_MODEL` / hyperparameters, then:

```bash
set HF_TOKEN=your_token
set AZURE_SUBSCRIPTION_ID=...
set AZURE_RESOURCE_GROUP=...
set AZUREML_WORKSPACE_NAME=...
python scripts/submit_training_job.py
```

Or: `az ml job create --file azureml/command_job.yaml ...`

4. After the job completes, download outputs from Studio or `az ml job download`, then register:

```bash
python scripts/register_model.py --path ./path/to/daisy-lora --name daisy-finetuned-lora --version 1
```

## Inference deployment

Package layout expected at runtime: registered **custom model** artifact containing PEFT adapter files (`adapter_config.json`, weights) plus tokenizer files if needed. `score.py` loads `BASE_MODEL` from Hugging Face and merges the LoRA adapter from `AZUREML_MODEL_DIR`, **except** when `INFERENCE_QUANTIZATION` is `4bit` or `8bit` (adapter stays as PEFT on top of quantized weights — see `inference/model_loader.py`).

Edit `azureml/deployment-v14.yaml` (endpoint name, model reference, `instance_type`), set secrets in the deployment (not in git), then deploy with Azure ML CLI or Studio.

### Inference quantization (`INFERENCE_QUANTIZATION`)

Set in the deployment environment (default in `azureml/deployment.yaml` is **`4bit`**):

| Value | Meaning |
|-------|--------|
| `none` | fp16 weights on GPU (legacy default if unset). |
| `4bit` | NF4 + double quant via **bitsandbytes** — **recommended** for **Qwen3-8B** on a **single T4 16 GB**; quality loss is usually small vs fp16 for dialogue. |
| `8bit` | 8-bit weights — between fp16 and 4bit in VRAM. |

Requires `bitsandbytes` (listed in `inference/conda.yaml`).

### Which GPU to request (Daisy stack)

- **Default (single model, no coordinator):** **`Standard_NC4as_T4_v3`** (1× T4 16 GB) **+ `INFERENCE_QUANTIZATION=4bit`** runs Qwen3-8B with headroom. This is the confirmed live deployment SKU.
- **Tighter latency:** set `BASE_MODEL=Qwen/Qwen3-4B-Instruct` on the same T4 to comfortably hit the <3s target.
- **Heavier traffic:** same SKU with **more instances** — autoscaling (below), not a bigger GPU per replica.

### Scale on demand (Azure ML)

Managed online endpoints support **min/max instance count** and **autoscale** (CPU/GPU metrics or custom schedule). Typical pattern: **min replicas = 0** (pay less when idle; cold start) or **min = 1** (stable latency; higher baseline cost), **max replicas** tied to peak concurrency. Configure in Azure ML Studio for the endpoint or via CLI/ARM — not in `deployment.yaml` alone.

### Phase routing (coordinator removed)

The previous coordinator/router JSON-plan model is gone. Routing is now a single deterministic phase decision (`router.detect_phase` over `state_detector`), with no second forward pass — this halves `init()` time and removes the `ENABLE_ROUTER_PASS` / `COORDINATOR_*` env vars. The phase (intake / disclosure / psychoeducation / action_planning / crisis) drives both the system prompt and RAG retrieval. `debug_context.phase` and `debug_context.layer_trace` expose the decision for tracing.

### `user_image` (optional personalization)

The scoring endpoint accepts an optional JSON field **`user_image`**: a compact summary for fast personalization (aligned with Daisy web: `ai_profile`, psych indices, boundaries, memory highlights). Parsed by `inference/user_image.py` (`USER_IMAGE_SCHEMA_VERSION=1`).

Example shape:

```json
{
  "version": "1",
  "summary": "One or two sentences synthesizing DB + session context.",
  "goals": ["…"],
  "concerns": ["…"],
  "communication_style": ["warm_friend"],
  "risk_level": "medium",
  "indices": { "ESI": 45, "BSI": 62, "SSI": 55, "PVI": 30, "MRI": 48 },
  "memory_highlights": ["…"],
  "boundaries": { "avoid_topics": [], "sensitive": [] },
  "protocol_hint": "REGULATE (DBT + CBT light)"
}
```

`debug_context.has_user_image` is true when a non-empty normalized `user_image` was applied.

## Local checks

```bash
pip install pytest
pytest tests/
```

## API contract

See [docs/API_CONTRACT.md](docs/API_CONTRACT.md). For multi-agent expectations vs this repo and how the Daisy app routes traffic (Azure ML vs legacy CBT API), see [docs/MULTI_AGENT_AND_ROUTING.md](docs/MULTI_AGENT_AND_ROUTING.md).
