# Azure live state (verified)

Captured via `az ml online-endpoint show` and `az ml online-deployment list/show` on **2026-07-01**.

## Endpoint

| Field | Value |
|-------|--------|
| Name | `daisy-therapy` |
| URI | `https://daisy-therapy.westus2.inference.ml.azure.com/score` |
| Workspace | `Daisy` / RG `Daisy_group` / `westus2` |
| Subscription | `9239bc75-105c-486e-8957-da8e49309c55` |

## Traffic split

| Deployment | Traffic % |
|------------|-----------|
| **gpu-deployment-finetuned** | **100** |
| gpu-deployment-ru-translate | 0 |
| gpu-deployment-v14 | 0 |

## Production deployment (`gpu-deployment-finetuned`)

| Field | Value |
|-------|--------|
| Model | `daisy-finetuned-lora:11` |
| Base | `Qwen/Qwen2.5-7B-Instruct` (4bit) |
| INFERENCE_BUILD | `2026-07-lora-v11-natural4-ru` |
| DAISY_INFERENCE_MODE | `simple` |
| DAISY_PROMPT_MODE | `full` |
| DAISY_DIRECT_MULTILINGUAL | `true` |
| DAISY_RAG / DAISY_BM25 | `false` |
| DAISY_DEFAULT_MAX_TOKENS | `90` |
| DAISY_LORA_DEFAULT_TEMP | `0.55` |

## A/B deployments (0% traffic)

- **gpu-deployment-ru-translate** — LoRA v11, `DAISY_DIRECT_MULTILINGUAL=false`, build `2026-07-lora-v11-ru-translate`
- **gpu-deployment-v14** — Qwen3-8B base (no LoRA), build `2026-06-hollow-qc-gate-v11`

## Re-capture

```powershell
cd E:\WebstormProjects\Daisy-1
.\scripts\capture_azure_live_state.ps1
```

Machine-readable snapshot (no secrets): [`eval/results/azure_live_state.json`](../eval/results/azure_live_state.json).
