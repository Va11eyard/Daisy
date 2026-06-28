# GPU: training vs inference (Daisy-Model)

Consolidated guidance for **Azure** SKU choice. Verify availability in **your** region:

```powershell
az vm list-skus --location <region> --size Standard_NC4as_T4_v3 --resource-type virtualMachines -o table
```

More detail and sample output: [AZURE_TRAINING_AND_DEPLOY.md](AZURE_TRAINING_AND_DEPLOY.md) (section 9).

## Training (LoRA, default)

| Item | Recommendation |
|------|----------------|
| **VM size** | **`Standard_NC4as_T4_v3`** — 1× NVIDIA T4, 16 GB VRAM |
| **Job** | [azureml/command_job.yaml](../azureml/command_job.yaml) — `USE_LORA=true`, 8-bit base + LoRA in [training/train.py](../training/train.py) |
| **Speed** | Tune `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `MAX_SEQ_LENGTH`; smaller `NUM_EPOCHS` on small corpora |
| **Full fine-tune** | Requires **A100-class** (≈40–80 GB); not the default path |

## Inference (fast + cost-effective)

| Item | Recommendation |
|------|----------------|
| **VM size** | **`Standard_NC4as_T4_v3`** for a single 7B deployment |
| **Quantization** | **`INFERENCE_QUANTIZATION=4bit`** — see [inference/model_loader.py](../inference/model_loader.py) and [azureml/deployment.yaml](../azureml/deployment.yaml) |
| **Optional 3B router** | **`COORDINATOR_MODEL=Qwen/Qwen2.5-3B-Instruct`** — same quantization; fits on one T4 with 7B when both 4-bit |
| **Latency** | Lower `max_tokens` in clients; `max_concurrent_requests_per_instance: 1` until profiled; autoscale replicas for throughput (not necessarily a larger GPU) |
| **Step-up** | **24 GB** (A10 / L4 class) if fp16 7B+3B or very long KV |

## Quick reference

- Training and inference **SKU names are the same** (`Standard_NC4as_T4_v3`) for the default LoRA + 4-bit inference stack.
- **Quantization** saves VRAM, not the hourly VM price — staying on T4 saves money vs A10.
