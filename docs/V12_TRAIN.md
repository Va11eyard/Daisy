# v12 training and deploy

## Root cause fix

v11 was trained on `train_v3.jsonl` with **full** voice-contract prompt. Serve with `DAISY_PROMPT_MODE=full`, not `aligned`.

## 1. Build dataset (good base only)

```powershell
cd E:\WebstormProjects\Daisy-Model
$env:DAISY_PROMPT_MODE = "full"
python scripts/build_v12_ru_seed.py
python scripts/prepare_v12_dataset.py
```

Sources: `train_v2.jsonl` + `train_v3.jsonl` + RU seed (10 scenarios). **Not** `train.jsonl` book-dumps.

Optional RU enrichment:

```powershell
python scripts/md_distill_api.py --md-root <path-to-Rus-md> --output data/raw/md_distilled_ru.jsonl
python scripts/prepare_v12_dataset.py --ru-distilled data/raw/md_distilled_ru.jsonl
```

## 2. Train v12

```powershell
$env:TRAIN_FILE = "train_v12.jsonl"
$env:VAL_FILE = "val_v12.jsonl"
$env:OUTPUT_DIR = "./outputs/daisy-lora-v12"
$env:LORA_R = "16"
$env:NUM_EPOCHS = "3"
python scripts/submit_training_job.py --display-name daisy-lora-v12 `
  --subscription-id <sub> --resource-group Daisy_group --workspace-name Daisy
```

Latest submitted job: `teal_square_jp2nqmk59z`

## 3. Register + deploy v12

v12 registered as `daisy-finetuned-lora:12` from job `teal_square_jp2nqmk59z`.

```powershell
python scripts/register_model.py --path ./outputs/daisy-lora-v12 --name daisy-finetuned-lora --version 12
python scripts/merge_deployment_env.py azureml/deployment-v12-patch.yaml
az ml online-deployment update --name gpu-deployment-finetuned --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml -g Daisy_group -w Daisy
python scripts/verify_deploy.py
python scripts/ab_model_probe.py --label-a v11 --label-b v12 ...
```

`deployment-v12-patch.yaml` uses `DAISY_PROMPT_MODE=full`.
