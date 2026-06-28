# v13 training and deploy

## Dataset (shape-balanced, no book dumps)

```powershell
cd E:\WebstormProjects\Daisy-Model
$env:DAISY_PROMPT_MODE = "full"
python scripts/build_v12_ru_seed.py
python scripts/build_v13_shape_synth.py
python scripts/prepare_v13_dataset.py
python eval/analyze_offline.py --train data/train_v13.jsonl
```

Targets: `reflect_plus_question_shape_rate` <= 0.70, `book_dump_row_fraction` = 0.

Sources: `train_v2` + `train_v3` + RU seed + `v13_shape_synth.json`. **Not** `data/archive/train.jsonl.retired`.

## Train v13

```powershell
$env:TRAIN_FILE = "train_v13.jsonl"
$env:VAL_FILE = "val_v13.jsonl"
$env:OUTPUT_DIR = "./outputs/daisy-lora-v13"
$env:LORA_R = "16"
$env:NUM_EPOCHS = "3"
python scripts/submit_training_job.py --display-name daisy-lora-v13 `
  --subscription-id <sub> --resource-group Daisy_group --workspace-name Daisy
```

Submitted job: `keen_celery_g5pdmg2lch` (Completed)

Registered: `daisy-finetuned-lora:13`

## Register + deploy

```powershell
python scripts/register_model.py --job-name <completed-job> --name daisy-finetuned-lora --version 13
python scripts/merge_deployment_env.py azureml/deployment-v13-patch.yaml
az ml online-deployment update --name gpu-deployment-finetuned --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml -g Daisy_group -w Daisy
python scripts/verify_deploy.py
python scripts/ab_model_probe.py --label-a v12 --label-b v13 ...
```

`deployment-v13-patch.yaml` uses latency trims (`DAISY_MAX_REGENS=2`, `DAISY_RUBRIC_JUDGE=false`).
