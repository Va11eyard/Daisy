# v12 training pipeline (run from repo root)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:DAISY_PROMPT_MODE = "full"

Write-Host "Building RU seed + train_v12.jsonl / val_v12.jsonl..."
python scripts/build_v12_ru_seed.py
python scripts/prepare_v12_dataset.py

$env:TRAIN_FILE = "train_v12.jsonl"
$env:VAL_FILE = "val_v12.jsonl"
$env:OUTPUT_DIR = "./outputs/daisy-lora-v12"
$env:USE_LORA = "true"
$env:LORA_R = "16"
$env:NUM_EPOCHS = "3"

Write-Host "Submitting Azure ML training job..."
python scripts/submit_training_job.py --display-name daisy-lora-v12

Write-Host @"

After job completes:
  python scripts/register_model.py --path ./outputs/daisy-lora-v12 --name daisy-finetuned-lora --version 12
  python scripts/merge_deployment_env.py azureml/deployment-v12-patch.yaml
  az ml online-deployment update --name gpu-deployment-finetuned --endpoint-name daisy-therapy `
    --file azureml/.merged-deploy.yaml -g Daisy_group -w Daisy
  python scripts/verify_deploy.py
"@
