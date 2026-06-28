# Run after keen_celery_g5pdmg2lch (or latest v13 job) completes
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$job = $args[0]
if (-not $job) { $job = "keen_celery_g5pdmg2lch" }

$env:AZURE_SUBSCRIPTION_ID = "9239bc75-105c-486e-8957-da8e49309c55"
$env:AZURE_RESOURCE_GROUP = "Daisy_group"
$env:AZUREML_WORKSPACE_NAME = "Daisy"

$st = az ml job show --name $job -g Daisy_group -w Daisy --query status -o tsv
if ($st -ne "Completed") {
  Write-Error "Job $job status=$st (expected Completed)"
}

python scripts/register_model.py --job-name $job --name daisy-finetuned-lora --version 13 --description "Daisy LoRA v13 shape-balanced train_v13"
python scripts/merge_deployment_env.py azureml/deployment-v13-patch.yaml
az ml online-deployment update `
  --name gpu-deployment-finetuned `
  --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml `
  --resource-group Daisy_group `
  --workspace-name Daisy `
  --subscription 9239bc75-105c-486e-8957-da8e49309c55

python scripts/verify_deploy.py
python scripts/eval_v13_gates.py
Write-Host "A/B optional: python scripts/ab_model_probe.py (same endpoint before/after snapshot)"
