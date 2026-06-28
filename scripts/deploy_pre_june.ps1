# Deploy v11 LoRA + pre-June inference restore
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python scripts/merge_deployment_env.py azureml/deployment-pre-june-patch.yaml
az ml online-deployment update `
  --name gpu-deployment-finetuned `
  --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml `
  --resource-group Daisy_group `
  --workspace-name Daisy `
  --subscription 9239bc75-105c-486e-8957-da8e49309c55

Write-Host "Pre-June restore submitted (model v11). Warm-up ~10 min, then: python scripts/verify_deploy.py"
