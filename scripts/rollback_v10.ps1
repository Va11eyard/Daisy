# Roll back endpoint to daisy-finetuned-lora:10 + v8 inference settings
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python scripts/merge_deployment_env.py azureml/deployment-rollback-v10-patch.yaml
az ml online-deployment update `
  --name gpu-deployment-finetuned `
  --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml `
  --resource-group Daisy_group `
  --workspace-name Daisy `
  --subscription 9239bc75-105c-486e-8957-da8e49309c55

Write-Host "Rollback submitted (model v10). Warm-up ~10 min, then: python scripts/verify_deploy.py"
