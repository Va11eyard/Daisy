# Run after v12 training job completes and LoRA is registered as daisy-finetuned-lora:12
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$job = $args[0]
if ($job) {
  Write-Host "Download artifacts from job $job if needed:"
  Write-Host "  az ml job download --name $job -g Daisy_group -w Daisy --download-path ./outputs/daisy-lora-v12"
}

python scripts/register_model.py --path ./outputs/daisy-lora-v12 --name daisy-finetuned-lora --version 12
python scripts/merge_deployment_env.py azureml/deployment-v12-patch.yaml
az ml online-deployment update `
  --name gpu-deployment-finetuned `
  --endpoint-name daisy-therapy `
  --file azureml/.merged-deploy.yaml `
  --resource-group Daisy_group `
  --workspace-name Daisy `
  --subscription 9239bc75-105c-486e-8957-da8e49309c55

python scripts/verify_deploy.py
Write-Host "Run A/B: python scripts/ab_model_probe.py --label-a v11 --label-b v12 ..."
