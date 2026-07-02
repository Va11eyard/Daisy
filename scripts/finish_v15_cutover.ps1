# Finish LoRA v15 pipeline after training job completes.
# Usage: .\scripts\finish_v15_cutover.ps1 -JobName purple_net_55lwkbc0zp

param(
    [string]$JobName = "purple_net_55lwkbc0zp",
    [string]$ResourceGroup = "Daisy_group",
    [string]$Workspace = "Daisy",
    [switch]$SkipCutover
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:AZURE_SUBSCRIPTION_ID = "9239bc75-105c-486e-8957-da8e49309c55"
$env:AZURE_RESOURCE_GROUP = $ResourceGroup
$env:AZUREML_WORKSPACE_NAME = $Workspace

Write-Host "Waiting for job $JobName..."
while ($true) {
    $status = (az ml job show --name $JobName -g $ResourceGroup -w $Workspace --query status -o tsv 2>$null)
    Write-Host "  status=$status"
    if ($status -eq "Completed") { break }
    if ($status -in @("Failed", "Canceled")) {
        throw "Training job $JobName ended with status $status"
    }
    Start-Sleep -Seconds 120
}

Write-Host "Registering daisy-finetuned-lora:15 from job $JobName..."
python scripts/register_model.py --job-name $JobName --name daisy-finetuned-lora --version 15

Write-Host "Deploying gpu-deployment-v14 (Qwen3 + LoRA v15)..."
python scripts/merge_deployment_env.py azureml/deployment-qwen3-lora-v15.yaml
az ml online-deployment update --name gpu-deployment-v14 --endpoint-name daisy-therapy --file azureml/.merged-deploy.yaml -g $ResourceGroup -w $Workspace

Write-Host "Waiting for v14 readiness (10 min)..."
Start-Sleep -Seconds 600

$key = (az ml online-endpoint get-credentials --name daisy-therapy -g $ResourceGroup -w $Workspace -o tsv --query primaryKey).Trim()
$env:DAISY_ENDPOINT_KEY = $key
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Running Qwen3 regression on gpu-deployment-v14..."
python scripts/run_cross_topic_regression.py --deployment gpu-deployment-v14 --delay 2.5 --concurrency 1 --output eval/results/qwen3_regression.json

Write-Host "Comparing baseline vs Qwen3..."
python scripts/compare_regression_reports.py eval/results/baseline_regression.json eval/results/qwen3_regression.json --format markdown | Tee-Object eval/results/regression_comparison.md

if (-not $SkipCutover) {
    $report = Get-Content eval/results/qwen3_regression.json | ConvertFrom-Json
    $passRate = $report.overall.pass_rate
    $scriptLeaks = $report.failure_breakdown.script_leak
  if ($passRate -ge 0.90 -and $scriptLeaks -eq 0) {
        Write-Host "Gates passed ($([math]::Round($passRate*100,1))%). Cutting over to v14..."
        .\scripts\cutover_traffic.ps1 -Action cutover -Deployment gpu-deployment-v14 -Percent 100 -Confirm
    } else {
        Write-Host "Gates NOT met (pass=$passRate script_leak=$scriptLeaks). Cutover skipped."
    }
}

Write-Host "Done."
