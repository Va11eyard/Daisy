# Capture Azure ML endpoint traffic + deployment env (secrets redacted).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$sub = "9239bc75-105c-486e-8957-da8e49309c55"
$rg = "Daisy_group"
$ws = "Daisy"
$ep = "daisy-therapy"

$endpoint = az ml online-endpoint show --name $ep -g $rg -w $ws --subscription $sub -o json | ConvertFrom-Json
$deps = az ml online-deployment list --endpoint-name $ep -g $rg -w $ws --subscription $sub -o json | ConvertFrom-Json

$secretKeys = @(
  "HF_TOKEN", "AZURE_OPENAI_KEY", "AZURE_TRANSLATOR_KEY",
  "AZURE_OPENAI_ENDPOINT", "AZURE_TRANSLATOR_ENDPOINT"
)

$deploymentDetails = @()
foreach ($d in $deps) {
  $full = az ml online-deployment show --name $d.name --endpoint-name $ep -g $rg -w $ws --subscription $sub -o json | ConvertFrom-Json
  $env = @{}
  if ($full.environment_variables) {
    $full.environment_variables.PSObject.Properties | ForEach-Object {
      if ($secretKeys -contains $_.Name) {
        $env[$_.Name] = "[REDACTED]"
      } else {
        $env[$_.Name] = $_.Value
      }
    }
  }
  $modelVer = $null
  if ($full.model -match "/versions/(\d+)") { $modelVer = [int]$Matches[1] }
  $deploymentDetails += [ordered]@{
    name = $full.name
    provisioning_state = $full.provisioning_state
    instance_type = $full.instance_type
    model_version = $modelVer
    environment_variables = $env
  }
}

$out = [ordered]@{
  captured_at = (Get-Date).ToUniversalTime().ToString("o")
  subscription_id = $sub
  resource_group = $rg
  workspace = $ws
  endpoint = [ordered]@{
    name = $endpoint.name
    scoring_uri = $endpoint.scoring_uri
    provisioning_state = $endpoint.provisioning_state
    traffic_percent = $endpoint.traffic
  }
  deployments = $deploymentDetails
}

$path = "eval/results/azure_live_state.json"
$out | ConvertTo-Json -Depth 8 | Set-Content -Path $path -Encoding utf8
Write-Host "Wrote $path"
