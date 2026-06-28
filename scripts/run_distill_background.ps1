# Start API distillation (JSONL append + resume). Requires ANTHROPIC_API_KEY or set OPENAI_API_KEY + --provider openai.
# Run from repo root:
#   .\scripts\run_distill_background.ps1
# Or pass key once:
#   .\scripts\run_distill_background.ps1 -AnthropicKey "sk-ant-..."

param(
    [string] $AnthropicKey = $env:ANTHROPIC_API_KEY
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $AnthropicKey) {
    Write-Host "Set ANTHROPIC_API_KEY or pass -AnthropicKey" -ForegroundColor Red
    exit 1
}
$env:ANTHROPIC_API_KEY = $AnthropicKey

$out = "data/raw/md_distilled.jsonl"
$log = "data/raw/md_distill.log"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

$arg = @(
    "scripts/md_distill_api.py"
    "--output", $out
    "--resume"
    "--sleep", "0.35"
)
$p = Start-Process -FilePath "python" -ArgumentList $arg -RedirectStandardOutput $log -RedirectStandardError $log -PassThru -NoNewWindow
Write-Host "Distillation PID $($p.Id); log: $log; output: $out"
Write-Host "Stop: Stop-Process -Id $($p.Id)"
