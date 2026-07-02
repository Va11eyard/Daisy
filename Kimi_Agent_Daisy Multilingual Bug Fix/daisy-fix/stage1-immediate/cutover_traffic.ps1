#requires -Version 5.1
<#
.SYNOPSIS
    Daisy Therapy — Safe Azure ML Traffic Cutover Script
    Manages traffic routing between Daisy deployments on the daisy-therapy endpoint.

.DESCRIPTION
    Provides functions to view, adjust, and validate traffic splits across multiple
    deployments on the daisy-therapy Azure ML online endpoint.

    Safety features:
      - Refuses to set 0% on all deployments
      - Warns if traffic total != 100%
      - Requires -Confirm for changes > 50%
      - Logs all changes with timestamps

    Deployments:
      - gpu-deployment-finetuned  (production / control)
      - gpu-deployment-ru-translate (A/B — EN-generate + translate-to-RU)
      - gpu-deployment-v14        (Qwen3 candidate)

.PARAMETER Action
    Action to perform: show | cutover | abtest | rollback | health

.PARAMETER Deployment
    Target deployment name for cutover/rollback/health actions.

.PARAMETER Percent
    Target traffic percent (0-100) for cutover action.

.PARAMETER TestDeployment
    Challenger deployment for A/B test.

.PARAMETER ControlDeployment
    Control deployment for A/B test (default: gpu-deployment-finetuned).

.PARAMETER TestPercent
    Traffic percent for the challenger in A/B test (default: 10).

.PARAMETER Reason
    Reason for rollback (logged).

.PARAMETER Confirm
    Confirm large traffic changes (>50%).

.PARAMETER WhatIf
    Show what would happen without making changes.

.EXAMPLE
    # Show current traffic distribution
    .\cutover_traffic.ps1 -Action show

.EXAMPLE
    # Gradual cutover to Qwen3
    .\cutover_traffic.ps1 -Action cutover -Deployment gpu-deployment-v14 -Percent 10 -Confirm
    .\cutover_traffic.ps1 -Action cutover -Deployment gpu-deployment-v14 -Percent 50 -Confirm
    .\cutover_traffic.ps1 -Action cutover -Deployment gpu-deployment-v14 -Percent 100 -Confirm

.EXAMPLE
    # RU routing: 100% to translate deployment
    .\cutover_traffic.ps1 -Action cutover -Deployment gpu-deployment-ru-translate -Percent 100

.EXAMPLE
    # A/B test at 50/50
    .\cutover_traffic.ps1 -Action abtest -TestDeployment gpu-deployment-v14 -TestPercent 50

.EXAMPLE
    # Rollback to production
    .\cutover_traffic.ps1 -Action rollback -Deployment gpu-deployment-finetuned -Reason "quality regression"

.EXAMPLE
    # Health check a deployment
    .\cutover_traffic.ps1 -Action health -Deployment gpu-deployment-v14

.NOTES
    File: cutover_traffic.ps1
    Author: DevOps Team
    Requires: Az.Accounts, Az.MachineLearning modules OR Azure CLI (az ml)
    Endpoint: daisy-therapy
    Region: westus2
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("show", "cutover", "abtest", "rollback", "health")]
    [string]$Action = "show",

    [Parameter(Mandatory = $false)]
    [string]$Deployment,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0, 100)]
    [int]$Percent,

    [Parameter(Mandatory = $false)]
    [string]$TestDeployment,

    [Parameter(Mandatory = $false)]
    [string]$ControlDeployment = "gpu-deployment-finetuned",

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 99)]
    [int]$TestPercent = 10,

    [Parameter(Mandatory = $false)]
    [string]$Reason
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

$Script:SubscriptionId    = "9239bc75-105c-486e-8957-da8e49309c55"
$Script:ResourceGroup     = "Daisy_group"
$Script:Workspace         = "Daisy"
$Script:EndpointName      = "daisy-therapy"
$Script:Region            = "westus2"

$Script:LogDir            = Join-Path $PSScriptRoot "logs"
$Script:Timestamp         = Get-Date -Format "yyyyMMdd-HHmmss"
$Script:LogFile           = Join-Path $Script:LogDir "cutover_$($Script:Timestamp).log"

$Script:KnownDeployments  = @(
    "gpu-deployment-finetuned",
    "gpu-deployment-ru-translate",
    "gpu-deployment-v14"
)

$Script:HealthCheckProbe  = @{
    messages = @(@{ role = "user"; content = "I'm feeling anxious today" })
    locale   = "en"
    max_tokens = 10
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

function Initialize-Logging {
    if (!(Test-Path $Script:LogDir)) {
        New-Item -ItemType Directory -Path $Script:LogDir -Force | Out-Null
    }
}

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("INFO", "WARN", "ERROR", "SUCCESS")]
        [string]$Level,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line      = "[$timestamp] [$Level] $Message"

    # Console output with color
    switch ($Level) {
        "ERROR"   { Write-Host $line -ForegroundColor Red }
        "WARN"    { Write-Host $line -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $line -ForegroundColor Green }
        default   { Write-Host $line -ForegroundColor Cyan }
    }

    # File append (always UTF8, ensure newline)
    $line | Out-File -FilePath $Script:LogFile -Encoding utf8 -Append
}

# ---------------------------------------------------------------------------
# Azure Authentication
# ---------------------------------------------------------------------------

function Test-AzureConnection {
    Write-Log "INFO" "Checking Azure connection..."

    # Check if already logged in
    $account = $null
    try {
        $account = az account show --output json 2>$null | ConvertFrom-Json
    } catch {
        $account = $null
    }

    if ($account -and $account.id -eq $Script:SubscriptionId) {
        Write-Log "SUCCESS" "Already logged in to subscription $($Script:SubscriptionId)"
        return $true
    }

    Write-Host "`nNot logged in or wrong subscription. Initiating az login..." -ForegroundColor Yellow
    try {
        az login --output none
        az account set --subscription $Script:SubscriptionId
        Write-Log "SUCCESS" "Logged in and set subscription"
        return $true
    } catch {
        Write-Log "ERROR" "Failed to log in to Azure: $_"
        return $false
    }
}

# ---------------------------------------------------------------------------
# Helper: Build az ml command
# ---------------------------------------------------------------------------

function Build-AzMlCommand {
    param([string]$SubCommand)

    $cmd = "az ml online-endpoint $SubCommand " +
           "--resource-group $($Script:ResourceGroup) " +
           "--workspace-name $($Script:Workspace) " +
           "--name $($Script:EndpointName)"

    return $cmd
}

# ---------------------------------------------------------------------------
# Get-CurrentTraffic
# ---------------------------------------------------------------------------

function Get-CurrentTraffic {
    <#
    .SYNOPSIS
        Fetches and displays the current traffic split across deployments.
    #>
    [CmdletBinding()]
    param()

    Write-Log "INFO" "Fetching current traffic for endpoint '$($Script:EndpointName)'..."

    $cmd = Build-AzMlCommand -SubCommand "show"
    Write-Log "INFO" "Executing: $cmd"

    $result = $null
    try {
        $json = Invoke-Expression $cmd
        $result = $json | ConvertFrom-Json
    } catch {
        Write-Log "ERROR" "Failed to fetch endpoint info: $_"
        return $null
    }

    if (-not $result) {
        Write-Log "ERROR" "Empty response from Azure CLI"
        return $null
    }

    Write-Host "`n=== Current Traffic Split ===" -ForegroundColor Cyan
    Write-Host "Endpoint  : $($result.name)" -ForegroundColor White
    Write-Host "Status    : $($result.provisioning_state)" -ForegroundColor White
    Write-Host "Scoring URI: $($result.scoring_uri)" -ForegroundColor DarkGray
    Write-Host "----------------------------------------" -ForegroundColor DarkGray

    $traffic = $result.traffic
    if (-not $traffic) {
        Write-Log "WARN" "No traffic configuration found on endpoint"
        return $result
    }

    $total = 0
    $traffic.PSObject.Properties | ForEach-Object {
        $name   = $_.Name
        $pct    = $_.Value
        $total += $pct
        $bar    = "█" * [math]::Floor($pct / 2) + "░" * (50 - [math]::Floor($pct / 2))
        $color  = if ($pct -eq 100) { "Green" } elseif ($pct -gt 0) { "Cyan" } else { "DarkGray" }
        Write-Host ("{0,-30} {1,3}% {2}" -f $name, $pct, $bar) -ForegroundColor $color
    }

    Write-Host "----------------------------------------" -ForegroundColor DarkGray
    if ($total -ne 100) {
        Write-Host "TOTAL: $total% " -NoNewline -ForegroundColor Red
        Write-Host "(WARNING: does not sum to 100%)" -ForegroundColor Yellow
    } else {
        Write-Host "TOTAL: 100%" -ForegroundColor Green
    }
    Write-Host ""

    return $result
}

# ---------------------------------------------------------------------------
# Set-TrafficSplit
# ---------------------------------------------------------------------------

function Set-TrafficSplit {
    <#
    .SYNOPSIS
        Sets traffic distribution across one or more deployments.

    .PARAMETER DeploymentName
        Name of the deployment to receive traffic.

    .PARAMETER TrafficPercent
        Percentage (0-100) for the target deployment.

    .PARAMETER OtherDeployments
        Hashtable of @{DeploymentName = Percent} for remaining deployments.

    .PARAMETER Validate
        Validate the endpoint after update.
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeploymentName,

        [Parameter(Mandatory = $true)]
        [ValidateRange(0, 100)]
        [int]$TrafficPercent,

        [Parameter(Mandatory = $false)]
        [hashtable]$OtherDeployments = @{},

        [switch]$Validate
    )

    # Validate deployment name
    if ($DeploymentName -notin $Script:KnownDeployments) {
        Write-Log "WARN" "Deployment '$DeploymentName' not in known list: $($Script:KnownDeployments -join ', ')"
    }

    # Build traffic hash
    $trafficMap = @{ $DeploymentName = $TrafficPercent }
    foreach ($key in $OtherDeployments.Keys) {
        $trafficMap[$key] = $OtherDeployments[$key]
    }

    # Safety: refuse if all would be 0%
    $allZero = $true
    foreach ($val in $trafficMap.Values) {
        if ($val -gt 0) { $allZero = $false; break }
    }
    if ($allZero) {
        Write-Log "ERROR" "REFUSED: Cannot set 0% on all deployments. At least one must have traffic."
        return $false
    }

    # Safety: warn if total != 100%
    $total = ($trafficMap.Values | Measure-Object -Sum).Sum
    if ($total -ne 100) {
        Write-Log "WARN" "Traffic total is $total% (expected 100%). Azure ML will normalize, but review carefully."
    }

    # Safety: require -Confirm for >50% changes
    # Compare against current
    $current = Get-CurrentTraffic
    $currentPct = 0
    if ($current -and $current.traffic -and $current.traffic.$DeploymentName) {
        $currentPct = $current.traffic.$DeploymentName
    }
    $delta = [math]::Abs($TrafficPercent - $currentPct)
    if ($delta -gt 50 -and -not $PSCmdlet.ShouldProcess(
        "$DeploymentName = ${TrafficPercent}% (was ${currentPct}%, delta ${delta}%)",
        "LARGE traffic change ($delta% delta)"
    )) {
        Write-Log "INFO" "User declined large traffic change. No action taken."
        return $false
    }

    # Build --traffic argument
    $trafficArg = ($trafficMap.GetEnumerator() | ForEach-Object {
        "$($_.Key)=$($_.Value)"
    }) -join " "

    $cmd = Build-AzMlCommand -SubCommand "update" +
           " --traffic '$trafficArg'"

    Write-Log "INFO" "Setting traffic: $trafficArg"

    if ($WhatIfPreference) {
        Write-Log "INFO" "[WHATIF] Would execute: $cmd"
        return $true
    }

    try {
        $output = Invoke-Expression $cmd 2>&1
        Write-Log "SUCCESS" "Traffic updated successfully"
        if ($output) {
            Write-Log "INFO" "Output: $output"
        }
    } catch {
        Write-Log "ERROR" "Failed to update traffic: $_"
        return $false
    }

    # Log the change
    $logEntry = @{
        timestamp    = (Get-Date -Format "o")
        action       = "set_traffic"
        deployment   = $DeploymentName
        percent      = $TrafficPercent
        total_config = $trafficMap
        delta_from   = $currentPct
        user         = $env:USER
    } | ConvertTo-Json -Compress
    Write-Log "INFO" "LOG_ENTRY: $logEntry"

    # Validation
    if ($Validate) {
        Write-Log "INFO" "Validating endpoint after traffic change..."
        Start-Sleep -Seconds 5
        $updated = Get-CurrentTraffic
        $newPct = 0
        if ($updated -and $updated.traffic) {
            $newPct = $updated.traffic.$DeploymentName
        }
        if ($newPct -eq $TrafficPercent) {
            Write-Log "SUCCESS" "Validation passed: $DeploymentName = $newPct%"
        } else {
            Write-Log "WARN" "Validation mismatch: expected $TrafficPercent%, got $newPct%"
        }
    }

    return $true
}

# ---------------------------------------------------------------------------
# Start-ABTest
# ---------------------------------------------------------------------------

function Start-ABTest {
    <#
    .SYNOPSIS
        Configures an A/B test between a control and a test deployment.

    .PARAMETER ControlDeployment
        The stable/production deployment.

    .PARAMETER TestDeployment
        The challenger deployment.

    .PARAMETER TestPercent
        Traffic percent for the challenger (1-99). Control gets the rest.
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Mandatory = $false)]
        [string]$ControlDeployment = "gpu-deployment-finetuned",

        [Parameter(Mandatory = $true)]
        [string]$TestDeployment,

        [Parameter(Mandatory = $false)]
        [ValidateRange(1, 99)]
        [int]$TestPercent = 10
    )

    if ($ControlDeployment -eq $TestDeployment) {
        Write-Log "ERROR" "Control and test deployments must be different"
        return $false
    }

    $controlPercent = 100 - $TestPercent
    Write-Log "INFO" "Starting A/B test: $ControlDeployment=$controlPercent%, $TestDeployment=$TestPercent%"

    return Set-TrafficSplit `
        -DeploymentName $TestDeployment `
        -TrafficPercent $TestPercent `
        -OtherDeployments @{ $ControlDeployment = $controlPercent } `
        -Validate
}

# ---------------------------------------------------------------------------
# Invoke-Rollback
# ---------------------------------------------------------------------------

function Invoke-Rollback {
    <#
    .SYNOPSIS
        Instant rollback — routes 100% traffic to a single deployment.

    .PARAMETER DeploymentName
        Deployment to receive 100% traffic.

    .PARAMETER Reason
        Reason for rollback (logged).
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeploymentName,

        [Parameter(Mandatory = $false)]
        [string]$Reason
    )

    Write-Host "`n" -NoNewline
    Write-Log "WARN" "ROLLBACK INITIATED: Routing 100% to '$DeploymentName'"
    if ($Reason) {
        Write-Log "WARN" "Rollback reason: $Reason"
    }

    # Prompt for extra confirmation
    if (-not $PSCmdlet.ShouldProcess($DeploymentName, "ROLLBACK 100% traffic")) {
        Write-Log "INFO" "Rollback cancelled by user"
        return $false
    }

    $result = Set-TrafficSplit -DeploymentName $DeploymentName -TrafficPercent 100

    if ($result) {
        Write-Log "SUCCESS" "Rollback complete: 100% traffic to '$DeploymentName'"
        # Extra log entry for rollbacks
        $rollbackEntry = @{
            timestamp     = (Get-Date -Format "o")
            action        = "rollback"
            deployment    = $DeploymentName
            reason        = ($Reason -or "emergency")
            previous_state = (Get-CurrentTraffic | ForEach-Object { $_.traffic })
        } | ConvertTo-Json -Compress
        Write-Log "INFO" "ROLLBACK_LOG: $rollbackEntry"
    }

    return $result
}

# ---------------------------------------------------------------------------
# Test-DeploymentHealth
# ---------------------------------------------------------------------------

function Test-DeploymentHealth {
    <#
    .SYNOPSIS
        Performs a health check by sending a probe request to the endpoint,
        targeting a specific deployment via header.

    .PARAMETER DeploymentName
        Deployment to target (via azureml-model-deployment header).

    .PARAMETER TimeoutSec
        Timeout in seconds.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$DeploymentName,

        [Parameter(Mandatory = $false)]
        [int]$TimeoutSec = 30
    )

    Write-Log "INFO" "Health check: targeting deployment '$DeploymentName' (timeout=${TimeoutSec}s)"

    # Get endpoint key
    $cmd = Build-AzMlCommand -SubCommand "list-keys"
    $keys = $null
    try {
        $keysJson = Invoke-Expression $cmd
        $keys = $keysJson | ConvertFrom-Json
    } catch {
        Write-Log "ERROR" "Failed to fetch endpoint keys: $_"
        return $false
    }

    if (-not $keys -or -not $keys.primaryKey) {
        Write-Log "ERROR" "Could not retrieve primary key for endpoint"
        return $false
    }

    $uri = "https://$($Script:EndpointName).$($Script:Region).inference.ml.azure.com/score"
    $headers = @{
        "Content-Type"             = "application/json"
        "Authorization"            = "Bearer $($keys.primaryKey)"
        "azureml-model-deployment" = $DeploymentName
    }
    $body = $Script:HealthCheckProbe | ConvertTo-Json -Depth 5

    Write-Log "INFO" "Probe URI: $uri"
    Write-Log "INFO" "Probe body: $($body -replace '"content"\s*:\s*"[^"]*"', '"content":"..."')"

    $probeStart = Get-Date
    $response = $null
    try {
        $response = Invoke-WebRequest -Uri $uri `
            -Method Post `
            -Headers $headers `
            -Body $body `
            -ContentType "application/json" `
            -TimeoutSec $TimeoutSec `
            -UseBasicParsing `
            -ErrorAction Stop
    } catch {
        Write-Log "ERROR" "Health check FAILED for '$DeploymentName': $($_.Exception.Message)"
        return $false
    }

    $latency = ((Get-Date) - $probeStart).TotalMilliseconds
    $status  = $response.StatusCode
    $content = $response.Content

    Write-Log "INFO" "Response status: $status, Latency: $([math]::Round($latency,1))ms"

    # Validate response content
    $reply = ""
    try {
        $parsed = $content | ConvertFrom-Json
        if ($parsed.reply) { $reply = $parsed.reply }
        elseif ($Parsed.choices[0].message.content) { $reply = $parsed.choices[0].message.content }
        else { $reply = $content }
    } catch {
        $reply = $content
    }

    $checks = @{
        status_ok     = ($status -eq 200)
        has_content   = ($reply.Length -gt 0)
        min_length    = ($reply.Length -ge 10)
        no_error      = ($reply -notmatch "error|exception|traceback")
    }

    Write-Host "`n  Health Check Results for '$DeploymentName':" -ForegroundColor Cyan
    $allPassed = $true
    foreach ($check in $checks.GetEnumerator()) {
        $statusSymbol = if ($check.Value) { "PASS" } else { "FAIL" }
        $color        = if ($check.Value) { "Green" } else { "Red" }
        Write-Host ("    {0,-15} : {1}" -f $check.Key, $statusSymbol) -ForegroundColor $color
        if (-not $check.Value) { $allPassed = $false }
    }
    Write-Host "    Latency       : $([math]::Round($latency,1))ms" -ForegroundColor Cyan

    # Check for known failure patterns
    $failurePatterns = @(
        "Hey\s*[-—]\s*I'm glad you're here",
        "Assistant:\s*",
        "Question:\s*",
        "\.\s*,\s*\.\s*,",
        "Daisy noticed"
    )
    foreach ($pattern in $failurePatterns) {
        if ($reply -match $pattern) {
            Write-Log "WARN" "Detected banned pattern in response: '$pattern'"
            $allPassed = $false
        }
    }

    if ($allPassed) {
        Write-Log "SUCCESS" "Health check PASSED for '$DeploymentName'"
    } else {
        Write-Log "ERROR" "Health check FAILED for '$DeploymentName'"
    }

    # Log result
    $hcEntry = @{
        timestamp      = (Get-Date -Format "o")
        action         = "health_check"
        deployment     = $DeploymentName
        latency_ms     = [math]::Round($latency, 1)
        status_code    = $status
        content_length = $reply.Length
        checks         = $checks
        passed         = $allPassed
    } | ConvertTo-Json -Compress
    Write-Log "INFO" "HC_LOG: $hcEntry"

    return $allPassed
}

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

function Invoke-Main {
    param([string]$Action)

    Initialize-Logging
    Write-Log "INFO" "=== Daisy Traffic Cutover Script Started ==="
    Write-Log "INFO" "Action: $Action"

    # Check Azure CLI
    $azVersion = az version --output tsv 2>$null
    if (-not $azVersion) {
        Write-Log "ERROR" "Azure CLI (az) not found. Please install: https://aka.ms/installazurecli"
        exit 1
    }
    Write-Log "INFO" "Azure CLI version: $(az version --query '"azure-cli"' -o tsv)"

    # Auth
    if (-not (Test-AzureConnection)) {
        Write-Log "ERROR" "Azure authentication failed. Exiting."
        exit 1
    }

    # Route action
    switch ($Action.ToLower()) {
        "show" {
            Get-CurrentTraffic | Out-Null
        }

        "cutover" {
            if (-not $Deployment) {
                Write-Log "ERROR" "-Deployment is required for cutover action"
                exit 1
            }
            if ($null -eq $Percent) {
                Write-Log "ERROR" "-Percent is required for cutover action"
                exit 1
            }
            Write-Log "INFO" "Cutover: $Deployment -> $Percent%"

            # Default: others go to 0 (unless specified)
            $others = @{}
            foreach ($d in $Script:KnownDeployments) {
                if ($d -ne $Deployment) {
                    $others[$d] = 0
                }
            }
            Set-TrafficSplit -DeploymentName $Deployment -TrafficPercent $Percent -OtherDeployments $others -Validate
        }

        "abtest" {
            if (-not $TestDeployment) {
                Write-Log "ERROR" "-TestDeployment is required for abtest action"
                exit 1
            }
            Start-ABTest -ControlDeployment $ControlDeployment -TestDeployment $TestDeployment -TestPercent $TestPercent
        }

        "rollback" {
            if (-not $Deployment) {
                Write-Log "ERROR" "-Deployment is required for rollback action"
                exit 1
            }
            Invoke-Rollback -DeploymentName $Deployment -Reason $Reason
        }

        "health" {
            if (-not $Deployment) {
                Write-Log "ERROR" "-Deployment is required for health action"
                exit 1
            }
            $healthy = Test-DeploymentHealth -DeploymentName $Deployment
            exit ($healthy ? 0 : 1)
        }

        default {
            Write-Log "ERROR" "Unknown action: $Action. Use: show | cutover | abtest | rollback | health"
            exit 1
        }
    }

    Write-Log "INFO" "=== Script complete ==="
}

# Run
Invoke-Main -Action $Action
