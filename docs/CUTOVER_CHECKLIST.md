# Daisy Therapy Production Cutover Checklist

> **Version:** 1.0  
> **Date:** 2026-07-01  
> **Endpoint:** `daisy-therapy` (westus2)  
> **Subscription:** `9239bc75-105c-486e-8957-da8e49309c55`  
> **RG/Workspace:** `Daisy_group` / `Daisy`  
> **Primary Deployment:** `gpu-deployment-finetuned`  
> **Candidate Deployment:** `gpu-deployment-v14` (Qwen3-8B)  

---

## Integration status (2026-07-01)

| Stage | Artifact | Status |
|-------|----------|--------|
| Baseline regression | `eval/results/baseline_regression.json` | **Done** — 16/56 (28.6%); 0 script/structural leaks |
| train_v15 | `data/train_v15.jsonl` | **Done** — 2675 rows, 48% RU, 0 Latin leaks |
| RU translate v2 | `gpu-deployment-ru-translate` | **Done** — `INFERENCE_BUILD=2026-07-lora-v11-ru-translate-v2`; RU +10.7pp |
| LoRA v15 training | `purple_net_55lwkbc0zp` | **Done** — registered `daisy-finetuned-lora:15` |
| Qwen3 v14 deploy | `gpu-deployment-v14` | **Done** — `score_qwen3_aml.py`, Qwen3-8B + LoRA v15 |
| Qwen3 regression | `eval/results/qwen3_regression.json` | **Done** — 32/56 (57.1%); 0 script/structural leaks |
| Cutover | — | **Blocked** — gates require ≥90% overall, ≥85% per cluster (work/clarity/stress/anxiety <85%) |

**Baseline snapshot:** EN 39.3%, RU 17.9%; top failure `keyword_mismatch` (40). No `script_leak` on prod finetuned slot.

**Qwen3 v14 (post-fix):** EN 50.0%, RU 64.3%; +28.5pp overall vs baseline. **Do not cut over** until ≥90% / per-cluster ≥85%. Short-term: route RU via `AML_DEPLOYMENT_NAME_RU=gpu-deployment-ru-translate` (+10.7pp RU vs baseline).

---

## Table of Contents

1. [Pre-Cutover Verification](#1-pre-cutover-verification)
2. [Deployment Steps](#2-deployment-steps)
3. [Rollback Triggers](#3-rollback-triggers)
4. [Rollback Procedure](#4-rollback-procedure)
5. [Post-Cutover Validation](#5-post-cutover-validation)
6. [Short-Term RU Routing (Option B)](#6-short-term-ru-routing-option-b)

---

## 1. Pre-Cutover Verification

**All items below must be checked ( ✅ ) before any traffic is routed to the new deployment.**

If any item fails, **do not proceed**. Fix the issue and re-run verification.

### 1.1 Regression Quality Gates

| # | Check | Command / Method | Gate | Status |
|---|-------|-----------------|------|--------|
| 1 | **Overall pass rate >= 90%** | `python scripts/run_cross_topic_regression.py --deployment gpu-deployment-v14 --output eval/results/qwen3_regression.json` | >= 90% | [ ] |
| 2 | **Per-cluster pass rate >= 85%** | Inspect `v14_report.json` `by_cluster` | All 8 clusters >= 85% | [ ] |
| 3 | **Zero structural_leak failures** | `cat eval/v14_report.json \| jq '.failure_breakdown.structural_leak'` | == 0 | [ ] |
| 4 | **Zero script_leak failures** | `cat eval/v14_report.json \| jq '.failure_breakdown.script_leak'` | == 0 | [ ] |
| 5 | **Zero canned_greeting failures** | `cat eval/v14_report.json \| jq '.failure_breakdown.canned_greeting'` | == 0 | [ ] |
| 6 | **RU: informal `ty` usage** | Manually inspect RU cases in report | No formal `vy` detected | [ ] |
| 7 | **RU: no EN/PL/DE/ES mid-sentence** | `cat eval/v14_report.json \| jq '.cases[] \| select(.locale=="ru" and .passed==false) \| {id, failure_reasons}'` | No `script_leak` on RU | [ ] |
| 8 | **EN: no identical template on paraphrased inputs** | Manually compare `reply_preview` for paraphrased EN cases | Templates vary | [ ] |
| 9 | **P50 latency < 15s on T4** | `cat eval/v14_report.json \| jq '[.cases[].latency_ms] \| sort \| .[(length/2\|floor)]'` | < 15000 ms | [ ] |
| 10 | **All 8 clusters present and passing** | `cat eval/v14_report.json \| jq 'keys[] as $k \| .by_cluster \| keys'` | breakup, work, anxiety, stress, grief, clarity, somatic | [ ] |
| 11 | **Error messages locale-aware (EN/RU/KK)** | Send probe with `locale: ru` and `locale: kk`, verify response text | RU→Russian, KK→Kazakh | [ ] |

### 1.2 Before/After Comparison

Run the comparison tool to validate improvement over baseline:

```bash
# Baseline (recorded 2026-07-01)
python scripts/run_cross_topic_regression.py \
    --deployment gpu-deployment-finetuned \
    --output eval/results/baseline_regression.json

# Candidate
python scripts/run_cross_topic_regression.py \
    --deployment gpu-deployment-v14 \
    --output eval/results/qwen3_regression.json

# Compare
python scripts/compare_regression_reports.py \
    eval/results/baseline_regression.json \
    eval/results/qwen3_regression.json \
    --format markdown
```

**The `release_gate.can_release` field in `delta_report.json` must be `true`.** If `false`, do not proceed.

### 1.3 Azure Infrastructure Checks

| # | Check | Azure CLI Command | Status |
|---|-------|------------------|--------|
| 1 | Candidate deployment is healthy | `az ml online-deployment show --name gpu-deployment-v14 --endpoint-name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query '{name:name, provisioning_state:provisioning_state, scoring_uri:scoring_uri}'` | [ ] |
| 2 | Endpoint scoring URI is accessible | `curl -s -o /dev/null -w "%{http_code}" https://daisy-therapy.westus2.inference.ml.azure.com/score` | 200 | [ ] |
| 3 | Endpoint key is valid | `az ml online-endpoint list-keys --name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query primaryKey` | Key retrieved | [ ] |
| 4 | No ongoing Azure incidents in westus2 | `az account list-locations --query "[?name=='westus2'].{name:name, regional_display_name:regionalDisplayName}"` | No incidents | [ ] |
| 5 | Log Analytics workspace is receiving data | Azure Portal > Daisy workspace > Logs > run `traces \| take 5` | Data flowing | [ ] |

### 1.4 Sign-off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| QA Lead | _______________ | _______________ | _____ |
| ML Engineer | _______________ | _______________ | _____ |
| DevOps | _______________ | _______________ | _____ |
| Product Owner | _______________ | _______________ | _____ |

---

## 2. Deployment Steps

Execute in strict order. Do not skip steps. If a step fails, halt and assess before continuing.

### Step 1: Deploy `gpu-deployment-v14` at 0% Traffic

Deploy the Qwen3 candidate with zero live traffic to verify it is online and responsive.

```bash
# Verify deployment exists and is in "Succeeded" state
az ml online-deployment show \
    --name gpu-deployment-v14 \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query "{name:name, provisioning_state:provisioning_state, traffic_percentage:traffic}"
```

Expected output: `provisioning_state: "Succeeded"`, `traffic_percentage: 0`

If the deployment does not exist yet, create it:

```bash
az ml online-deployment create \
    --file stage2-qwen3-migration/deployment-qwen3-lora-v15.yaml \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --endpoint-name daisy-therapy \
    --all-traffic false
```

**Wait for provisioning state = "Succeeded" before proceeding.**

```bash
# Poll until ready (timeout: 30 minutes)
for i in $(seq 1 60); do
    STATE=$(az ml online-deployment show \
        --name gpu-deployment-v14 \
        --endpoint-name daisy-therapy \
        --resource-group Daisy_group \
        --workspace-name Daisy \
        --query provisioning_state -o tsv)
    echo "[$i/60] State: $STATE"
    if [ "$STATE" = "Succeeded" ]; then break; fi
    sleep 30
done
```

### Step 2: Run Regression Against `v14` via `--deployment` Header

Run the full 56-case regression against `gpu-deployment-v14` while it is at 0% traffic (routed via header).

```bash
cd stage1-immediate

export DAISY_ENDPOINT_KEY=$(az ml online-endpoint list-keys \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query primaryKey -o tsv)

python run_cross_topic_regression.py \
    --deployment gpu-deployment-v14 \
    --output eval/v14_step2_report.json \
    --delay 1.0
```

**Validation criteria:**

```bash
# Check overall pass rate
python -c "
import json
with open('eval/v14_step2_report.json') as f:
    r = json.load(f)
print(f\"Overall: {r['overall']['passed']}/{r['overall']['total']} ({r['overall']['pass_rate']*100:.1f}%)\")
assert r['overall']['pass_rate'] >= 0.90, 'FAIL: overall < 90%'
print('PASS: overall >= 90%')
"
```

If regression fails, **do not proceed to Step 3**. Debug `v14` offline.

### Step 3: A/B Test at 5% Traffic for 24 Hours

Route 5% of production traffic to `gpu-deployment-v14`, keeping 95% on `gpu-deployment-finetuned`.

```powershell
# Using the cutover script
.\cutover_traffic.ps1 -Action abtest `
    -ControlDeployment gpu-deployment-finetuned `
    -TestDeployment gpu-deployment-v14 `
    -TestPercent 5
```

Or via Azure CLI:

```bash
az ml online-endpoint update \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --traffic "gpu-deployment-finetuned=95 gpu-deployment-v14=5"
```

Verify traffic split:

```bash
az ml online-endpoint show \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query traffic
```

Expected: `{"gpu-deployment-finetuned": 95, "gpu-deployment-v14": 5}`

**Duration:** 24 hours minimum. Monitor continuously during this window.

### Step 4: Monitor Latency, Errors, and User Feedback

During the 5% A/B test window, monitor the following dashboards and metrics:

| Metric | Source | Threshold | Action if Breached |
|--------|--------|-----------|-------------------|
| P50 Latency | Azure ML > Endpoint Metrics | < 15s | Alert; proceed only if < 20s |
| P95 Latency | Azure ML > Endpoint Metrics | < 30s | Alert; rollback if > 30s for > 10 min |
| Error Rate (5xx) | Azure ML > Endpoint Metrics | < 2% | Alert; rollback if > 5% for > 5 min |
| Request Volume | Azure ML > Endpoint Metrics | Normal | Confirm both deployments receiving traffic |
| User Feedback | TalkToDaisy feedback DB | No spike | Monitor sentiment; flag complaints |
| RU Script Leaks | Custom Log Analytics query | 0 | Rollback if any detected |

**Azure CLI monitoring commands:**

```bash
# List endpoint metrics (via Azure Monitor)
az monitor metrics list \
    --resource $(az ml online-endpoint show --name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query id -o tsv) \
    --metric request_latency_p50,request_latency_p95,errors_total,requests_total \
    --interval PT5M

# Check deployment-specific logs
az ml online-deployment get-logs \
    --name gpu-deployment-v14 \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --lines 100
```

**Log Analytics query for RU script leak detection:**

```kusto
let ScriptLeakPattern = @"[a-zA-Z]{3,}";
traces
| where timestamp > ago(1h)
| where customDimensions.locale == "ru"
| where message contains_cs "script_leak" or message matches regex ScriptLeakPattern
| summarize count() by bin(timestamp, 5m), deployment=customDimensions.deployment
| where count_ > 0
```

### Step 5: Increase to 25% if Metrics Are Good

After 24 hours at 5%, if all metrics are green:

```powershell
.\cutover_traffic.ps1 -Action abtest `
    -ControlDeployment gpu-deployment-finetuned `
    -TestDeployment gpu-deployment-v14 `
    -TestPercent 25
```

Or via Azure CLI:

```bash
az ml online-endpoint update \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --traffic "gpu-deployment-finetuned=75 gpu-deployment-v14=25"
```

**Hold at 25% for minimum 12 hours.** Continue monitoring from Step 4.

### Step 6: Increase to 50%

After 12 hours at 25%, if all metrics are green:

```powershell
.\cutover_traffic.ps1 -Action abtest `
    -ControlDeployment gpu-deployment-finetuned `
    -TestDeployment gpu-deployment-v14 `
    -TestPercent 50
```

Or via Azure CLI:

```bash
az ml online-endpoint update \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --traffic "gpu-deployment-finetuned=50 gpu-deployment-v14=50"
```

**Hold at 50% for minimum 12 hours.** This is the highest-risk phase — both deployments serve equal traffic.

### Step 7: Full Cutover to 100%

After 12 hours at 50%, if all metrics are green:

```powershell
.\cutover_traffic.ps1 -Action cutover `
    -Deployment gpu-deployment-v14 `
    -Percent 100
```

Or via Azure CLI:

```bash
az ml online-endpoint update \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --traffic "gpu-deployment-v14=100 gpu-deployment-finetuned=0"
```

**Immediate post-cutover verification (within 5 minutes):**

```bash
# Verify traffic split
az ml online-endpoint show \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query traffic

# Run health check against v14
.\stage1-immediate\cutover_traffic.ps1 -Action health -Deployment gpu-deployment-v14

# Run 8-case smoke test
python stage1-immediate/run_cross_topic_regression.py \
    --deployment gpu-deployment-v14 \
    --limit 8 \
    --output eval/smoke_test_100pct.json
```

### Step 8: Retire `gpu-deployment-finetuned` After 48 Hours Stable

Keep `gpu-deployment-finetuned` at 0% traffic for 48 hours as a hot-standby. After 48 hours of stable operation:

```bash
# Step 8a: Confirm 48h stability
# (Check monitoring dashboards, verify no issues)

# Step 8b: Delete the old deployment
az ml online-deployment delete \
    --name gpu-deployment-finetuned \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --yes

# Step 8c: Verify only v14 remains
az ml online-endpoint show \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query "{name:name, traffic:traffic}"
```

**Before deleting, capture final snapshot of old deployment for rollback safety:**

```bash
# Save deployment config
az ml online-deployment show \
    --name gpu-deployment-finetuned \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query "{name:name, model:model, environment_variables:environment_variables}" \
    > eval/finetuned_deployment_snapshot.json
```

---

## 3. Rollback Triggers

**ANY of the following conditions triggers an instant rollback to `gpu-deployment-finetuned`.** No approval needed — rollback first, investigate second.

| # | Trigger | Threshold | Detection Method | Severity |
|---|---------|-----------|-----------------|----------|
| 1 | **Overall regression pass rate drops below 85%** | < 85% | `run_cross_topic_regression.py` scheduled run | CRITICAL |
| 2 | **Any structural/script leak in production traffic** | > 0 | Log Analytics query on `traces` table | CRITICAL |
| 3 | **P50 latency exceeds 30s for more than 10 minutes** | > 30s sustained | Azure Monitor `request_latency_p50` | CRITICAL |
| 4 | **Error rate exceeds 5% for more than 5 minutes** | > 5% sustained | Azure Monitor `errors_total / requests_total` | CRITICAL |
| 5 | **User complaints of wrong language** | Any confirmed | Support ticket / feedback DB | HIGH |
| 6 | **Any cluster pass rate drops below 75%** | < 75% | Regression report `by_cluster` | HIGH |
| 7 | **RU responses revert to formal `vy`** | > 2 cases | Manual QA spot-check | MEDIUM |
| 8 | **Endpoint returns 503/504 errors** | Any | Health check probe | CRITICAL |

**Escalation contacts:**

| Role | Contact | Phone | Slack |
|------|---------|-------|-------|
| On-call Engineer | {{ONCALL_ENGINEER}} | {{ONCALL_PHONE}} | #daisy-alerts |
| ML Lead | {{ML_LEAD}} | {{ML_LEAD_PHONE}} | @{{ML_LEAD_HANDLE}} |
| DevOps Lead | {{DEVOPS_LEAD}} | {{DEVOPS_PHONE}} | @{{DEVOPS_HANDLE}} |
| Product Owner | {{PO}} | {{PO_PHONE}} | @{{PO_HANDLE}} |

---

## 4. Rollback Procedure

Execute these steps in order. Target rollback time: **< 5 minutes**.

### Step 1: Execute Rollback Script

```powershell
# Instant rollback to gpu-deployment-finetuned
.\stage1-immediate\cutover_traffic.ps1 -Action rollback `
    -Deployment gpu-deployment-finetuned `
    -Reason "{{INSERT_TRIGGER_REASON_HERE}}"
```

Or via Azure CLI (if PowerShell is unavailable):

```bash
az ml online-endpoint update \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --traffic "gpu-deployment-finetuned=100 gpu-deployment-v14=0"
```

**Expected response time:** < 30 seconds for traffic routing to take effect.

### Step 2: Verify Traffic Split

```bash
# Confirm 100% to gpu-deployment-finetuned
az ml online-endpoint show \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query traffic
```

Expected output: `{"gpu-deployment-finetuned": 100, "gpu-deployment-v14": 0}`

If the split is not 100/0, retry Step 1. If still failing, escalate to DevOps Lead immediately.

### Step 3: Run Smoke Test (8 Cases)

```bash
export DAISY_ENDPOINT_KEY=$(az ml online-endpoint list-keys \
    --name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query primaryKey -o tsv)

python stage1-immediate/run_cross_topic_regression.py \
    --limit 8 \
    --output eval/rollback_smoke_test.json
```

**Smoke test must achieve >= 75% pass rate (6/8 cases).** If it fails, there may be a broader issue. Escalate immediately.

```bash
# Quick check
python -c "
import json
with open('eval/rollback_smoke_test.json') as f:
    r = json.load(f)
rate = r['overall']['pass_rate']
print(f'Smoke test: {rate*100:.0f}% ({r[\"overall\"][\"passed\"]} passed)')
assert rate >= 0.75, 'CRITICAL: Smoke test failed after rollback!'
print('Smoke test PASSED')
"
```

### Step 4: Notify Team

Post in `#daisy-alerts` Slack channel:

```
:rotating_light: ROLLBACK EXECUTED :rotating_light:

Deployment: gpu-deployment-v14 -> gpu-deployment-finetuned
Reason: {{INSERT_TRIGGER_REASON}}
Triggered by: @{{YOUR_HANDLE}}
Time: {{TIMESTAMP}}

Traffic split: 100% gpu-deployment-finetuned
Smoke test: {{PASS/FAIL}}

Impact: Users are now on the stable deployment.
Next: Investigating v14 issue offline.
```

Create a P1 incident ticket with:
- Trigger condition that fired
- Time of rollback
- Any relevant logs/metrics screenshots
- Link to the v14 deployment logs

### Step 5: Debug `v14` Offline

`gpu-deployment-v14` is now at 0% traffic but remains deployed. Debug without affecting users.

```bash
# Pull logs from v14 (last 500 lines)
az ml online-deployment get-logs \
    --name gpu-deployment-v14 \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --lines 500 > eval/v14_rollback_logs.txt

# Run full regression against v14 via header (does not affect live traffic)
python stage1-immediate/run_cross_topic_regression.py \
    --deployment gpu-deployment-v14 \
    --output eval/v14_debug_report.json

# Compare with baseline
python stage3-verify/compare_regression_reports.py \
    eval/baseline_report.json \
    eval/v14_debug_report.json \
    --output eval/rollback_delta.md \
    --format markdown
```

**Do not re-route traffic to v14 until the root cause is identified, fixed, and the full regression passes again.**

---

## 5. Post-Cutover Validation (24h After 100%)

After `gpu-deployment-v14` has been serving 100% of traffic for 24 hours, perform these checks:

### 5.1 Automated Checks

| # | Check | Command / Query | Pass Criteria | Status |
|---|-------|----------------|---------------|--------|
| 1 | **User-facing metrics stable** | Compare 24h before/after in Log Analytics | No significant drop in session length or message count | [ ] |
| 2 | **No increase in error rate** | `az monitor metrics list --metrics errors_total --interval PT1H` | Error rate <= pre-cutover baseline + 1pp | [ ] |
| 3 | **Latency P50 < 15s sustained** | `az monitor metrics list --metrics request_latency_p50` | P50 < 15000 ms for 95% of 5-min windows | [ ] |
| 4 | **No regression complaints** | Feedback DB + support tickets | Zero complaints about quality/language/template | [ ] |
| 5 | **RU quality maintained** | Run regression on RU cases only | RU pass rate >= pre-cutover + 0pp | [ ] |
| 6 | **KK quality maintained** | Run regression on KK cases only | KK pass rate >= pre-cutover + 0pp | [ ] |

### 5.2 Log Analytics Queries

```kusto
// 24h error rate comparison
let Before = toscalar(
    traces
    | where timestamp between (ago(48h) .. ago(24h))
    | where message contains "error" or customDimensions.status_code >= 500
    | count
);
let After = toscalar(
    traces
    | where timestamp > ago(24h)
    | where message contains "error" or customDimensions.status_code >= 500
    | count
);
print BeforeErrors=Before, AfterErrors=After, Ratio=todouble(After)/todouble(Before)
```

```kusto
// Latency distribution over 24h
requests
| where timestamp > ago(24h)
| summarize 
    P50=percentile(duration, 50),
    P95=percentile(duration, 95),
    P99=percentile(duration, 99),
    Count=count()
    by bin(timestamp, 1h)
| order by timestamp desc
```

### 5.3 Sign-off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| QA Lead | _______________ | _______________ | _____ |
| DevOps | _______________ | _______________ | _____ |
| Product Owner | _______________ | _______________ | _____ |

---

## 6. Short-Term RU Routing (Option B)

If the Qwen3 `gpu-deployment-v14` is not ready for full RU support, execute this parallel track for Russian language stabilization using the translate pipeline.

### 6.1 Deploy `gpu-deployment-ru-translate-v2` at 10% RU Traffic

```bash
# Ensure the deployment exists and is healthy
az ml online-deployment show \
    --name gpu-deployment-ru-translate \
    --endpoint-name daisy-therapy \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --query "{name:name, provisioning_state:provisioning_state}"
```

If it needs to be updated to v2:

```bash
az ml online-deployment update \
    --file stage1-immediate/deployment-lora-v11-ru-translate-v2.yaml \
    --resource-group Daisy_group \
    --workspace-name Daisy \
    --endpoint-name daisy-therapy
```

The frontend (`route.ts`) handles RU routing automatically via the `azureml-model-deployment` header when `locale == "ru"`. No traffic split changes are needed on the endpoint itself.

### 6.2 Monitor RU Quality Specifically

Run RU-specific regression twice daily during the evaluation window:

```bash
# Filter RU cases from the full regression
python -c "
import json
with open('eval/cross_topic_regression.jsonl') as f:
    cases = [json.loads(line) for line in f if line.strip()]
ru_cases = [c for c in cases if c.get('locale') == 'ru']
with open('eval/ru_cases.jsonl', 'w') as f:
    for c in ru_cases:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print(f'Wrote {len(ru_cases)} RU cases')
"

# Run regression against both pipelines
python stage1-immediate/run_cross_topic_regression.py \
    --cases-file eval/ru_cases.jsonl \
    --deployment gpu-deployment-v14 \
    --output eval/ru_direct_report.json

python stage1-immediate/run_cross_topic_regression.py \
    --cases-file eval/ru_cases.jsonl \
    --deployment gpu-deployment-ru-translate \
    --output eval/ru_translate_report.json

# Compare
python stage3-verify/compare_regression_reports.py \
    eval/ru_direct_report.json \
    eval/ru_translate_report.json \
    --output eval/ru_comparison.json
```

### 6.3 Compare Direct vs Translate Per-Cluster

For each of the 8 clusters, compare direct Qwen3 RU vs translate-to-RU:

| Cluster | Direct (Qwen3) | Translate | Delta | Recommendation |
|---------|---------------|-----------|-------|---------------|
| breakup | ___% | ___% | +/-___% | |
| work | ___% | ___% | +/-___% | |
| anxiety | ___% | ___% | +/-___% | |
| stress | ___% | ___% | +/-___% | |
| grief | ___% | ___% | +/-___% | |
| clarity | ___% | ___% | +/-___% | |
| somatic | ___% | ___% | +/-___% | |

### 6.4 Cutover RU to Translate if Improvement > 10%

If `gpu-deployment-ru-translate` outperforms direct Qwen3 RU by **> 10 percentage points overall**:

1. **Update frontend routing** to route 100% of RU traffic to `gpu-deployment-ru-translate`:

```typescript
// In route.ts — update the RU deployment resolver
function resolveDeployment(locale: string): string | undefined {
  if (locale === 'ru') return 'gpu-deployment-ru-translate';  // Force RU to translate
  if (locale === 'kk') return process.env.AML_DEPLOYMENT_NAME_KK;
  return undefined;
}
```

2. **Verify RU traffic is routing correctly:**

```bash
# Check deployment request distribution
az monitor metrics list \
    --resource $(az ml online-endpoint show --name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query id -o tsv) \
    --metric requests_total \
    --interval PT5M \
    --filter "deployment_name eq 'gpu-deployment-ru-translate'"
```

3. **Run 24h validation** — same checks as Section 5 but scoped to RU locale only.

4. **Keep monitoring** — re-evaluate when Qwen3 RU direct quality improves.

### 6.5 Long-Term: Migrate RU Back to Direct Qwen3

When Qwen3 RU direct quality meets or exceeds the translate pipeline:

```bash
# Re-run comparison
python stage3-verify/compare_regression_reports.py \
    eval/ru_translate_report.json \
    eval/ru_direct_report.json \
    --output eval/ru_reverse_comparison.json

# If direct >= translate, update frontend to remove forced routing
# Revert route.ts to use AML_DEPLOYMENT_NAME_RU env var
```

---

## Appendix A: Quick Reference Commands

### Azure ML Commands

```bash
# Show all deployments on endpoint
az ml online-endpoint show --name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query traffic

# Get deployment logs
az ml online-deployment get-logs --name gpu-deployment-v14 --endpoint-name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --lines 200

# Restart a deployment
az ml online-deployment update --name gpu-deployment-v14 --endpoint-name daisy-therapy --resource-group Daisy_group --workspace-name Daisy

# Scale deployment instance count
az ml online-deployment update --name gpu-deployment-v14 --endpoint-name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --set instance_count=2
```

### Regression Commands

```bash
# Full 56-case regression
python stage1-immediate/run_cross_topic_regression.py --deployment gpu-deployment-v14 --output eval/report.json

# Limited smoke test
python stage1-immediate/run_cross_topic_regression.py --limit 8 --output eval/smoke.json

# Compare reports
python stage3-verify/compare_regression_reports.py eval/before.json eval/after.json --format markdown --output eval/delta.md
```

### Health Check

```bash
# Quick health probe
curl -X POST https://daisy-therapy.westus2.inference.ml.azure.com/score \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(az ml online-endpoint list-keys --name daisy-therapy --resource-group Daisy_group --workspace-name Daisy --query primaryKey -o tsv)" \
  -H "azureml-model-deployment: gpu-deployment-v14" \
  -d '{"messages":[{"role":"user","content":"I am feeling anxious"}],"locale":"en"}'
```

---

## Appendix B: Azure Resource Map

| Resource | Name | Region | Notes |
|----------|------|--------|-------|
| Subscription | `9239bc75-105c-486e-8957-da8e49309c55` | — | Daisy production |
| Resource Group | `Daisy_group` | westus2 | Contains all ML resources |
| Workspace | `Daisy` | westus2 | Main ML workspace |
| Online Endpoint | `daisy-therapy` | westus2 | Production scoring endpoint |
| Deployment (prod) | `gpu-deployment-finetuned` | westus2 | Legacy production deployment |
| Deployment (RU) | `gpu-deployment-ru-translate` | westus2 | RU translate pipeline |
| Deployment (Qwen3) | `gpu-deployment-v14` | westus2 | Qwen3 candidate deployment |
| Container Registry | `daisymodels.azurecr.io` | westus2 | Model image registry |

---

## Appendix C: Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-07-01 | QA/Platform | Initial checklist for Qwen3 v14 cutover |
