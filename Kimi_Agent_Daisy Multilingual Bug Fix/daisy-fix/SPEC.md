# Daisy Fix SPEC — Production Implementation Package

## Overview
Complete fix package for Daisy (talktodaisy.com) multilingual AI therapy chatbot. Addresses train/serve misalignment, 99.9%-EN training data, 23-module inference debt, and disabled QC.

## Architecture Decision
**Option A (primary):** Clean-slate Qwen3-8B + balanced EN/RU/KK LoRA + simplified 3-layer inference
**Option B (parallel):** RU locale-split routing via gpu-deployment-ru-translate as short-term stabilization

## Directory Layout
```
daisy-fix/
├── SPEC.md                              # This file
├── stage1-immediate/                    # Day 1-2: Baseline + RU routing
│   ├── run_cross_topic_regression.py    # 56-case regression runner
│   ├── deployment-lora-v11-ru-translate-v2.yaml  # Updated A/B deployment
│   ├── route.ts                         # Locale-aware frontend (full replacement)
│   ├── audit_training_locale.py         # Training data locale audit + fix
│   └── cutover_traffic.ps1              # Azure traffic cutover script
├── stage2-qwen3-migration/              # Day 3-7: New architecture
│   ├── deployment-qwen3-lora-v15.yaml   # Qwen3 production deployment
│   ├── score_qwen3.py                   # Simplified 3-layer inference
│   ├── system_prompt_qwen3.py           # Clean system prompt (no dead vars)
│   ├── voice_qc_lightweight.py          # Re-enabled voice QC
│   ├── build_balanced_dataset.py        # Build ≥20% RU dataset
│   ├── synthesize_ru_kk_dialogues.py    # Generate RU/KK therapy dialogues
│   └── strip_latin_leaks.py             # Clean Latin from Cyrillic turns
├── stage3-verify/                       # Day 8-10: Verification
│   ├── compare_regression_reports.py    # Before/after comparison
│   └── cutover_checklist.md             # Production cutover checklist
└── README.md                            # Execution guide
```

## Module Specifications

### 1. run_cross_topic_regression.py
**Purpose**: Baseline production quality before any changes.

**Input**: `eval/cross_topic_regression.jsonl` (56 cases, 8 clusters × EN/RU)

**Output**: `eval/results/cross_topic_regression_report.json` with per-cluster breakdown.

**Interface**:
```python
def run_case(endpoint: str, case: dict, deployment: str = None) -> dict:
    """Run single case, return score dict."""
    
def score_response(text: str, case: dict) -> dict:
    """Score response against criteria. Returns:
    {
        'passed': bool,
        'length_ok': bool,
        'no_canned_greeting': bool,
        'no_structural_leak': bool,
        'no_script_leak': bool,
        'keyword_match': bool,
        'not_hollow': bool,
        'locale_correct': bool,  # RU: no EN/PL/DE/ES mid-sentence
        'latency_ms': int,
        'failure_reasons': list[str]
    }
    """

def main():
    # Load cases
    # Run each against endpoint
    # Aggregate by cluster
    # Print: overall pass %, per-cluster pass %, failure breakdown
```

**Pass criteria per case**:
- Response length >= 25 chars
- No canned greeting ("Hey -- I'm glad you're here...")
- No structural leak ("Assistant:", rubric tokens, ".\,.\,.")
- No script leak (3+ ASCII words in Cyrillic, Polish diacritics)
- At least one keyword match
- Not hollow one-liner
- RU: informal ты, no foreign script sentences

**Scoring rules**:
- `structural_leak`: regex detects `Assistant:`, `Question:`, `\.\.\.+`, rubric tokens
- `script_leak`: Latin word count >= 3 in Cyrillic response OR Polish diacritics detected
- `canned_greeting`: exact match or >80% Jaccard with known canned greetings
- `locale_correct`: for `ru`/`kk` locale, no sentence with >50% Latin words

### 2. deployment-lora-v11-ru-translate-v2.yaml
**Purpose**: Production-ready A/B deployment for RU translate routing.

**Changes from v1**:
- Add `DAISY_POST_TRANSLATE_QC: "true"` — enable post-translate quality check
- Add `DAISY_TRANSLATE_MIN_LENGTH: "40"` — reject too-short translations
- Add `DAISY_TRANSLATE_SCRIPT_GUARD: "true"` — Latin leak check after translate
- Update `INFERENCE_BUILD: "2026-07-lora-v11-ru-translate-v2"`
- Keep all other env vars identical for controlled comparison

### 3. route.ts
**Purpose**: Locale-aware error handling + deployment header routing.

**Interface**:
```typescript
// Error messages by locale
const ERROR_MESSAGES: Record<string, { user: string; detail: string }> = {
  en: { user: "Something went wrong. Please try again.", detail: "..." },
  ru: { user: "Извини, произошла ошибка. Попробуй ещё раз.", detail: "..." },
  kk: { user: "Қате орын алды. Қайта байқап көріңіз.", detail: "..." },
};

// Deployment routing by locale
function resolveDeployment(locale: string): string | undefined {
  if (locale === 'ru') return process.env.AML_DEPLOYMENT_NAME_RU;
  if (locale === 'kk') return process.env.AML_DEPLOYMENT_NAME_KK;
  return undefined; // Use default
}

// In fetch: headers['azureml-model-deployment'] = resolveDeployment(body.locale)
```

### 4. audit_training_locale.py
**Purpose**: Audit and fix training JSONL files.

**Interface**:
```python
def audit_file(path: str) -> dict:
    """Return locale mix, leak counts, sample leaks."""

def fix_latin_leaks(input_path: str, output_path: str) -> dict:
    """Strip English meta-phrases from Cyrillic assistant turns.
    Returns: rows_processed, rows_fixed, fix_samples."""

def main():
    # Audit all data/*.jsonl files
    # Fix: remove "Daisy noticed" from RU assistant turns
    # Fix: strip English parentheticals from Cyrillic text
    # Report: locale mix summary, before/after leak counts
```

**Fix rules**:
- Replace `Daisy noticed[,]? ` with empty string in Cyrillic turns
- Strip English parentheticals like `(trauma bonding)`, `(compassion fatigue)` — keep Cyrillic equivalent only
- Remove lines where Cyrillic assistant turn is >30% Latin after stripping

### 5. score_qwen3.py
**Purpose**: Simplified 3-layer inference pipeline for Qwen3.

**Architecture** (3 layers, not 23 modules):
```
Layer 0: Safety — crisis detection, injection guard, off-topic routing
Layer 1: Generate — single generation call with stop strings, temp, max_tokens
Layer 2: Quality Control — lightweight checks (script leak, structural leak, min length)
```

**Interface**:
```python
async def run(request: dict) -> dict:
    """Main entry. Request: {messages, locale, history}.
    Response: {reply, metadata}.
    """
    
async def layer0_safety(messages: list, locale: str) -> SafetyResult:
    """Crisis → escalation; injection → block; else pass."""

async def layer1_generate(messages: list, locale: str) -> GenerationResult:
    """Single generate with locale-aware system prompt."""

async def layer2_qc(text: str, locale: str) -> QCResult:
    """Check script leak, structural leak, min length. 
    Fail → one regen, then ship with warning flag."""
```

**Env vars (minimal, all functional)**:
- `BASE_MODEL`: Qwen3-8B or Qwen3-4B
- `INFERENCE_BUILD`: version string
- `INFERENCE_QUANTIZATION`: 4bit/8bit/none
- `DAISY_DIRECT_MULTILINGUAL`: true
- `DAISY_DEFAULT_MAX_TOKENS`: 120 (up from 90)
- `DAISY_LORA_DEFAULT_TEMP`: 0.6
- `DAISY_VOICE_QC`: true (re-enabled)
- `DAISY_MAX_VOICE_REGENS`: 2

**Removed/dead vars** (not referenced in code):
- DAISY_RAG, DAISY_BM25, DAISY_CONFIDENCE_GATE, DAISY_RUBRIC_JUDGE
- DAISY_AGGRESSIVE_TRIM, ENABLE_ROUTER_PASS

### 6. build_balanced_dataset.py
**Purpose**: Build training dataset with balanced EN/RU/KK.

**Interface**:
```python
def build_dataset(
    en_sources: list[str],
    ru_sources: list[str], 
    kk_sources: list[str],
    target_mix: dict = {"en": 0.40, "ru": 0.35, "kk": 0.25},
    output_path: str = "data/train_v15.jsonl"
) -> dict:
    """Build balanced dataset from sources.
    Returns: row_counts, locale_mix, quality_metrics.
    """

def validate_dataset(path: str) -> dict:
    """Check: ≥20% RU, ≥15% KK, 0 Latin leaks in Cyrillic, no canned repeats."""
```

**Data sources**:
- EN: `data/train_v13.jsonl` (cleaned, shape-balanced) + synthesized EN turns
- RU: Cleaned `training/train.jsonl` legacy RU rows + `synthesize_ru_kk_dialogues.py` output
- KK: Synthesized KK dialogues from Qwen3 + verified translations

### 7. synthesize_ru_kk_dialogues.py
**Purpose**: Generate high-quality RU and KK therapy dialogues.

**Interface**:
```python
SCENARIOS = [
    # breakup
    "user just broke up with partner, feeling empty",
    "user ended relationship, now has regret",
    # work
    "user's boss is yelling daily, burnout",
    "user fears being fired, financial anxiety",
    # anxiety
    "user has tight chest, can't calm down",
    "user has panic attacks, scared",
    # stress
    "everything overwhelming, can't switch off",
    "physical tension in shoulders",
    # grief
    "parent died, holidays are hard",
    "feel guilty when having fun after loss",
    # clarity
    "head spinning with decisions",
    "can't tell what actually want",
    # somatic
    "emptiness in chest after breakup",
    "body won't relax at night",
]

async def synthesize_dialogue(scenario: str, locale: str, model: str) -> dict:
    """Generate one dialogue turn pair (user_message, assistant_response).
    Returns ChatML format dict."""

async def main():
    # For each scenario × locale (ru, kk):
    #   Generate 20-30 variations with different phrasing
    #   Validate: no Latin leaks, informal ты, ≥50 chars
    #   Save to data/synthesized/{locale}/batch_{n}.jsonl
```

### 8. voice_qc_lightweight.py
**Purpose**: Re-enabled voice quality control, simplified.

**Interface**:
```python
class VoiceQC:
    BANNED_PATTERNS = [
        r"^Hey\s*[-—]\s*I'm glad you're here",
        r"Assistant:\s*",
        r"Question:\s*",
        r"\.\s*,\s*\.\s*,",  # punctuation loops
    ]
    
    MIN_LENGTH = 25
    MAX_LATIN_RATIO_IN_CYRILLIC = 0.10  # down from 0.12
    
    def check(self, text: str, locale: str) -> QCResult:
        """Check banned patterns, min length, script leaks.
        Returns: {passed, failures, can_regen}.
        """
    
    def regenerate_prompt(self, original_prompt: str, failures: list) -> str:
        """Add guardrails to prompt for regen attempt."""
```

## Data Flow

### Stage 1 (Immediate)
```
cross_topic_regression.jsonl ──► run_cross_topic_regression.py ──► report.json
                                                               ▼
deployment-lora-v11-ru-translate-v2.yaml ──► Azure A/B deployment
route.ts ──► Frontend locale-aware errors + deployment routing
audit_training_locale.py ──► Cleaned training files + audit report
cutover_traffic.ps1 ──► Azure traffic management
```

### Stage 2 (Qwen3 Migration)
```
Raw training files ──► audit_training_locale.py ──► Cleaned files
                                                        ▼
Synthesize RU/KK ──► synthesize_ru_kk_dialogues.py ──► data/synthesized/
                                                        ▼
Cleaned + synthesized ──► build_balanced_dataset.py ──► train_v15.jsonl
                                                        ▼
score_qwen3.py + deployment-qwen3-lora-v15.yaml ──► gpu-deployment-v14
```

## Integration Points

### Azure ML
- Endpoint: `daisy-therapy`
- Subscription: `9239bc75-105c-486e-8957-da8e49309c55`
- RG: `Daisy_group`, Workspace: `Daisy`, Region: `westus2`
- Deployments: gpu-deployment-finetuned (prod), gpu-deployment-ru-translate (A/B), gpu-deployment-v14 (Qwen3)

### Frontend
- File: `C:\Users\Valleyard\Daisy\src\app\api\chat\route.ts`
- Env: `AI_API_URL`, `AI_API_KEY`, `AML_DEPLOYMENT_NAME_RU`, `AML_DEPLOYMENT_NAME_KK`

### Repos
- Inference: `E:\WebstormProjects\Daisy-1`
- Training: `E:\WebstormProjects\Daisy-Model`
- Frontend: `C:\Users\Valleyard\Daisy`

## Success Criteria
- ≥90% pass rate overall on 56-case regression
- ≥85% pass rate per cluster
- 0 structural_leak or script_leak failures
- No canned greeting on any case
- RU: informal ты, no EN/PL/DE/ES mid-sentence
- EN: no identical template on paraphrased inputs
- P50 latency <15s on T4
- Locale-aware error messages
