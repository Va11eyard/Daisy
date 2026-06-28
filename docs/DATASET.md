# Training data format (aligned with inference)

Fine-tuning uses `tokenizer.apply_chat_template` on full conversations ([scripts/prepare_dataset.py](../scripts/prepare_dataset.py)). The **system** block should match what [inference/score.py](../inference/score.py) assembles via [inference/system_prompt.py](../inference/system_prompt.py) so LoRA does not learn a mismatched style.

## Record shape (JSON array or JSONL)

Each item:

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | yes | OpenAI-style list: `role` in `system`, `user`, `assistant` (and optional `tool`). Multi-turn encouraged. |
| `meta` | no | Object passed to `build_system_prompt` (see below). If present and no `system`, system text is built automatically. |
| `system` | no | Raw system string; **overrides** `meta` when no `system` message exists inside `messages`. |

Do **not** duplicate `system` both in `messages` and as a top-level field unless intentional.

## `meta` schema (maps to production inputs)

Same semantics as the scoring JSON body / Daisy web payload:

| Key | Type | Maps to inference |
|-----|------|-------------------|
| `locale` | string | `locale` / language line (`ru`, `en`, `kk`, …) |
| `persona` | string | `persona` — canonical keys: `warm_friend`, `practical_helper`, `gentle_explorer`, `calm_mentor`, `wise_teacher`, `flexible`, `soft_explorer` (aliases resolved in [inference/personas.py](../inference/personas.py)) |
| `onboarding_summary` | string | `onboarding_summary` (free text or JSON stringified for “About this person”) |
| `user_context` | string | `user_context` (memory / “Remember from past…”) |
| `psych_profile` | object | `psych_profile` — e.g. `ESI`, `BSI`, `SSI`, `PVI`, `MRI`, `riskLevel` |
| `user_image` | object | Optional compact profile ([inference/user_image.py](../inference/user_image.py) v1) |
| `user_gender` | string | `female` / `male` (optional, for Russian/Kazakh agreement lines) |
| `force_english` | bool | Same as translation path in `score.py` |
| `is_onboarding` | bool | First session after onboarding |
| `onboarding_step` | int | 0 = first greeting, 1 = one goal question, else closing |

**Daisy web / DB alignment (conceptual):** `onboarding_summary` ≈ `OnboardingData.responses` + slice of `User.aiProfile`; `user_context` ≈ `conversationMemory` + episodic memory bundle; `psych_profile` ≈ latest `PsychProfileSnapshot`; `user_image` ≈ coordinator-synthesized or server-built profile.

## Persona keys and aliases

Canonical keys live in `PERSONA_MAP`. Common **aliases** (also accepted in `meta.persona` when matching lowercase):

- `active_listener`, `тёплая подруга` → `warm_friend`
- `behavior_coach`, `практичный помощник` → `practical_helper`
- `questioner`, `мягкий исследователь` → `gentle_explorer`
- `emotion_control_provider`, `спокойный наставник` → `calm_mentor`
- `psychoeducator`, `мудрый учитель` → `wise_teacher`
- `flexible_companion`, `all_dynamic`, `гибкая собеседница` → `flexible`

Use **canonical English keys** in new data when possible.

## Persona buckets (style hints for data producers)

Short guidance per persona (full instructions are injected via `resolve_persona`):

| Persona | Train assistant turns to be… | Example user situations |
|---------|------------------------------|-------------------------|
| `warm_friend` | Validating, emotionally close, gentle | Loneliness, shame, “nobody gets me” |
| `practical_helper` | Concrete steps, small experiments | Overwhelm, procrastination, habits |
| `gentle_explorer` | Open questions, curiosity about meaning | Ambivalence, identity, values |
| `calm_mentor` | Steady, space, non-rushing | Anger regulation, guilt |
| `wise_teacher` | Name distortions lightly, psychoeducation | Catastrophizing, mind-reading |
| `flexible` | Mix styles as the thread needs | Default mixed threads |

Include **negative examples** sparingly: refusals to diagnose, crisis redirection (aligned with [config/crisis_resources.yaml](../config/crisis_resources.yaml)).

## Retired / banned sources (do not use for training)

The following corpora are **retired** — they teach book-summary voice and homogenous reflect+question templates:

| Path | Status | Why |
|------|--------|-----|
| `data/archive/train.jsonl.retired` | **Banned** | ~14k rows from `md_to_dialogues` / book dumps (`source_md:`, "In plain language:", chapter headings) |
| `data/archive/val.jsonl.retired` | **Banned** | Val split from the same book-dump pipeline |
| `data/raw/md_dialogues.json` | Use only via v13 filters | Raw md expansion; must pass [scripts/prepare_v13_dataset.py](../scripts/prepare_v13_dataset.py) quality gates |

**Allowed sources for v12+ training:**

- `data/train_v2.jsonl`, `data/train_v3.jsonl` (curated, full voice-contract prompt)
- `data/raw/v12_ru_dialogues.json` / `build_v12_ru_seed.py` output
- `data/raw/md_distilled_ru.jsonl` (teacher-distilled RU, filtered)
- `data/raw/v13_shape_synth.json` (shape-balanced synthetic buckets)
- Human-reviewed synth under `data/synthesized/`

Build curated sets with:

```powershell
python scripts/prepare_v12_dataset.py   # train_v12 / val_v12
python scripts/prepare_v13_dataset.py   # train_v13 / val_v13 (shape-balanced)
```

Never point `submit_training_job.py` at `data/train.jsonl` from a blind `prepare_dataset.py --input data/raw/md_dialogues.json` run.

## Quality and safety

- **PII:** strip or syntheticize names, phones, addresses.
- **Languages:** maintain target mix (e.g. ru / en / kk) in both train and val.
- **Length:** prefer multi-turn (≥4 messages) for context learning; cap extreme lengths before `prepare_dataset` if needed.
- **Dedup:** near-duplicate user utterances across rows hurt generalization — deduplicate or paraphrase.

## Markdown corpus (`data/md`) → диалоги

Книги/заметки в `data/md` можно развернуть в обучающие диалоги:

```powershell
# Офлайн (по умолчанию --quality rich): секции ##/###, до 22 чанков/файл, 2 архетипа на чанк
# (explain / apply / reflect / …), длинные чанки → 6 реплик; Emergency → хвост про очную помощь
python scripts/md_to_dialogues.py --md-root data/md --output data/raw/md_dialogues.json

# Компактнее / как раньше: один простой 4-ходовый диалог на чанк
python scripts/md_to_dialogues.py --quality standard --output data/raw/md_dialogues_std.json

# Дистилляция с учителем (нужен ключ и SDK)
$env:ANTHROPIC_API_KEY = "..."
pip install anthropic
python scripts/md_to_dialogues.py --mode api --provider anthropic --limit-files 20 --output data/raw/md_dialogues_api.json

# Полный корпус: отдельный скрипт с JSONL, flush и --resume (один вызов API на чанк)
python scripts/md_distill_api.py --output data/raw/md_distilled.jsonl --resume
python scripts/prepare_dataset.py --input data/raw/md_distilled.jsonl --output-dir data
```

`md_distilled.jsonl` крупный — в `.gitignore`; пересобирайте локально. Ограничить тест: `--limit-total-chunks 50`, `--sleep 0.5`, `--provider openai` при наличии `OPENAI_API_KEY`.

Параметры: `--max-chunks-per-file`, `--max-chars`, `--limit-files` (пробный прогон), `--mode heuristic|api`. Для API: `TEACHER_MODEL`, `OPENAI_API_KEY` + `--provider openai`.

Файл `data/raw/md_dialogues.json` получается большим (~25 MB+ на полном корпусе) и **в .gitignore** — пересобирайте локально. Полный прогон по `data/md` (сотни файлов) даёт порядка **тысяч** примеров диалогов; затем `prepare_dataset.py` режет **train/val** (например ~90 % / 10 %).

Затем тот же `prepare_dataset.py` с `--input` на получившийся JSON.

## Build command

```powershell
cd <repo-root>
set BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
python scripts/prepare_dataset.py --input data/raw/md_dialogues.json --output-dir data --val-ratio 0.1
```

Или из старого curated JSON:

```powershell
python scripts/prepare_dataset.py --input data/raw/daisy_curated.json --output-dir data --val-ratio 0.1
```

Outputs `data/train.jsonl` and `data/val.jsonl` for [training/train.py](../training/train.py).

## Azure ML training

[scripts/submit_training_job.py](../scripts/submit_training_job.py) copies `data/train.jsonl` and `data/val.jsonl` into `training/` before creating the job (the uploaded code bundle is only the `training/` folder). Run prepare_dataset first, then `python scripts/submit_training_job.py` with Azure env vars set.
