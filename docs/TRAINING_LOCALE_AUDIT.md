# Training data locale audit

Run: `python scripts/audit_training_locale.py`

Output: [`eval/results/training_locale_audit.json`](../eval/results/training_locale_audit.json)

## Key findings (2026-07-01)

| File | Examples | Row locales | Cyrillic assistant + Latin leak rows |
|------|----------|-------------|--------------------------------------|
| `data/train_v3.jsonl` (v11 train) | 2,827 | 99.7% EN | 740 (26%) |
| `data/train_v12.jsonl` | 2,565 | 99.9% EN | 677 (26%) |
| `data/train_v13.jsonl` | 2,414 | 99.9% EN | 445 (18%) |
| `training/train.jsonl` (legacy) | 14,660 | 69% RU rows | 3,710 |
| `data/raw/_rendered_v12_ru_dialogues` | 9 | EN rows, RU assistants | 0 |

**Production LoRA v11** was trained on `train_v3.jsonl` — almost entirely English user rows with mixed RU/EN assistant text and frequent `Daisy noticed` English fragments inside Cyrillic replies.

**Implication:** Retrain must (1) balance locales, (2) strip English meta-phrases from RU assistant turns, (3) not use `training/train.jsonl` without audit (legacy mix).

## train_v15 (2026-07-01 — Kimi integration)

Pipeline:

```powershell
python scripts/strip_latin_leaks.py -i data/train_v13.jsonl -o data/cleaned/train_v13_clean.jsonl
python scripts/synthesize_ru_kk_dialogues.py --output-dir data/synthesized
python scripts/build_balanced_dataset.py --en-sources data/cleaned/train_v13_clean.jsonl --ru-sources data/synthesized/ru data/raw/_rendered_v12_ru_dialogues --kk-sources data/synthesized/kk --output data/train_v15.jsonl
python scripts/strip_latin_leaks.py -i data/train_v15.jsonl -o data/train_v15.jsonl.tmp  # in-place clean
python scripts/build_balanced_dataset.py --validate-only data/train_v15.jsonl
```

| Metric | Value |
|--------|-------|
| `data/train_v15.jsonl` rows | 2,675 |
| `data/val_v15.jsonl` rows | 140 |
| Assistant locale mix (validated) | EN 48%, RU 48%, KK 25% |
| Latin leaks in Cyrillic assistants | **0** |
| `build_balanced_dataset` validation | **PASSED** |

Artifacts: `eval/results/training_v15_audit.json`, Azure ML job `daisy-lora-v15-qwen3`.
