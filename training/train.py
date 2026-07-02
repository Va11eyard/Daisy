"""
Fine-tuning for Daisy (Azure ML Command job entry).

Two modes (USE_LORA env):
  USE_LORA=true (default) — PEFT QLoRA (4-bit NF4 by default); fits a 7B run on one T4.
  USE_4BIT=false — LoRA on 8-bit base (needs more VRAM; lower batch / seq on T4).
  USE_LORA=false — full-parameter fine-tuning of the pretrained model (all weights train).
    Still starts from pretrained weights; it does NOT train "from random init" and cannot
    erase general world knowledge in one small run — only shifts behavior toward your data.

Environment:
  BASE_MODEL   — Hugging Face model id (default: Qwen/Qwen2.5-7B-Instruct)
  USE_LORA     — true | false (default: true)
  HF_TOKEN     — Hugging Face token if needed
  TRAIN_FILE   — train JSONL (default: train_v2.jsonl)
  VAL_FILE     — val JSONL (default: val_v2.jsonl)
  OUTPUT_DIR   — default: ./outputs/daisy-lora-v11 or ./outputs/daisy-full
  USE_4BIT     — true | false (default: true) — 4-bit QLoRA vs 8-bit k-bit LoRA
  MAX_SEQ_LENGTH — default 512 on LoRA (T4-safe); raise if you have headroom
"""

from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "train_v2.jsonl")
VAL_FILE = os.environ.get("VAL_FILE", "val_v2.jsonl")
USE_LORA = os.environ.get("USE_LORA", "true").lower() in ("1", "true", "yes")
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() in ("1", "true", "yes")

_DEFAULT_OUT = "./outputs/daisy-lora-v11" if USE_LORA else "./outputs/daisy-full"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUT)


def main() -> None:
    print("=" * 72)
    print("Daisy fine-tuning — LoRA" if USE_LORA else "Daisy fine-tuning — FULL weights")
    print(f"BASE_MODEL={BASE_MODEL}")
    print(f"USE_LORA={USE_LORA} USE_4BIT={USE_4BIT}")
    print(f"TRAIN_FILE={TRAIN_FILE} VAL_FILE={VAL_FILE}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    print("=" * 72)

    if USE_LORA and not torch.cuda.is_available():
        raise RuntimeError(
            "LoRA + k-bit requires a working CUDA GPU. PyTorch did not see one — often a "
            "CUDA mismatch (use cu118 wheels with Azure's cuda11.8 image; see training/conda.yaml). "
            "Without GPU, training can be killed by the OS (OOM / SIGKILL)."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if USE_LORA:
        # QLoRA 4-bit (default) fits 7B on a 16GB T4; 8-bit needs smaller batch/seq.
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        if USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=int(os.environ.get("LORA_R", "16")),
            lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
            target_modules=os.environ.get(
                "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
            ).split(","),
            lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable:,} trainable parameters (all weights)")

    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_ds = load_dataset("json", data_files=VAL_FILE, split="train")
    print(f"Train rows: {len(train_ds)}, Val rows: {len(val_ds)}")

    def _messages_to_text(messages: list[dict]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = (msg.get("content") or "").strip()
            if content:
                parts.append(f"<|im_start|>{role}\n{content}")
        return "\n".join(parts)

    def to_text(example: dict) -> dict:
        text = (example.get("text") or "").strip()
        if not text:
            text = _messages_to_text(example.get("messages") or [])
        return {"text": text}

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_text, remove_columns=val_ds.column_names)
    train_ds = train_ds.filter(lambda x: bool((x.get("text") or "").strip()))
    val_ds = val_ds.filter(lambda x: bool((x.get("text") or "").strip()))
    print(f"After format: train={len(train_ds)}, val={len(val_ds)}")

    _default_max = "512" if USE_LORA else "1024"
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", _default_max))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    epochs = float(os.environ.get("NUM_EPOCHS", "3"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "200"))
    save_steps = int(os.environ.get("SAVE_STEPS", str(eval_steps)))

    if USE_LORA:
        # T4 16GB: batch 1 + grad_accum 16 keeps headroom for 7B QLoRA + seq 512.
        train_bs = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
        eval_bs = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "1"))
        grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16"))
        lr = float(os.environ.get("LEARNING_RATE", "2e-4"))
        optim = "paged_adamw_8bit"
        fp16 = True
        bf16 = False
    else:
        train_bs = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
        eval_bs = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "1"))
        grad_accum = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16"))
        lr = float(os.environ.get("LEARNING_RATE", "2e-5"))
        optim = "adamw_torch"
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = not use_bf16
        bf16 = use_bf16

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=int(os.environ.get("WARMUP_STEPS", "100")),
        logging_steps=int(os.environ.get("LOGGING_STEPS", "10")),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "2")),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=True,
        optim=optim,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "2")),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    metrics = trainer.evaluate()
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Artifacts:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
