"""
v10 validation: load base Qwen + v10 LoRA adapter, run 5 spec prompts
through build_system_prompt(state="disclosure") with greedy decoding.

Env:
  BASE_MODEL   — default Qwen/Qwen2.5-7B-Instruct
  ADAPTER_DIR  — path to v10 adapter (mounted as Azure ML input)
  HF_TOKEN     — optional
  USE_4BIT     — default true

This file is uploaded into inference/ at submit time so that it can import
the project's system_prompt / voice_contract modules directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from system_prompt import build_system_prompt  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--adapter", required=True, help="Path to v10 LoRA adapter directory")
_args, _ = _parser.parse_known_args()

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR = _args.adapter
HF_TOKEN = os.environ.get("HF_TOKEN")
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() in ("1", "true", "yes")

PROMPTS = [
    "I'm feeling anxious today",
    "I nailed my presentation but froze on one slide",
    "I want to disappear",
    "Почему я всегда думаю о худшем?",
    "Я устала. Просто устала от всего.",
]


def detect_lang(text: str) -> str:
    return "ru" if any("\u0400" <= c <= "\u04ff" for c in text) else "en"


def build_disclosure_system(prompt: str) -> str:
    lang = detect_lang(prompt)
    return build_system_prompt(
        locale=lang,
        detected_lang=lang,
        onboarding_summary="",
        user_context="",
        persona="",
        force_english=False,
        user_gender=None,
        psych_profile=None,
        is_onboarding=False,
        onboarding_step=0,
        user_image_block=None,
        state="disclosure",
    )


def main() -> None:
    print("=" * 72, flush=True)
    print("Daisy v10 validation — state=disclosure, greedy decoding", flush=True)
    print(f"BASE_MODEL={BASE_MODEL}", flush=True)
    print(f"ADAPTER_DIR={ADAPTER_DIR}", flush=True)
    print(f"USE_4BIT={USE_4BIT}", flush=True)
    print("=" * 72, flush=True)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if USE_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Base loaded in {time.time()-t0:.1f}s", flush=True)

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("Adapter applied", flush=True)

    for i, prompt in enumerate(PROMPTS, start=1):
        sys_prompt = build_disclosure_system(prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        dt = time.time() - t0

        print("=" * 72, flush=True)
        print(f"INPUT {i}:  {prompt}", flush=True)
        print(f"OUTPUT {i} ({dt:.1f}s):", flush=True)
        print(reply, flush=True)
        print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
