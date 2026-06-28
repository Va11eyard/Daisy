"""
Local smoke test for Daisy v10 LoRA adapter.

Loads base Qwen2.5-7B-Instruct + local v10 adapter, runs 5 spec prompts
through build_system_prompt(state="disclosure") with greedy decoding.

CPU-only (no CUDA required). Expect several minutes per prompt on 7B fp16.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "inference"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from system_prompt import build_system_prompt  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = str(REPO_ROOT / "outputs" / "daisy-lora-v10")

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
    print("=" * 72)
    print("Daisy v10 local smoke test — state=disclosure, greedy decoding")
    print(f"Base:    {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Device:  CPU (torch {torch.__version__})")
    print("=" * 72, flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (7B fp16, CPU) — this takes a few minutes...", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  base loaded in {time.time()-t0:.1f}s", flush=True)

    print("Applying v10 LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("  adapter applied", flush=True)

    for i, prompt in enumerate(PROMPTS, start=1):
        sys_prompt = build_disclosure_system(prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")

        print(f"\n[{i}/5] generating...", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        dt = time.time() - t0

        print("=" * 72)
        print(f"INPUT {i}:  {prompt}")
        print(f"OUTPUT {i} ({dt:.1f}s):")
        print(reply)
        print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
