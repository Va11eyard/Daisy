"""Load HF model: full checkpoint, or base + optional PEFT adapter (Azure ML model dir).

Default base is Qwen/Qwen3-8B (Apache-2.0), which fits a single T4 16GB in 4-bit.
The coordinator/router second model has been removed — routing is now a pure
phase-detection step (see router.py), so init() loads exactly one model.

Optional inference quantization (VRAM savings, recommended for 8B on T4):
  INFERENCE_QUANTIZATION=none | 4bit | 8bit   (default: none — fp16 on GPU)

When 4bit/8bit is set and a LoRA adapter is present, the adapter is kept as PEFT (no merge)
so bitsandbytes quantized weights stay valid.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

_adapter_loaded: bool = False


def adapter_loaded() -> bool:
    return _adapter_loaded


def _has_full_model_weights(d: Path) -> bool:
    """Hugging Face full save: config.json + weight shards (no adapter_config.json)."""
    if not (d / "config.json").exists():
        return False
    if (d / "adapter_config.json").exists():
        return False
    if list(d.glob("*.safetensors")):
        return True
    if (d / "pytorch_model.bin").exists():
        return True
    if (d / "model.safetensors.index.json").exists():
        return True
    return False


def _find_full_checkpoint(model_dir: Path) -> Path | None:
    if _has_full_model_weights(model_dir):
        return model_dir
    for config in model_dir.rglob("config.json"):
        parent = config.parent
        if _has_full_model_weights(parent):
            return parent
    return None


def _find_adapter_root(model_dir: Path) -> Path | None:
    if (model_dir / "adapter_config.json").exists():
        return model_dir
    for p in model_dir.rglob("adapter_config.json"):
        return p.parent
    return None


def _inference_quantization_config() -> BitsAndBytesConfig | None:
    """BitsAndBytes config for GPU inference; None = fp16/bf16 path."""
    mode = os.environ.get("INFERENCE_QUANTIZATION", "none").lower().strip()
    if mode in ("none", "", "fp16", "bf16", "float16"):
        return None
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    logger.warning("Unknown INFERENCE_QUANTIZATION=%s; using fp16", mode)
    return None


def _from_pretrained_causal(
    model_id: str,
    *,
    hf_token: str | None,
    device: str,
    quant_cfg: BitsAndBytesConfig | None,
):
    common = {
        "token": hf_token,
        "trust_remote_code": True,
    }
    if device == "cuda" and quant_cfg is not None:
        logger.info("Loading model with INFERENCE_QUANTIZATION (bitsandbytes)")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map="auto",
            **common,
        )
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        **common,
    )


def load_model_and_tokenizer(
    base_model: str,
    hf_token: str | None,
):
    global _adapter_loaded
    _adapter_loaded = False
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "/var/azureml-app/azureml-models"))
    logger.info("AZUREML_MODEL_DIR=%s", model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_cfg = _inference_quantization_config() if device == "cuda" else None
    if quant_cfg is not None and device == "cpu":
        logger.warning("INFERENCE_QUANTIZATION ignored on CPU")

    dtype = torch.float16 if device == "cuda" else torch.float32

    full_ckpt: Path | None = None
    if model_dir.exists():
        for sub in ["daisy-full", "daisy-finetuned-full", "model", "."]:
            cand = model_dir / sub if sub != "." else model_dir
            if cand.is_dir():
                full_ckpt = _find_full_checkpoint(cand)
                if full_ckpt:
                    break
        if full_ckpt is None:
            full_ckpt = _find_full_checkpoint(model_dir)

    if full_ckpt is not None:
        logger.info("Loading full fine-tuned checkpoint from %s", full_ckpt)
        _adapter_loaded = True
        tokenizer = AutoTokenizer.from_pretrained(str(full_ckpt), token=hf_token, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = _from_pretrained_causal(str(full_ckpt), hf_token=hf_token, device=device, quant_cfg=quant_cfg)
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    adapter_root: Path | None = None
    if model_dir.exists():
        for sub in ["daisy-finetuned-lora", "model", "."]:
            cand = model_dir / sub if sub != "." else model_dir
            if cand.is_dir():
                adapter_root = _find_adapter_root(cand)
                if adapter_root:
                    break
        if adapter_root is None:
            adapter_root = _find_adapter_root(model_dir)

    model = _from_pretrained_causal(base_model, hf_token=hf_token, device=device, quant_cfg=quant_cfg)

    if adapter_root and adapter_root.exists():
        logger.info("Loading LoRA from %s", adapter_root)
        _adapter_loaded = True
        model = PeftModel.from_pretrained(model, str(adapter_root))
        if quant_cfg is None:
            model = model.merge_and_unload()
            if device == "cuda":
                model = model.to(dtype=dtype)
        else:
            logger.info("Keeping LoRA adapter merged=False (quantized inference)")
    else:
        _adapter_loaded = False
        logger.warning("No LoRA adapter or full checkpoint found; using BASE_MODEL weights only")

    model.eval()
    return model, tokenizer
