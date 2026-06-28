"""
Standalone ablation runner — composes inference components without calling score.run().

Configs (cumulative):
  C0: base model, neutral prompt, no LoRA/RAG/QC
  C1: + LoRA
  C2: + full prompt wrapper (DAISY_PROMPT_MODE=full)
  C3: + RAG block
  C4: + anti-hallucination layers (voice QC, rubric judge, retries, fallback, ensure_open_question)

Also runs decoding sweep at C1: temperature x top_p grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent if (EVAL_DIR.parent / "inference" / "book_knowledge.py").exists() else EVAL_DIR
INFERENCE_DIR = ROOT / "inference" if (ROOT / "inference" / "book_knowledge.py").exists() else ROOT
if (EVAL_DIR / "metrics.py").exists():
    sys.path.insert(0, str(EVAL_DIR))
else:
    EVAL_DIR = ROOT / "eval"
    sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(INFERENCE_DIR))

from metrics import adjacent_deltas, summarize_config  # noqa: E402

NEUTRAL_SYSTEM = (
    "You are Daisy, a warm companion for emotional support. "
    "Respond helpfully in 2-4 sentences. End with one open question."
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_CONTEXT_TOKENS = 4096
RESERVED_FOR_RESPONSE_TOKENS = 900
DEFAULT_MAX_TOKENS = 256


@dataclass
class AblationConfig:
    name: str
    use_lora: bool = False
    prompt_mode: str = "neutral"  # neutral | full
    use_rag: bool = False
    anti_halluc: bool = False
    use_fallback: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15


CUMULATIVE_CONFIGS: list[AblationConfig] = [
    AblationConfig("C0_base"),
    AblationConfig("C1_lora", use_lora=True),
    AblationConfig("C2_prompt", use_lora=True, prompt_mode="full"),
    AblationConfig("C3_rag", use_lora=True, prompt_mode="full", use_rag=True),
    AblationConfig("C4_full", use_lora=True, prompt_mode="full", use_rag=True, anti_halluc=True),
    AblationConfig(
        "C4_nofallback",
        use_lora=True,
        prompt_mode="full",
        use_rag=True,
        anti_halluc=True,
        use_fallback=False,
    ),
]

DECODING_SWEEP: list[tuple[float, float]] = [
    (0.5, 0.9),
    (0.7, 0.9),
    (0.9, 0.9),
    (0.7, 1.0),
]


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def order_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process multi-turn chains after their dependencies."""
    by_id = {c["id"]: c for c in cases}
    done: set[str] = set()
    ordered: list[dict[str, Any]] = []

    def visit(cid: str) -> None:
        if cid in done:
            return
        c = by_id.get(cid)
        if not c:
            return
        dep = c.get("depends_on")
        if dep:
            visit(dep)
        ordered.append(c)
        done.add(cid)

    for c in cases:
        visit(c["id"])
    return ordered


def resolve_history(
    history: list[dict[str, str]],
    prior_responses: dict[str, str],
    case_id: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in history:
        content = m.get("content", "")
        if content == "PLACEHOLDER":
            content = prior_responses.get(case_id, "")
        out.append({"role": m["role"], "content": content})
    return out


def load_model_and_tokenizer(use_lora: bool, adapter_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_mode = os.environ.get("INFERENCE_QUANTIZATION", "4bit").lower()
    quant_cfg = None
    if device == "cuda" and quant_mode == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    common = {"token": hf_token, "trust_remote_code": True}
    if device == "cuda" and quant_cfg:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_cfg, device_map="auto", **common
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map="auto" if device == "cuda" else None, **common
        )

    if use_lora and adapter_path and Path(adapter_path).exists():
        print(f"Loading LoRA from {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        if quant_cfg is None:
            model = model.merge_and_unload()
    elif use_lora:
        print("WARNING: LoRA requested but adapter path missing; using base weights", flush=True)

    model.eval()
    return model, tokenizer


def build_system(
    case: dict[str, Any],
    cfg: AblationConfig,
    rag_block: str = "",
) -> str:
    if cfg.prompt_mode == "neutral":
        return NEUTRAL_SYSTEM

    os.environ["DAISY_PROMPT_MODE"] = "full"
    from system_prompt import build_system_prompt  # noqa: WPS433

    state = case.get("state", "intake")
    locale = case.get("locale", "en")
    system = build_system_prompt(
        locale=locale,
        detected_lang=locale,
        onboarding_summary="",
        user_context="",
        persona="flexible_companion",
        force_english=False,
        user_gender=None,
        psych_profile=None,
        is_onboarding=False,
        onboarding_step=0,
        user_image_block=None,
        state=state,
    )
    if rag_block:
        system += "\n\n" + rag_block
    return system


def generate_raw(
    model: Any,
    tokenizer: Any,
    system: str,
    history: list[dict[str, str]],
    user_message: str,
    cfg: AblationConfig,
) -> str:
    import torch
    from context_policy import fit_messages_to_token_budget  # noqa: WPS433

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    max_input = max(512, MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE_TOKENS)
    messages = fit_messages_to_token_budget(messages, tokenizer, max_input)
    prompt = cast(
        str,
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": DEFAULT_MAX_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": cfg.repetition_penalty,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "do_sample": True,
    }
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def apply_anti_halluc_layers(
    response: str,
    *,
    model: Any,
    tokenizer: Any,
    system: str,
    history: list[dict[str, str]],
    user_message: str,
    case: dict[str, Any],
    cfg: AblationConfig,
) -> tuple[str, dict[str, int]]:
    """Mirror score.py safety loops (voice QC, rubric, brief/degenerate, ensure_open_question)."""
    from book_knowledge import rubric_similarity_fail  # noqa: WPS433
    from generation import (  # noqa: WPS433
        ensure_open_question,
        fallback_reply,
        is_degenerate_output,
        is_near_duplicate_reply,
        is_too_brief,
        postprocess_model_response,
    )
    from voice_qc import violates_voice_contract, voice_regen_suffix  # noqa: WPS433

    os.environ.setdefault("DAISY_BOOK_KNOWLEDGE", "true")
    os.environ.setdefault("DAISY_RUBRIC_JUDGE", "true")

    state = case.get("state", "intake")
    locale = case.get("locale", "en")
    stats = {
        "voice_retry_count": 0,
        "used_fallback": 0,
        "brief_retry_count": 0,
        "degenerate_retry_count": 0,
    }

    response, _ = postprocess_model_response(response, locale)
    last_assistant = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
        None,
    )

    while stats["voice_retry_count"] < 2 and violates_voice_contract(
        response, state, reply_lang=locale, user_message=user_message
    ):
        stats["voice_retry_count"] += 1
        suffix = voice_regen_suffix(state, locale)
        regen_system = system + suffix
        raw = generate_raw(model, tokenizer, regen_system, history, user_message, cfg)
        response, _ = postprocess_model_response(raw, locale)

    if violates_voice_contract(response, state, reply_lang=locale, user_message=user_message):
        if cfg.use_fallback:
            stats["used_fallback"] = 1
            response = fallback_reply(
                locale,
                user_message=user_message,
                avoid=response,
            )
        else:
            stats["used_fallback"] = 0

    if cfg.use_fallback and rubric_similarity_fail(response, state, locale):
        stats["used_fallback"] = 1
        response = fallback_reply(locale, user_message=user_message, avoid=response)

    if is_degenerate_output(response):
        stats["degenerate_retry_count"] = 1
        raw = generate_raw(model, tokenizer, system, history, user_message, cfg)
        response, _ = postprocess_model_response(raw, locale)

    if is_too_brief(
        response,
        state=state,
        user_message=user_message,
        last_assistant=last_assistant,
    ):
        stats["brief_retry_count"] = 1
        brief_suffix = "\n\nReply too brief. Reflect in 1-2 sentences, then one open question."
        raw = generate_raw(model, tokenizer, system + brief_suffix, history, user_message, cfg)
        response, _ = postprocess_model_response(raw, locale)

    if last_assistant and is_near_duplicate_reply(response, last_assistant):
        if cfg.use_fallback:
            response = fallback_reply(
                locale,
                user_message=user_message,
                avoid=last_assistant,
            )
            stats["used_fallback"] = 1

    response = ensure_open_question(
        response, state=state, reply_lang=locale, avoid=last_assistant
    )
    return response, stats


def run_case(
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    cfg: AblationConfig,
    prior_responses: dict[str, str],
) -> dict[str, Any]:
    from book_knowledge import format_rag_block, retrieve_technique_passages  # noqa: WPS433

    if cfg.use_rag:
        os.environ["DAISY_BOOK_KNOWLEDGE"] = "true"
        os.environ["DAISY_BOOK_RAG"] = "true"
    else:
        os.environ["DAISY_BOOK_RAG"] = "false"
        os.environ["DAISY_RUBRIC_JUDGE"] = "false"

    history = resolve_history(case.get("history") or [], prior_responses, case["id"])
    user_message = case["message"]
    state = case.get("state", "intake")
    locale = case.get("locale", "en")

    rag_block = ""
    if cfg.use_rag:
        passages = retrieve_technique_passages(
            user_message, state=state, reply_lang=locale, top_k=3
        )
        rag_block = format_rag_block(passages)

    system = build_system(case, cfg, rag_block=rag_block)
    raw = generate_raw(model, tokenizer, system, history, user_message, cfg)
    stats: dict[str, int] = {}

    if cfg.anti_halluc:
        from generation import postprocess_model_response  # noqa: WPS433

        response, stats = apply_anti_halluc_layers(
            raw,
            model=model,
            tokenizer=tokenizer,
            system=system,
            history=history,
            user_message=user_message,
            case=case,
            cfg=cfg,
        )
    else:
        from generation import postprocess_model_response  # noqa: WPS433

        response, _ = postprocess_model_response(raw, locale)

    return {
        "id": case["id"],
        "cluster": case.get("cluster"),
        "locale": locale,
        "message": user_message,
        "response": response.strip(),
        "config": cfg.name,
        "stats": stats,
    }


def run_config(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    cfg: AblationConfig,
) -> list[dict[str, Any]]:
    ordered = order_cases(cases)
    prior: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for case in ordered:
        t0 = time.time()
        rec = run_case(model, tokenizer, case, cfg, prior)
        rec["latency_s"] = round(time.time() - t0, 2)
        prior[case["id"]] = rec["response"]
        results.append(rec)
        print(f"  [{cfg.name}] {case['id']}: {rec['response'][:80]}…", flush=True)
    return results


def get_embed_fn():
    os.environ.setdefault("DAISY_BOOK_KNOWLEDGE", "true")
    from book_knowledge import embed_text  # noqa: WPS433

    return embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Daisy genericness ablation")
    parser.add_argument(
        "--eval",
        default=str(EVAL_DIR / "genericness_eval.jsonl"),
        help="Path to eval JSONL",
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("ADAPTER_PATH", ""),
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        default=str(EVAL_DIR / "results" / "ablation_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--configs",
        default="cumulative",
        choices=["cumulative", "decoding", "all"],
        help="Which config set to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit eval cases (0 = all)",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip configs that require LoRA (smoke test on base only)",
    )
    args = parser.parse_args()

    cases = load_eval_cases(Path(args.eval))
    if args.limit:
        cases = cases[: args.limit]

    configs: list[AblationConfig] = []
    if args.configs in ("cumulative", "all"):
        configs.extend(CUMULATIVE_CONFIGS)
    if args.configs in ("decoding", "all"):
        for temp, top_p in DECODING_SWEEP:
            configs.append(
                AblationConfig(
                    f"D1_temp{temp}_top{top_p}",
                    use_lora=True,
                    temperature=temp,
                    top_p=top_p,
                )
            )

    if args.skip_lora:
        configs = [c for c in configs if not c.use_lora]

    adapter = args.adapter or None
    needs_lora = any(c.use_lora for c in configs)
    model, tokenizer = load_model_and_tokenizer(needs_lora, adapter)

    embed_fn = get_embed_fn()
    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": BASE_MODEL,
        "adapter": adapter,
        "n_cases": len(cases),
        "configs": {},
        "summaries": [],
    }

    for cfg in configs:
        print(f"\n=== Running {cfg.name} ===", flush=True)
        records = run_config(model, tokenizer, cases, cfg)
        summary = summarize_config(cfg.name, records, embed_fn)
        summary["mean_latency_s"] = round(
            sum(r.get("latency_s", 0) for r in records) / max(len(records), 1), 2
        )
        summary["fallback_rate"] = round(
            sum(1 for r in records if r.get("stats", {}).get("used_fallback")) / max(len(records), 1),
            4,
        )
        all_results["configs"][cfg.name] = {"records": records, "summary": summary}
        all_results["summaries"].append(summary)

    cumulative_order = [c.name for c in CUMULATIVE_CONFIGS if c.name in all_results["configs"]]
    all_results["adjacent_deltas"] = adjacent_deltas(
        cumulative_order,
        {s["config"]: s for s in all_results["summaries"]},
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
