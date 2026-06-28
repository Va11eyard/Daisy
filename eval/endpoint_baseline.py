"""
Endpoint proxy ablation: measure production genericness while GPU ablation runs.

Cannot toggle components independently — measures live C4_full behavior only.
Useful as production baseline and to validate post-fix improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import summarize_config  # noqa: E402


def invoke_az(payload: dict, probe_path: Path) -> dict:
    probe_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        raise RuntimeError("Azure CLI not found")
    r = subprocess.run(
        [
            az, "ml", "online-endpoint", "invoke",
            "--name", "daisy-therapy",
            "-g", "Daisy_group",
            "-w", "Daisy",
            "--request-file", str(probe_path),
            "-o", "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    outer = json.loads(r.stdout)
    body = json.loads(outer) if isinstance(outer, str) else outer
    if isinstance(body, str):
        body = json.loads(body)
    return body


def get_embed_fn():
    repo = EVAL_DIR.parent
    sys.path.insert(0, str(repo / "inference"))
    os.environ.setdefault("DAISY_BOOK_KNOWLEDGE", "true")
    from book_knowledge import embed_text  # noqa: WPS433
    return embed_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(EVAL_DIR / "genericness_eval.jsonl"))
    parser.add_argument("--output", default=str(EVAL_DIR / "results" / "endpoint_baseline.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[: args.limit]

    probe = Path(tempfile.gettempdir()) / "daisy_ablation_probe.json"
    records: list[dict] = []
    prior: dict[str, str] = {}

    for case in cases:
        history = []
        for m in case.get("history") or []:
            content = m["content"]
            if content == "PLACEHOLDER":
                dep = case.get("depends_on", case["id"])
                content = prior.get(dep, "")
            history.append({"role": m["role"], "content": content})

        payload = {
            "message": case["message"],
            "locale": case.get("locale", "en"),
            "history": history,
            "max_tokens": 256,
        }
        t0 = time.time()
        body = invoke_az(payload, probe)
        resp = (body.get("response") or "").strip()
        dbg = body.get("debug_context") or {}
        prior[case["id"]] = resp
        records.append({
            "id": case["id"],
            "cluster": case.get("cluster"),
            "locale": case.get("locale", "en"),
            "message": case["message"],
            "response": resp,
            "config": "C4_endpoint_live",
            "voice_retry_count": dbg.get("voice_retry_count"),
            "brief_retry_count": dbg.get("brief_retry_count"),
            "total_regen_count": dbg.get("total_regen_count"),
            "max_regen_budget": dbg.get("max_regen_budget"),
            "fallback_suspected": dbg.get("output_sanitized"),
            "latency_s": round(time.time() - t0, 2),
        })
        print(f"  {case['id']}: {resp[:90].encode('ascii', 'replace').decode()}", flush=True)

    embed_fn = get_embed_fn()
    summary = summarize_config("C4_endpoint_live", records, embed_fn)
    out = {"config": "C4_endpoint_live", "records": records, "summary": summary}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"  genericness={summary['cross_scenario_genericness']} canned={summary['canned_rate']}")


if __name__ == "__main__":
    main()
