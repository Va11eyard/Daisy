"""Evaluate v13 + inference v9 quality/latency gates (see docs/BASE_MODEL_GATE.md)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "results" / "v13_gate_report.json"),
    )
    parser.add_argument("--skip-endpoint", action="store_true")
    args = parser.parse_args()

    # Offline dataset gate
    rc = _run([sys.executable, "eval/analyze_offline.py", "--train", "data/train_v13.jsonl"])
    audit_path = ROOT / "eval" / "results" / "offline_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    lora = audit.get("lora_data_audit") or {}

    report: dict = {
        "dataset_gate": {
            "reflect_plus_question_shape_rate": lora.get("reflect_plus_question_shape_rate"),
            "target_max": 0.70,
            "pass": (lora.get("reflect_plus_question_shape_rate") or 1.0) <= 0.70,
        },
        "base_model_swap": {
            "required": False,
            "reason": "Run after v13 deploy; set required=true only if gates fail",
            "candidates": [
                "Qwen/Qwen2.5-3B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ],
        },
    }

    if not args.skip_endpoint:
        rc_verify = _run([sys.executable, "scripts/verify_deploy.py"])
        report["verify_deploy_pass"] = rc_verify == 0
        rc_lat = _run([sys.executable, "scripts/latency_probe.py"])
        report["latency_probe_rc"] = rc_lat

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if report["dataset_gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
