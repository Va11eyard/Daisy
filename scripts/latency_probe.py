"""Quick latency probe against daisy-therapy endpoint (requires az CLI login)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "_probe_turn.json"

MESSAGES = [
    "я чувствую себя плохо",
    "мне тревожно на работе",
    "не знаю что делать",
]


def invoke(payload: dict) -> dict:
    PROBE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        raise RuntimeError("Azure CLI (az) not found in PATH")
    r = subprocess.run(
        [
            az,
            "ml",
            "online-endpoint",
            "invoke",
            "--name",
            "daisy-therapy",
            "-g",
            "Daisy_group",
            "-w",
            "Daisy",
            "--request-file",
            str(PROBE),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    outer = json.loads(r.stdout)
    body = json.loads(outer) if isinstance(outer, str) else outer
    if isinstance(body, str):
        body = json.loads(body)
    return body


def main() -> None:
    history: list[dict[str, str]] = []
    times: list[float] = []
    for msg in MESSAGES:
        payload = {
            "message": msg,
            "locale": "ru",
            "history": history,
            "max_tokens": 220,
        }
        t0 = time.perf_counter()
        body = invoke(payload)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        resp = body.get("response", "")
        dbg = body.get("debug_context", {})
        print(
            f"turn={len(times)} sec={elapsed:.1f} "
            f"voice={dbg.get('voice_retry_count')} brief={dbg.get('brief_retry_count')} "
            f"total_regen={dbg.get('total_regen_count')} budget={dbg.get('max_regen_budget')} "
            f"cjk={dbg.get('cjk_retry_count')} degen={dbg.get('degenerate_retry_count')}"
        )
        print("  resp:", resp[:140].replace("\n", " "))
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": resp})
    print(f"max_sec={max(times):.1f} avg_sec={sum(times) / len(times):.1f}")


if __name__ == "__main__":
    main()
