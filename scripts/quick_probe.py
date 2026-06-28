"""Quick live probes for common failure messages."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "_probe_turn.json"

MSGS = (
    "hello still feeling sad",
    "HI HOW ARE YOU",
)


def invoke(msg: str) -> dict:
    PROBE.write_text(
        json.dumps({"message": msg, "locale": "en", "history": [], "max_tokens": 256}),
        encoding="utf-8",
    )
    az = shutil.which("az") or shutil.which("az.cmd")
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
    for msg in MSGS:
        body = invoke(msg)
        dbg = body.get("debug_context", {})
        resp = body.get("response", "")
        print(f"USER: {msg}")
        print(f"DAISY: {resp}")
        print(f"  state={dbg.get('daisy_state')} brief={dbg.get('brief_retry_count')}")
        print()


if __name__ == "__main__":
    main()
