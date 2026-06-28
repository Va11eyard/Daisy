"""Single-message live checks."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def invoke(path: Path) -> dict:
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
            str(path),
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
    for label, name in (("RU", "_check_ru.json"), ("EN", "_check_en.json")):
        body = invoke(ROOT / "scripts" / name)
        dbg = body.get("debug_context", {})
        print(f"=== {label} ===")
        print("state:", dbg.get("daisy_state"))
        print("brief_retry:", dbg.get("brief_retry_count"))
        print("degenerate_retry:", dbg.get("degenerate_retry_count"))
        print("response:", body.get("response", ""))
        print()


if __name__ == "__main__":
    main()
