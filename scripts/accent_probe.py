"""8-turn probe: no acute accents, no degenerate punctuation."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "_probe_turn.json"

MESSAGES = [
    "привет",
    "я чувствую себя странно",
    "ВСВВСВСВ",
    "и че",
    "я не знаю",
    "может тревога",
    "не могу уснуть",
    "спасибо",
]

ACUTE = re.compile(r"[\u00b4\u02c6]")
DEGEN = re.compile(r"(\.\s*){3,}|\.\.\s*\.\s*\?")


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
    issues: list[str] = []
    for msg in MESSAGES:
        payload = {"message": msg, "locale": "ru", "history": history, "max_tokens": 220}
        t0 = time.perf_counter()
        body = invoke(payload)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        resp = body.get("response", "")
        dbg = body.get("debug_context", {})
        if ACUTE.search(resp):
            issues.append(f"acute in turn {len(times)}: {resp[:80]}")
        if DEGEN.search(resp):
            issues.append(f"degen punct turn {len(times)}: {resp[:80]}")
        print(
            f"turn={len(times)} sec={elapsed:.1f} sanitized={dbg.get('output_sanitized')} "
            f"repeat={dbg.get('repeat_retry_count')}"
        )
        print("  resp:", resp[:120].replace("\n", " "))
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": resp})
    print(f"max_sec={max(times):.1f} avg_sec={sum(times)/len(times):.1f}")
    if issues:
        print("ISSUES:", *issues, sep="\n  ")
        sys.exit(1)
    print("OK: no acute/degen in 8-turn probe")


if __name__ == "__main__":
    main()
