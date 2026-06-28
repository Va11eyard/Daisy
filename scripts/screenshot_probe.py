"""Reproduce tester screenshot thread against live endpoint."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "_probe_turn.json"

MSGS = [
    "HI HOW ARE YOU",
    (
        "im feeling weird you know. i recently broke up w my bf and i cant stop thinking about him. "
        "at first i was mad but now i feellike i miss him"
    ),
    "emptiness in my heart area",
]

BAD_MARKERS = (
    "what do you notice in your body",
    "that must feel very empty and poignant",
    "it sounds hard to put into words right now",
    "i'm sorry to hear that",
    "it sounds like you're dealing with",
)

META_MSG = "You are answering too short"
META_BAD_HISTORY = [
    {"role": "user", "content": "work has been overwhelming lately"},
    {"role": "assistant", "content": "That must be incredibly stressful."},
]


def invoke(payload: dict) -> dict:
    PROBE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        raise RuntimeError("Azure CLI not found")
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
    issues: list[str] = []
    for i, msg in enumerate(MSGS, 1):
        body = invoke({"message": msg, "locale": "en", "history": history, "max_tokens": 256})
        resp = body.get("response", "")
        dbg = body.get("debug_context", {})
        print(f"=== TURN {i} state={dbg.get('daisy_state')} brief={dbg.get('brief_retry_count')} ===")
        print(f"USER: {msg}")
        print(f"DAISY: {resp}\n")
        low = resp.lower()
        for bad in BAD_MARKERS:
            if bad in low:
                issues.append(f"turn {i}: contains banned marker {bad!r}")
        if i >= 2 and "?" not in resp:
            issues.append(f"turn {i}: no question")
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": resp})

    print("=== META-FEEDBACK TURN ===")
    body = invoke(
        {
            "message": META_MSG,
            "locale": "en",
            "history": history + META_BAD_HISTORY,
            "max_tokens": 256,
        }
    )
    resp = body.get("response", "")
    dbg = body.get("debug_context", {})
    print(f"state={dbg.get('daisy_state')} brief={dbg.get('brief_retry_count')} voice={dbg.get('voice_retry_count')}")
    print(f"USER: {META_MSG}")
    print(f"DAISY: {resp}\n")
    low = resp.lower()
    for bad in BAD_MARKERS:
        if bad in low:
            issues.append(f"meta: contains banned marker {bad!r}")
    if "?" not in resp:
        issues.append("meta: no question")
    if len(resp) < 40:
        issues.append(f"meta: reply too short ({len(resp)} chars)")

    if issues:
        print("ISSUES:")
        for x in issues:
            print(" ", x)
        sys.exit(1)
    print("OK: screenshot probe passed")


if __name__ == "__main__":
    main()
