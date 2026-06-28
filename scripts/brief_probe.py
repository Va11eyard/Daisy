"""Verify replies are not hollow one-liners (tester screenshot scenarios)."""
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

HOLLOW = (
    re.compile(r"^i'?m sorry to hear that\.?\s*$", re.I),
    re.compile(r"^i'?m sad to hear that\.?\s*$", re.I),
    re.compile(r"^я понимаю, что сейчас (?:вам|тебе) очень сложно\.?\s*$", re.I),
    re.compile(r"^that must (?:feel|be) very .{0,40}\.\s*$", re.I),
)


def sentence_count(text: str) -> int:
    return len([p for p in re.split(r"[.!?…]+", text.strip()) if p.strip()])


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


def check(label: str, resp: str, *, min_sentences: int = 2) -> list[str]:
    issues: list[str] = []
    for pat in HOLLOW:
        if pat.match(resp.strip()):
            issues.append(f"{label}: hollow one-liner")
    if sentence_count(resp) < min_sentences:
        issues.append(f"{label}: fewer than {min_sentences} sentences")
    if "?" not in resp:
        issues.append(f"{label}: no question")
    return issues


def main() -> None:
    issues: list[str] = []

    ru_msg = (
        "я не знаю что делать, я запуталась, я чувствую напряжение - "
        "так как будто ничего мне не помогает"
    )
    t0 = time.perf_counter()
    body = invoke({"message": ru_msg, "locale": "ru", "history": [], "max_tokens": 256})
    elapsed = time.perf_counter() - t0
    resp = body.get("response", "")
    dbg = body.get("debug_context", {})
    print(f"RU sec={elapsed:.1f} state={dbg.get('daisy_state')} brief={dbg.get('brief_retry_count')}")
    print(" ", resp[:160].replace("\n", " "))
    issues.extend(check("RU", resp))
    if re.search(r"\b[Вв]ы\b", resp) and "Вы" not in ru_msg:
        issues.append("RU: formal Вы instead of ty")

    history: list[dict[str, str]] = []
    en_msgs = ["hey u here?", "broke up w my bf", "yea...", "well shit happens", "yea"]
    for i, msg in enumerate(en_msgs, 1):
        t0 = time.perf_counter()
        body = invoke({"message": msg, "locale": "en", "history": history, "max_tokens": 256})
        elapsed = time.perf_counter() - t0
        resp = body.get("response", "")
        dbg = body.get("debug_context", {})
        print(f"EN turn={i} sec={elapsed:.1f} state={dbg.get('daisy_state')} brief={dbg.get('brief_retry_count')}")
        print(" ", resp[:160].replace("\n", " "))
        if dbg.get("daisy_state") in ("disclosure", "action_planning"):
            issues.extend(check(f"EN-{i}", resp))
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": resp})

    breakup_msg = (
        "My boyfriend and I broke up last week after three years. "
        "I miss him so much and I don't know how to get through this."
    )
    t0 = time.perf_counter()
    body = invoke({"message": breakup_msg, "locale": "en", "history": [], "max_tokens": 256})
    elapsed = time.perf_counter() - t0
    resp1 = body.get("response", "")
    dbg1 = body.get("debug_context", {})
    print(f"SCREEN turn=1 sec={elapsed:.1f} state={dbg1.get('daisy_state')} brief={dbg1.get('brief_retry_count')}")
    print(" ", resp1[:160].replace("\n", " "))
    issues.extend(check("SCREEN-1", resp1))
    if "what do you notice in your body" in resp1.lower():
        issues.append("SCREEN-1: generic body-scan fallback on breakup thread")

    screen_history = [
        {"role": "user", "content": breakup_msg},
        {"role": "assistant", "content": resp1},
    ]
    follow_up = "emptiness in my heart area"
    t0 = time.perf_counter()
    body = invoke(
        {"message": follow_up, "locale": "en", "history": screen_history, "max_tokens": 256}
    )
    elapsed = time.perf_counter() - t0
    resp2 = body.get("response", "")
    dbg2 = body.get("debug_context", {})
    print(f"SCREEN turn=2 sec={elapsed:.1f} state={dbg2.get('daisy_state')} brief={dbg2.get('brief_retry_count')}")
    print(" ", resp2[:160].replace("\n", " "))
    issues.extend(check("SCREEN-2", resp2))

    if issues:
        print("ISSUES:")
        for x in issues:
            print(" ", x)
        sys.exit(1)
    print("OK: brief-reply probe passed")


if __name__ == "__main__":
    main()
