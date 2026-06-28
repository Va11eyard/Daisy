"""A/B probe: compare two model versions or endpoints on the same test cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_deploy import TESTS  # noqa: E402


def invoke(url: str, key: str, payload: dict) -> dict:
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    if isinstance(body, str):
        body = json.loads(body)
    return body


def score_response(resp: str, t: dict) -> list[str]:
    issues: list[str] = []
    low = resp.lower()
    for sub in t.get("forbid_substrings", []):
        if sub in low:
            issues.append(f"forbidden:{sub}")
    if t.get("require_question") and "?" not in resp:
        issues.append("no_question")
    if t.get("min_len") and len(resp) < t["min_len"]:
        issues.append("too_short")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-a", default=os.environ.get("AI_API_URL", ""))
    parser.add_argument("--key-a", default=os.environ.get("AI_API_KEY", ""))
    parser.add_argument("--label-a", default="v11")
    parser.add_argument("--url-b", default="")
    parser.add_argument("--key-b", default="")
    parser.add_argument("--label-b", default="v12")
    args = parser.parse_args()

    if not args.url_a or not args.key_a:
        raise SystemExit("Set AI_API_URL and AI_API_KEY or pass --url-a/--key-a")

    compare_b = bool(args.url_b and args.key_b)

    for t in TESTS:
        payload = {
            "message": t["message"],
            "locale": "en",
            "history": t.get("history", []),
            "max_tokens": 256,
        }
        body_a = invoke(args.url_a.rstrip("/"), args.key_a, payload)
        resp_a = (body_a.get("response") or "").strip()
        issues_a = score_response(resp_a, t)
        print(f"\n[{t['label']}] {args.label_a}: {resp_a[:100]}...")
        print(f"  issues: {issues_a or 'none'}")
        dbg = body_a.get("debug_context") or {}
        print(f"  voice_retry={dbg.get('voice_retry_count')} brief_retry={dbg.get('brief_retry_count')}")

        if compare_b:
            body_b = invoke(args.url_b.rstrip("/"), args.key_b, payload)
            resp_b = (body_b.get("response") or "").strip()
            issues_b = score_response(resp_b, t)
            print(f"  {args.label_b}: {resp_b[:100]}...")
            print(f"  issues: {issues_b or 'none'}")


if __name__ == "__main__":
    main()
