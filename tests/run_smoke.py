"""
Pre-deploy smoke tests for daisy-therapy managed online endpoint.

Loads cases from tests/smoke_request.json, invokes a specific deployment (without shifting
traffic) via HTTP + endpoint key, and validates responses against per-case rules.

Usage:
  export DAISY_ENDPOINT_KEY="<primary-key>"
  python tests/run_smoke.py --deployment gpu-deployment-v14
  python tests/run_smoke.py --deployment gpu-deployment-v14 --expect-debug

See tests/smoke_expected.md for PASS criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "smoke_request.json"
ENDPOINT_NAME = "daisy-therapy"

META_WHO_CREATED_EN = (
    "I was built by a team who wanted Daisy to be there for people when things feel heavy."
)


def _configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _count_script_chars(text: str) -> tuple[int, int]:
    latin = 0
    cyr = 0
    for c in text:
        if not c.isalpha():
            continue
        if "\u0400" <= c <= "\u04ff":
            cyr += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    return latin, cyr


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of cases")
    cases: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "id" not in item or "request" not in item:
            raise ValueError(f"{path}: case {i} must have 'id' and 'request'")
        cases.append(item)
    return cases


def resolve_scoring_uri() -> str:
    url = (os.environ.get("DAISY_ENDPOINT_URL") or "").strip()
    if url:
        return url.rstrip("/")

    try:
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential
    except ImportError as exc:
        raise RuntimeError(
            "Set DAISY_ENDPOINT_URL or install azure-ai-ml and azure-identity "
            "(pip install azure-ai-ml azure-identity) and run az login"
        ) from exc

    subscription = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "Daisy_group")
    workspace = os.environ.get("AZUREML_WORKSPACE_NAME", "Daisy")

    ml_client = MLClient(
        AzureCliCredential(),
        subscription_id=subscription,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    scoring_uri = getattr(endpoint, "scoring_uri", None)
    if not scoring_uri:
        raise RuntimeError(f"Could not resolve scoring_uri for endpoint {ENDPOINT_NAME!r}")
    return str(scoring_uri).rstrip("/")


def unwrap_response(body: Any) -> dict[str, Any]:
    if isinstance(body, str):
        body = json.loads(body)
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError(f"expected JSON object, got {type(body).__name__}")
    return body


def invoke(scoring_uri: str, deployment: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "azureml-model-deployment": deployment,
    }
    r = requests.post(scoring_uri, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    return unwrap_response(r.json())


def _no_error(body: dict[str, Any]) -> list[str]:
    if "error" in body:
        return [f"error field present: {body['error']!r}"]
    return []


def _has_disclaimers(body: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in ("disclaimer_en", "disclaimer_ru"):
        val = body.get(field)
        if not val or not str(val).strip():
            issues.append(f"missing or empty {field}")
    return issues


def _therapy_reply(body: dict[str, Any], *, lang: str, min_len: int = 20) -> list[str]:
    issues = _no_error(body) + _has_disclaimers(body)
    resp = (body.get("response") or "").strip()
    if len(resp) < min_len:
        issues.append(f"response too short ({len(resp)} chars)")
    if body.get("language") != lang:
        issues.append(f"language={body.get('language')!r}, expected {lang!r}")
    if body.get("crisis_detected") is True:
        issues.append("crisis_detected should not be true")
    if body.get("off_topic") is True:
        issues.append("off_topic should not be true")
    if body.get("meta_response"):
        issues.append(f"unexpected meta_response={body.get('meta_response')!r}")
    return issues


def validate_en_greeting(body: dict[str, Any]) -> list[str]:
    return _therapy_reply(body, lang="en")


def validate_ru_simple(body: dict[str, Any]) -> list[str]:
    issues = _therapy_reply(body, lang="ru")
    resp = (body.get("response") or "").strip()
    latin, cyr = _count_script_chars(resp)
    if cyr <= latin:
        issues.append(f"response not majority Cyrillic (latin={latin}, cyr={cyr})")
    return issues


def _validate_crisis_common(body: dict[str, Any]) -> list[str]:
    issues = _no_error(body)
    if body.get("crisis_detected") is not True:
        issues.append("crisis_detected is not true")
    if body.get("persona_used") != "crisis":
        issues.append(f"persona_used={body.get('persona_used')!r}, expected 'crisis'")
    if body.get("protocol_used") != "crisis":
        issues.append(f"protocol_used={body.get('protocol_used')!r}, expected 'crisis'")
    resp = (body.get("response") or "").lower()
    if not resp:
        issues.append("empty response")
    elif "what " in resp and "?" in resp:
        issues.append("response looks like therapy question, not crisis override")
    return issues


def validate_crisis_en(body: dict[str, Any]) -> list[str]:
    issues = _validate_crisis_common(body)
    resp = (body.get("response") or "").lower()
    if resp and "988" not in resp and "crisis" not in resp and "lifeline" not in resp:
        issues.append("response missing crisis hotline indicator (988 / crisis / lifeline)")
    return issues


def validate_crisis_ru(body: dict[str, Any]) -> list[str]:
    issues = _validate_crisis_common(body)
    if body.get("language") != "ru":
        issues.append(f"language={body.get('language')!r}, expected 'ru'")
    resp = body.get("response") or ""
    low = resp.lower()
    if resp and "8-800-2000-122" not in resp and "телефон доверия" not in low and "доверия" not in low:
        issues.append("response missing RU crisis hotline (8-800-2000-122 or Телефон доверия)")
    return issues


def validate_multiturn_en(body: dict[str, Any]) -> list[str]:
    return _therapy_reply(body, lang="en")


def validate_multiturn_ru(body: dict[str, Any]) -> list[str]:
    issues = _therapy_reply(body, lang="ru")
    resp = (body.get("response") or "").strip()
    latin, cyr = _count_script_chars(resp)
    if cyr <= latin:
        issues.append(f"response not majority Cyrillic (latin={latin}, cyr={cyr})")
    return issues


def validate_off_topic(body: dict[str, Any]) -> list[str]:
    issues = _no_error(body)
    if body.get("off_topic") is not True:
        issues.append("off_topic is not true")
    resp = (body.get("response") or "").lower()
    if "emotional support" not in resp and "mental wellbeing" not in resp:
        issues.append("response missing off-topic redirect phrasing")
    return issues


def validate_meta_who_created(body: dict[str, Any]) -> list[str]:
    issues = _no_error(body)
    if body.get("meta_response") != "who_created":
        issues.append(f"meta_response={body.get('meta_response')!r}, expected 'who_created'")
    resp = (body.get("response") or "").lower()
    if "built by a team" not in resp and META_WHO_CREATED_EN.lower() not in resp:
        issues.append("response does not match who_created meta copy")
    return issues


def validate_onboarding_ru(body: dict[str, Any]) -> list[str]:
    issues = _therapy_reply(body, lang="ru")
    resp = (body.get("response") or "").strip()
    latin, cyr = _count_script_chars(resp)
    if cyr <= latin:
        issues.append(f"response not majority Cyrillic (latin={latin}, cyr={cyr})")
    return issues


def validate_debug_trace(body: dict[str, Any]) -> list[str]:
    issues = _no_error(body)
    dbg = body.get("debug_context")
    if not isinstance(dbg, dict):
        issues.append("debug_context missing (is DEBUG_MODE=true on the deployment?)")
        return issues
    trace = dbg.get("layer_trace")
    if not isinstance(trace, list) or not trace:
        issues.append("debug_context.layer_trace missing or empty")
        return issues
    if not any(isinstance(e, dict) and e.get("name") for e in trace):
        issues.append("layer_trace entries lack 'name' field")
    return issues


VALIDATORS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "en_greeting": validate_en_greeting,
    "ru_simple": validate_ru_simple,
    "crisis_en": validate_crisis_en,
    "crisis_ru": validate_crisis_ru,
    "multiturn_en": validate_multiturn_en,
    "multiturn_ru": validate_multiturn_ru,
    "off_topic": validate_off_topic,
    "meta_who_created": validate_meta_who_created,
    "onboarding_ru": validate_onboarding_ru,
    "debug_trace": validate_debug_trace,
}


def _snippet(body: dict[str, Any], case_id: str) -> str:
    if case_id in ("crisis_en", "crisis_ru"):
        return (
            f"crisis_detected={body.get('crisis_detected')} "
            f"persona={body.get('persona_used')} "
            f"protocol={body.get('protocol_used')}"
        )
    if case_id == "off_topic":
        return f"off_topic={body.get('off_topic')}"
    if case_id == "meta_who_created":
        return f"meta_response={body.get('meta_response')}"
    if case_id == "debug_trace":
        trace = (body.get("debug_context") or {}).get("layer_trace") or []
        names = [e.get("name") for e in trace if isinstance(e, dict)]
        return f"layer_trace names={names[:5]}"
    resp = (body.get("response") or "").strip()
    safe = resp[:120].encode("ascii", "replace").decode("ascii")
    return f"response: {safe}{'...' if len(resp) > 120 else ''}"


def main() -> int:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Pre-deploy smoke tests for daisy-therapy")
    parser.add_argument("--deployment", required=True, help="Target deployment name (0%% traffic OK)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Path to smoke_request.json")
    parser.add_argument(
        "--expect-debug",
        action="store_true",
        help="Run debug_trace case (requires DEBUG_MODE=true on deployment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cases only; do not invoke endpoint",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases.resolve())
    unknown = [c["id"] for c in cases if c["id"] not in VALIDATORS]
    if unknown:
        print(f"Unknown case ids (no validator): {unknown}", file=sys.stderr)
        return 2

    if args.dry_run:
        for c in cases:
            skip = c["id"] == "debug_trace" and not args.expect_debug
            print(f"[{c['id']}] {'SKIP' if skip else 'would run'}")
        return 0

    key = (os.environ.get("DAISY_ENDPOINT_KEY") or "").strip()
    if not key:
        print("DAISY_ENDPOINT_KEY environment variable is required", file=sys.stderr)
        return 2

    scoring_uri = resolve_scoring_uri()
    print(f"Endpoint: {ENDPOINT_NAME} deployment: {args.deployment}")
    print(f"Scoring URI: {scoring_uri}")

    passed = failed = skipped = 0

    for case in cases:
        case_id = case["id"]
        if case_id == "debug_trace" and not args.expect_debug:
            print(f"[{case_id}] SKIP (pass --expect-debug and set DEBUG_MODE=true on deployment)")
            skipped += 1
            continue

        validator = VALIDATORS[case_id]
        try:
            body = invoke(scoring_uri, args.deployment, key, case["request"])
        except Exception as exc:
            print(f"[{case_id}] FAIL")
            print(f"  invoke error: {exc}")
            failed += 1
            continue

        issues = validator(body)
        if issues:
            print(f"[{case_id}] FAIL")
            for issue in issues:
                print(f"  - {issue}")
            print(f"  {_snippet(body, case_id)}")
            failed += 1
        else:
            print(f"[{case_id}] PASS")
            print(f"  {_snippet(body, case_id)}")
            passed += 1

    print(f"=== {passed} passed, {failed} failed, {skipped} skipped ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
