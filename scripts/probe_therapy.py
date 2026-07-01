"""Post-deploy probe for therapeutic depth — NOT used in production chat.

Qualitative checks: specificity and engagement, not mandatory question marks.

  python scripts/probe_therapy.py
  python scripts/probe_therapy.py --case anxiety_en --debug
  python scripts/probe_therapy.py --delay 3

Single-GPU endpoint: use --delay between cases to avoid connection resets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from reply_language import generation_has_script_leak  # noqa: E402

URI = "https://daisy-therapy.westus2.inference.ml.azure.com/score"
DEPLOYMENT = "gpu-deployment-finetuned"
EXPECTED_BUILD = "2026-07-lora-v11-natural4-ru"
PROBE_DELAY_SEC = 2.0
MAX_RETRIES = 3

CANNED_GREETING = "hey im glad youre here whats been on your mind lately"
FALLBACK_PHRASE = "i'm here — could you say a bit more"
RUBRIC_PHRASES = (
    "bracing myself",
    "one concrete thing",
    "in plain language",
    "one way to phrase it",
)

HOLLOW_OPENERS = (
    "it sounds like you're",
    "it sounds like you are",
    "i'm sorry you're feeling",
    "im sorry you're feeling",
    "i'm sorry you are feeling",
    "that must be really",
    "that must have felt",
    "of course—i'd love to help",
    "of course - i'd love to help",
    "that can be so helpful to sort",
)

ANXIETY_TURN1_SAMPLE = (
    "Thanks for sharing that. What has been going on for you today?"
)

ANXIETY_KEYWORDS = (
    "anxious",
    "anxiety",
    "worry",
    "worried",
    "nervous",
    "feel",
    "feeling",
    "today",
    "right now",
    "moment",
    "body",
    "chest",
    "mind",
)

WORK_KEYWORDS = (
    "ceo",
    "work",
    "boss",
    "site",
    "product",
    "consequence",
    "pressure",
    "authority",
    "office",
    "mistake",
    "yell",
    "rage",
    "breaking",
)

STRESS_KEYWORDS = ("stress", "stressed", "overwhelm", "pressure", "weigh", "tension", "load")
CLARITY_KEYWORDS = ("thought", "sort", "clarity", "untangle", "mind", "explore", "figure")

CASES: list[dict] = [
    {
        "id": "stress_en",
        "locale": "en",
        "message": "Let's talk about stress",
        "history": [],
        "max_tokens": None,
        "keywords": STRESS_KEYWORDS,
    },
    {
        "id": "anxiety_en",
        "locale": "en",
        "message": "I'm feeling anxious today",
        "history": [],
        "max_tokens": None,
        "keywords": ANXIETY_KEYWORDS,
    },
    {
        "id": "ceo_en",
        "locale": "en",
        "message": "My CEO is raging at me for breaking the site",
        "history": [],
        "max_tokens": None,
        "keywords": WORK_KEYWORDS,
    },
    {
        "id": "anxiety_en_256",
        "locale": "en",
        "message": "I'm feeling anxious today",
        "history": [],
        "max_tokens": 256,
        "keywords": ANXIETY_KEYWORDS,
    },
    {
        "id": "ceo_en_256",
        "locale": "en",
        "message": "My CEO is raging at me for breaking the product",
        "history": [],
        "max_tokens": 256,
        "keywords": WORK_KEYWORDS,
    },
    {
        "id": "multiturn_anxiety_then_ceo",
        "locale": "en",
        "message": "My CEO is raging at me for breaking the product",
        "history": [
            {"role": "user", "content": "I'm feeling anxious today"},
            {"role": "assistant", "content": ANXIETY_TURN1_SAMPLE},
        ],
        "max_tokens": None,
        "keywords": WORK_KEYWORDS,
    },
    {
        "id": "multiturn_after_canned",
        "locale": "en",
        "message": "My CEO is raging at me for breaking the site",
        "history": [
            {"role": "user", "content": "I'm feeling anxious today"},
            {
                "role": "assistant",
                "content": "Hey — I'm glad you're here. What's been on your mind lately?",
            },
        ],
        "max_tokens": None,
        "keywords": WORK_KEYWORDS,
    },
    {
        "id": "stress_ru",
        "locale": "ru",
        "message": "Давай поговорим о стрессе",
        "history": [],
        "max_tokens": None,
        "keywords": ("стресс", "напряж", "устал", "давлен", "тяжел", "чувств"),
    },
    {
        "id": "anxiety_ru",
        "locale": "ru",
        "message": "привет, сегодня мне не хорошо",
        "history": [],
        "max_tokens": None,
        "keywords": ("тревог", "плох", "тяжел", "чувств", "беспоко", "на уме", "сегодня"),
    },
    {
        "id": "morning_ru",
        "locale": "ru",
        "message": "сегодня я проснулась тревожно",
        "history": [],
        "max_tokens": None,
        "user_gender": "female",
        "keywords": ("утр", "тревог", "просн", "чувств", "сон", "беспоко"),
    },
    {
        "id": "clarity_en",
        "locale": "en",
        "message": "Help me sort out my thoughts",
        "history": [],
        "max_tokens": None,
        "keywords": CLARITY_KEYWORDS,
        "forbid_echo": ("sort out my thoughts", "sort through your thoughts"),
    },
]


def _normalize(text: str) -> str:
    t = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def _is_canned(text: str) -> bool:
    return _normalize(text) == CANNED_GREETING or _normalize(text).startswith(CANNED_GREETING)


def _has_rubric_phrase(text: str) -> str | None:
    low = text.lower()
    for phrase in RUBRIC_PHRASES:
        if phrase in low:
            return phrase
    if FALLBACK_PHRASE in _normalize(text):
        return FALLBACK_PHRASE
    return None


def _hollow_opener(text: str) -> str | None:
    low = text.lower().strip()
    for opener in HOLLOW_OPENERS:
        if low.startswith(opener):
            return opener
    return None


def _echoes_user(case: dict, resp: str) -> bool:
    msg = _normalize(case.get("message") or "")
    resp_n = _normalize(resp)
    if not msg or len(msg) < 12:
        return False
    msg_words = set(msg.split())
    resp_words = set(resp_n.split())
    overlap = len(msg_words & resp_words) / max(len(msg_words), 1)
    if overlap >= 0.55 and len(resp_n) < len(msg) + 40:
        return True
    for fragment in case.get("forbid_echo") or ():
        if fragment.lower() in resp.lower():
            return True
    return False


def _prior_assistant(case: dict) -> str:
    for msg in reversed(case.get("history") or []):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def _has_specificity(case: dict, resp: str) -> bool:
    keywords = case.get("keywords") or ()
    low = resp.lower()
    return any(k in low for k in keywords)


def _latin_leak(case: dict, resp: str) -> bool:
    if case.get("locale") not in ("ru", "kk"):
        return False
    return generation_has_script_leak(resp, case["locale"])


def _evaluate(case: dict, resp: str) -> tuple[bool, str]:
    sentences = [s for s in re.split(r"[.!?…]+", resp) if s.strip()]
    if _latin_leak(case, resp):
        return False, "latin_script_leak"

    rubric = _has_rubric_phrase(resp)
    if rubric:
        return False, f"rubric_phrase={rubric!r}"

    hollow = _hollow_opener(resp)
    if hollow:
        return False, f"hollow_opener={hollow!r}"

    if _echoes_user(case, resp):
        return False, "echoes_user"

    canned = _is_canned(resp)
    if canned:
        return False, "canned_greeting"

    if len(resp.strip()) < 45:
        return False, f"too_short len={len(resp)}"

    if case["id"].startswith("multiturn"):
        prior = _prior_assistant(case)
        dup = bool(prior and _normalize(resp) == _normalize(prior))
        if dup:
            return False, "dup_prior_assistant"
        ok = _has_specificity(case, resp) and len(sentences) >= 1
        reason = f"specific={ok} sentences={len(sentences)} len={len(resp)}"
        return ok, reason

    substantive = len(sentences) >= 2 or (len(sentences) == 1 and len(resp) >= 80)
    specific = _has_specificity(case, resp)
    ok = substantive and specific
    reason = f"substantive={substantive} specific={specific} sentences={len(sentences)} len={len(resp)}"
    return ok, reason


def _post_case(
    headers: dict,
    payload: dict,
    *,
    retries: int = MAX_RETRIES,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(URI, headers=headers, json=payload, timeout=300)
            r.raise_for_status()
            body = r.json()
            if isinstance(body, str):
                body = json.loads(body)
            return body
        except (RequestsConnectionError, ConnectionResetError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(PROBE_DELAY_SEC * (attempt + 1))
    raise last_err or RuntimeError("probe request failed")


def _run_deployment(
    deployment: str,
    key: str,
    cases: list[dict],
    *,
    use_debug: bool,
    delay_sec: float,
    expected_build: str,
) -> int:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "azureml-model-deployment": deployment,
    }
    print(f"\n=== deployment: {deployment} ===\n")
    failed = 0
    meta_ok = True
    replies: dict[str, str] = {}

    for i, case in enumerate(cases):
        if i > 0 and delay_sec > 0:
            time.sleep(delay_sec)
        payload: dict = {
            "message": case["message"],
            "locale": case["locale"],
            "history": case["history"],
        }
        if case.get("max_tokens") is not None:
            payload["max_tokens"] = case["max_tokens"]
        if case.get("user_gender"):
            payload["user_gender"] = case["user_gender"]
        if use_debug:
            payload["debug"] = True

        body = _post_case(headers, payload)
        resp = (body.get("response") or "").strip()
        replies[case["id"]] = resp
        ok, reason = _evaluate(case, resp)
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        build = body.get("inference_build", "?")
        mode = body.get("inference_mode", "?")
        adapter = body.get("adapter_loaded")
        dbg = body.get("debug_context") or {}
        raw = dbg.get("model_raw_text")
        trim_amp = dbg.get("trim_amputated")
        extra = ""
        if raw:
            extra = f" raw_len={len(raw)} trim_amputated={trim_amp}"
            if trim_amp:
                print(f"  [trim] raw: {raw[:240]}")
        print(f"[{status}] {case['id']} ({reason} build={build} mode={mode} adapter={adapter}{extra})")
        print(f"  {resp}\n")

        if build != expected_build:
            meta_ok = False
        if adapter is not True:
            meta_ok = False
        if mode != "simple":
            meta_ok = False

    anxiety = _normalize(replies.get("anxiety_en", ""))
    ceo = _normalize(replies.get("ceo_en", ""))
    if anxiety and ceo and anxiety == ceo:
        print("[FAIL] anxiety_en and ceo_en produced identical replies")
        failed += 1
    elif anxiety and ceo:
        print("[OK] anxiety_en vs ceo_en: distinct replies")

    if not meta_ok:
        print(f"[WARN] expected build={expected_build!r}, adapter=true, mode=simple")
        failed += 1
    return failed


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", default=DEPLOYMENT)
    parser.add_argument("--case", help="Run a single case id (e.g. anxiety_en)")
    parser.add_argument("--debug", action="store_true", help="Request debug_context with model_raw_text")
    parser.add_argument(
        "--delay",
        type=float,
        default=PROBE_DELAY_SEC,
        help="Seconds between probe cases (default 2)",
    )
    parser.add_argument(
        "--expected-build",
        default=EXPECTED_BUILD,
        help=f"Expected INFERENCE_BUILD (default {EXPECTED_BUILD})",
    )
    args = parser.parse_args()

    key = (os.environ.get("DAISY_ENDPOINT_KEY") or "").strip()
    if not key:
        print("DAISY_ENDPOINT_KEY required", file=sys.stderr)
        return 1

    cases = CASES
    if args.case:
        cases = [c for c in CASES if c["id"] == args.case]
        if not cases:
            print(f"Unknown case {args.case!r}", file=sys.stderr)
            return 1

    return 1 if _run_deployment(
        args.deployment,
        key,
        cases,
        use_debug=args.debug,
        delay_sec=args.delay,
        expected_build=args.expected_build,
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
