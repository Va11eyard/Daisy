#!/usr/bin/env python3
"""
Daisy Cross-Topic Regression Test Runner
========================================
Runs 56 regression cases (8 topic clusters x EN/RU) against the live Daisy
endpoint and produces a JSON report with per-cluster, per-locale, and per-case
breakdowns.

Usage:
    python run_cross_topic_regression.py [--deployment NAME] [--limit N] \
        [--delay SECONDS] [--output PATH]

Example:
    DAISY_ENDPOINT_KEY=<token> python run_cross_topic_regression.py \
        --deployment gpu-deployment-finetuned --limit 10 --delay 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("regression")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = (
    "https://daisy-therapy.westus2.inference.ml.azure.com/score"
)
DEFAULT_DEPLOYMENT = "gpu-deployment-finetuned"
DEFAULT_DELAY = 2.0
MAX_CONCURRENT = 1
MAX_RETRIES = 3
ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "eval" / "cross_topic_regression.jsonl"

# Known canned greeting variants
_CANNED_GREETINGS = [
    "Hey — I'm glad you're here",
    "Hey -- I'm glad you're here",
    "Hey -- I'm glad",
    "Hey – I'm glad you're here",
    "Hey - I'm glad you're here",
]

# Structural leak patterns
_STRUCTURAL_PATTERNS = [
    re.compile(r"Assistant\s*:", re.IGNORECASE),
    re.compile(r"Question\s*:", re.IGNORECASE),
    re.compile(r"CRITICAL OUTPUT", re.IGNORECASE),
    re.compile(r"NEVER USE", re.IGNORECASE),
    re.compile(r"RUBRIC", re.IGNORECASE),
    re.compile(r"\.{3,}"),  # three or more dots in sequence
]

# Polish diacritics
_POLISH_DIACRITICS = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")

# Latin word in Cyrillic text
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")

# Parenthetical English phrases in Cyrillic
_EN_PARENTHETICAL_RE = re.compile(r"\s*\([a-zA-Z\s\-]+\)\s*")

# Sentence delimiters (EN + Cyrillic)
_SENTENCE_DELIMITERS = re.compile(r"[.!?।。]+\s+")

# For RU locale: detect formal vs informal
_RU_FORMAL = re.compile(r"\bвы\b|\bвас\b|\bвам\b|\bваш\b", re.IGNORECASE)
_RU_INFORMAL = re.compile(r"\bты\b|\bтебя\b|\bтебе\b|\bтвой\b", re.IGNORECASE)

# Latin-heavy sentence detection (for locale_correct)
_LATIN_HEAVY_SENTENCE = re.compile(r"[a-zA-Z]")
_COMPLETE_EN_SENTENCE = re.compile(r"^[A-Z][a-zA-Z\s,;:'\"-]+[.!?]$")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _has_canned_greeting(text: str) -> bool:
    """Check if text starts with a known canned greeting."""
    text_lower = text.strip().lower()
    for greeting in _CANNED_GREETINGS:
        greeting_lower = greeting.lower()
        # Exact prefix match or >80% Jaccard on first words
        if text_lower.startswith(greeting_lower):
            return True
        # Jaccard on first N words
        text_words = text_lower.split()[:7]
        greeting_words = greeting_lower.split()
        if not text_words or not greeting_words:
            continue
        set_text = set(text_words)
        set_greeting = set(greeting_words)
        intersection = len(set_text & set_greeting)
        union = len(set_text | set_greeting)
        if union > 0 and intersection / union >= 0.80:
            return True
    return False


def _has_structural_leak(text: str) -> bool:
    """Check for structural leak patterns (rubric tokens, role prefixes, etc.)."""
    for pattern in _STRUCTURAL_PATTERNS:
        if pattern.search(text):
            return True
    # Check for punctuation-loop artifacts like ". , . ," or repeated ". ,"
    # Count occurrences of period-comma pattern (period, optional spaces, comma)
    period_comma_count = len(re.findall(r"\.\s*,", text))
    if period_comma_count >= 2:
        return True
    return False


def _count_latin_words(text: str) -> int:
    """Count ASCII alpha words in text."""
    return len(_LATIN_WORD_RE.findall(text))


def _has_script_leak(text: str, locale: str) -> bool:
    """
    Check for script leak:
    - For RU/KK: >=3 Latin words in a row, OR Polish diacritics present.
    """
    if locale not in ("ru", "kk"):
        return False

    # Polish diacritics anywhere
    if _POLISH_DIACRITICS.search(text):
        return True

    # Check for 3+ consecutive Latin words
    tokens = re.split(r"(\s+|[^\w])", text)
    consecutive = 0
    for token in tokens:
        if _LATIN_WORD_RE.fullmatch(token):
            consecutive += 1
            if consecutive >= 3:
                return True
        elif token and not token.isspace() and not token.isdigit():
            # Non-Latin, non-space token resets the count
            consecutive = 0
    return False


def _keyword_match(text: str, keywords: list[str]) -> bool:
    """Check if at least one keyword appears in text (case-insensitive prefix match)."""
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return True
    return False


def _is_hollow(text: str) -> bool:
    """Check if response is a hollow one-liner (< 2 sentences AND < 60 chars)."""
    sentences = [s for s in _SENTENCE_DELIMITERS.split(text) if s.strip()]
    has_two_plus = len(sentences) >= 2
    char_count = len(text.strip())
    return not (has_two_plus or char_count >= 60)


def _is_locale_correct(text: str, locale: str) -> bool:
    """
    For RU locale:
    - Check informal 'ты' usage (prefer informal over formal 'вы')
    - No complete English sentences embedded in Cyrillic text
    Returns True if locale checks pass.
    """
    if locale != "ru":
        return True

    # Check for complete English sentences embedded in Cyrillic text
    # Split into sentences and check for Latin-heavy ones
    text_stripped = text.strip()
    # Look for sentences that are mostly Latin (embedded English)
    sentences = [s.strip() for s in _SENTENCE_DELIMITERS.split(text_stripped) if s.strip()]
    for sentence in sentences:
        # If sentence has mostly Latin characters, it's an embedded EN sentence
        cyrillic_chars = len(re.findall(r"[\u0400-\u04FF]", sentence))
        latin_chars = len(re.findall(r"[a-zA-Z]", sentence))
        total_alpha = cyrillic_chars + latin_chars
        if total_alpha > 0 and latin_chars / total_alpha > 0.5:
            # This is a Latin-dominant sentence in RU response
            if _COMPLETE_EN_SENTENCE.match(sentence.strip()):
                return False
            # Also flag if clearly a standalone English sentence
            if cyrillic_chars == 0 and latin_chars > 10:
                return False

    return True


def score_response(text: str, case: dict[str, Any]) -> dict[str, Any]:
    """
    Score a response text against all criteria.

    Returns dict with:
        passed: bool
        length_ok: bool
        no_canned_greeting: bool
        no_structural_leak: bool
        no_script_leak: bool
        keyword_match: bool
        not_hollow: bool
        locale_correct: bool
        latency_ms: int (should be set by caller)
        failure_reasons: list[str]
    """
    locale = case.get("locale", "en")
    keywords = case.get("keywords", [])

    length_ok = len(text) >= 25
    no_canned_greeting = not _has_canned_greeting(text)
    no_structural_leak = not _has_structural_leak(text)
    no_script_leak = not _has_script_leak(text, locale)
    kw_match = _keyword_match(text, keywords)
    not_hollow = not _is_hollow(text)
    locale_correct = _is_locale_correct(text, locale)

    failure_reasons: list[str] = []
    if not length_ok:
        failure_reasons.append("too_short")
    if not no_canned_greeting:
        failure_reasons.append("canned_greeting")
    if not no_structural_leak:
        failure_reasons.append("structural_leak")
    if not no_script_leak:
        failure_reasons.append("script_leak")
    if not kw_match:
        failure_reasons.append("keyword_mismatch")
    if not not_hollow:
        failure_reasons.append("hollow")
    if not locale_correct:
        failure_reasons.append("locale_incorrect")

    passed = all([
        length_ok,
        no_canned_greeting,
        no_structural_leak,
        no_script_leak,
        kw_match,
        not_hollow,
        locale_correct,
    ])

    return {
        "passed": passed,
        "length_ok": length_ok,
        "no_canned_greeting": no_canned_greeting,
        "no_structural_leak": no_structural_leak,
        "no_script_leak": no_script_leak,
        "keyword_match": kw_match,
        "not_hollow": not_hollow,
        "locale_correct": locale_correct,
        "latency_ms": 0,  # caller fills this
        "failure_reasons": failure_reasons,
    }


# ---------------------------------------------------------------------------
# Async HTTP layer
# ---------------------------------------------------------------------------

def _get_auth_token() -> str:
    """Retrieve bearer token from env or Azure CLI."""
    token = os.environ.get("DAISY_ENDPOINT_KEY", "").strip()
    if token:
        return token
    # Attempt Azure CLI
    try:
        import subprocess

        result = subprocess.run(
            ["az", "account", "get-access-token", "--resource", "https://management.azure.com/"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("accessToken", "")
    except Exception:
        pass
    return ""


async def _http_post_with_retry(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict,
    deployment: str | None,
    token: str,
    retries: int = MAX_RETRIES,
) -> tuple[dict[str, Any] | None, int, str]:
    """
    POST with retry logic. Returns (json_body, latency_ms, error_reason).
    error_reason is empty string on success.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if deployment:
        headers["azureml-model-deployment"] = deployment

    last_error = ""
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if resp.status == 200:
                    try:
                        body = await resp.json()
                        return body, latency_ms, ""
                    except Exception as exc:
                        last_error = f"json_parse_error: {exc}"
                        return None, latency_ms, last_error
                elif resp.status == 429:
                    text = await resp.text()
                    last_error = f"http_{resp.status}: {text[:200]}"
                    wait = min(30, 2 ** attempt * 2)
                    logger.warning("Attempt %d/%d: HTTP 429, retrying in %.1fs", attempt, retries, wait)
                    await asyncio.sleep(wait)
                elif 500 <= resp.status < 600:
                    text = await resp.text()
                    last_error = f"server_error_{resp.status}: {text[:200]}"
                    wait = 2 ** attempt
                    logger.warning("Attempt %d/%d: HTTP %d, retrying in %.1fs", attempt, retries, resp.status, wait)
                    await asyncio.sleep(wait)
                else:
                    text = await resp.text()
                    last_error = f"http_{resp.status}: {text[:200]}"
                    # Non-retryable client error
                    return None, latency_ms, last_error
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            last_error = "timeout"
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d: timeout, retrying in %.1fs", attempt, retries, wait)
            await asyncio.sleep(wait)
        except aiohttp.ClientError as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            last_error = f"network_error: {exc}"
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d: %s, retrying in %.1fs", attempt, retries, last_error, wait)
            await asyncio.sleep(wait)

    return None, 0, last_error


async def run_case(
    endpoint: str,
    case: dict[str, Any],
    deployment: str | None,
    session: aiohttp.ClientSession,
    token: str,
    delay: float,
) -> dict[str, Any]:
    """
    Run a single regression case against the endpoint.
    Returns the full result dict including the case id, scoring, and metadata.
    """
    case_id = case["id"]
    locale = case.get("locale", "en")
    message = case["message"]

    payload = {
        "messages": [{"role": "user", "content": message}],
        "locale": locale,
        "history": case.get("history", []),
    }
    if case.get("user_gender"):
        payload["user_gender"] = case["user_gender"]

    # Optional delay between requests
    if delay > 0:
        await asyncio.sleep(delay)

    body, latency_ms, error = await _http_post_with_retry(
        session, endpoint, payload, deployment, token
    )

    if error:
        return {
            "id": case_id,
            "cluster": case.get("cluster", "unknown"),
            "locale": locale,
            "passed": False,
            "failure_reasons": [error],
            "latency_ms": latency_ms,
            "reply_preview": "",
            "scores": {
                "passed": False,
                "length_ok": False,
                "no_canned_greeting": False,
                "no_structural_leak": False,
                "no_script_leak": False,
                "keyword_match": False,
                "not_hollow": False,
                "locale_correct": False,
                "latency_ms": latency_ms,
                "failure_reasons": [error],
            },
        }

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {"response": body}
    reply = (body.get("response") or body.get("reply") or "") if isinstance(body, dict) else ""
    scores = score_response(reply, case)
    scores["latency_ms"] = latency_ms

    return {
        "id": case_id,
        "cluster": case.get("cluster", "unknown"),
        "locale": locale,
        "passed": scores["passed"],
        "failure_reasons": scores["failure_reasons"],
        "latency_ms": latency_ms,
        "reply_preview": reply[:200] if reply else "",
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-case results into cluster-level and locale-level summaries.
    Returns the full report dict.
    """
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0.0

    # By cluster
    clusters: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "en_pass": 0, "ru_pass": 0}
    )
    for r in results:
        c = r["cluster"]
        clusters[c]["total"] += 1
        if r["passed"]:
            clusters[c]["passed"] += 1
            if r["locale"] == "en":
                clusters[c]["en_pass"] += 1
            elif r["locale"] == "ru":
                clusters[c]["ru_pass"] += 1
        else:
            clusters[c]["failed"] += 1

    by_cluster: dict[str, dict[str, Any]] = {}
    for c, data in sorted(clusters.items()):
        by_cluster[c] = {
            "total": data["total"],
            "passed": data["passed"],
            "failed": data["failed"],
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0.0,
            "en_pass": data["en_pass"],
            "ru_pass": data["ru_pass"],
        }

    # By locale
    locales: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        loc = r["locale"]
        locales[loc]["total"] += 1
        if r["passed"]:
            locales[loc]["passed"] += 1

    by_locale: dict[str, dict[str, Any]] = {}
    for loc, data in sorted(locales.items()):
        by_locale[loc] = {
            "total": data["total"],
            "passed": data["passed"],
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0.0,
        }

    # Failure breakdown
    failure_counter: Counter = Counter()
    for r in results:
        if not r["passed"]:
            for reason in r["failure_reasons"]:
                # Map detailed reasons to coarse categories
                if "canned_greeting" in reason or reason == "canned_greeting":
                    failure_counter["canned_greeting"] += 1
                elif "structural_leak" in reason or reason == "structural_leak":
                    failure_counter["structural_leak"] += 1
                elif "script_leak" in reason or reason == "script_leak":
                    failure_counter["script_leak"] += 1
                elif "too_short" in reason or reason == "too_short":
                    failure_counter["too_short"] += 1
                elif "hollow" in reason or reason == "hollow":
                    failure_counter["hollow"] += 1
                elif "keyword_mismatch" in reason:
                    failure_counter["keyword_mismatch"] += 1
                elif "locale_incorrect" in reason:
                    failure_counter["locale_incorrect"] += 1
                else:
                    failure_counter[reason] += 1

    failure_breakdown = {
        "canned_greeting": failure_counter.get("canned_greeting", 0),
        "structural_leak": failure_counter.get("structural_leak", 0),
        "script_leak": failure_counter.get("script_leak", 0),
        "too_short": failure_counter.get("too_short", 0),
        "hollow": failure_counter.get("hollow", 0),
        "keyword_mismatch": failure_counter.get("keyword_mismatch", 0),
        "locale_incorrect": failure_counter.get("locale_incorrect", 0),
    }

    return {
        "overall": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
        },
        "by_cluster": by_cluster,
        "by_locale": by_locale,
        "failure_breakdown": failure_breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daisy cross-topic regression test runner (56 cases)"
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("DAISY_ENDPOINT", DEFAULT_ENDPOINT),
        help="Daisy scoring endpoint URL",
    )
    parser.add_argument(
        "--deployment",
        default=os.environ.get("DAISY_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        help="Azure ML deployment name header",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of cases to run (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay between requests (seconds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to write JSON report (default: stdout)",
    )
    parser.add_argument(
        "--cases-file",
        type=str,
        default=str(CASES_FILE),
        help="Path to JSONL cases file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENT,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL verification",
    )
    args = parser.parse_args()

    endpoint = args.endpoint
    deployment = args.deployment or None
    limit = args.limit
    delay = args.delay
    output_path = args.output
    cases_file = Path(args.cases_file)
    concurrency = args.concurrency

    # Load cases
    if not cases_file.exists():
        logger.error("Cases file not found: %s", cases_file)
        sys.exit(1)

    cases: list[dict[str, Any]] = []
    with cases_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON line: %s — %s", exc, line[:80])

    if limit > 0:
        cases = cases[:limit]

    logger.info("Loaded %d cases from %s", len(cases), cases_file)
    logger.info("Endpoint: %s | Deployment: %s", endpoint, deployment or "(default)")

    # Auth token
    token = _get_auth_token()
    if not token:
        logger.warning("No auth token found — set DAISY_ENDPOINT_KEY env var")

    # Run cases with semaphore
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=not args.no_verify_ssl),
            ) as session:
                return await run_case(endpoint, case, deployment, session, token, delay)

    # Create shared session for all requests
    connector = aiohttp.TCPConnector(
        ssl=not args.no_verify_ssl,
        limit=concurrency,
        limit_per_host=concurrency,
    )

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        # We'll wrap each call with the semaphore
        sem = asyncio.Semaphore(concurrency)

        async def _bound_run(case: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await run_case(endpoint, case, deployment, session, token, delay)

        tasks = [_bound_run(c) for c in cases]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(
                "[%s] %s (%s/%s) — %s (%dms)",
                status,
                result["id"],
                result["cluster"],
                result["locale"],
                ", ".join(result["failure_reasons"]) if result["failure_reasons"] else "ok",
                result["latency_ms"],
            )

    elapsed = time.perf_counter() - t_start
    logger.info("Completed %d cases in %.1fs (%.1f cases/s)", len(results), elapsed, len(results) / elapsed)

    # Aggregate
    aggregation = aggregate_results(results)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "deployment": deployment or "(default)",
        **aggregation,
        "cases": results,
    }

    # Print summary to stderr
    ov = report["overall"]
    logger.info("=" * 50)
    logger.info("OVERALL: %d passed / %d total (%.1f%%)", ov["passed"], ov["total"], ov["pass_rate"] * 100)
    logger.info("By cluster:")
    for c, d in report["by_cluster"].items():
        logger.info("  %-10s: %d/%d (%.1f%%)", c, d["passed"], d["total"], d["pass_rate"] * 100)
    logger.info("By locale:")
    for loc, d in report["by_locale"].items():
        logger.info("  %-3s: %d/%d (%.1f%%)", loc, d["passed"], d["total"], d["pass_rate"] * 100)
    logger.info("Failure breakdown: %s", report["failure_breakdown"])
    logger.info("=" * 50)

    # Output
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(report_json)
        logger.info("Report written to %s", out_path)
    else:
        print(report_json)


if __name__ == "__main__":
    asyncio.run(main())
