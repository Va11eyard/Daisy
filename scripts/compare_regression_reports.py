#!/usr/bin/env python3
"""
Daisy Cross-Topic Regression Report Comparison Tool
=====================================================
Compares two regression report JSON files (before and after) and produces a
detailed delta analysis including per-cluster, per-locale, failure-type changes,
case-level regressions/improvements, and a release gate evaluation.

Usage:
    python compare_regression_reports.py before.json after.json
    python compare_regression_reports.py before.json after.json --output delta.json
    python compare_regression_reports.py before.json after.json --format markdown --output delta.md

Exit codes:
    0  - comparison completed (check "can_release" in output for gate status)
    1  - file/parse error
    2  - invalid arguments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("compare_regression")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RELEASE_GATES = {
    "overall_90_percent": 0.90,
    "per_cluster_85_percent": 0.85,
}

FAILURE_CATEGORIES = [
    "canned_greeting",
    "structural_leak",
    "script_leak",
    "too_short",
    "hollow",
    "keyword_mismatch",
    "locale_incorrect",
]

ALL_CLUSTERS = [
    "breakup",
    "work",
    "anxiety",
    "stress",
    "grief",
    "clarity",
    "somatic",
]

DIRECTION_IMPROVED = "improved"
DIRECTION_REGRESSED = "regressed"
DIRECTION_UNCHANGED = "unchanged"

PASS_EMOJI = "\u2705"
FAIL_EMOJI = "\u274c"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_report(path: str) -> dict[str, Any]:
    """Load and validate a regression report JSON file."""
    p = Path(path)
    if not p.exists():
        logger.error("Report file not found: %s", p)
        sys.exit(1)
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", p, exc)
        sys.exit(1)

    # Validate required top-level keys
    required = {"overall", "by_cluster", "by_locale", "failure_breakdown", "cases"}
    missing = required - set(data.keys())
    if missing:
        logger.error("Report %s missing required keys: %s", p, ", ".join(sorted(missing)))
        sys.exit(1)
    return data


def _safe_rate(passed: int, total: int) -> float:
    """Compute pass rate safely."""
    return passed / total if total > 0 else 0.0


def _direction(before_rate: float, after_rate: float) -> str:
    """Determine direction of change."""
    delta = after_rate - before_rate
    if delta > 0.001:
        return DIRECTION_IMPROVED
    elif delta < -0.001:
        return DIRECTION_REGRESSED
    return DIRECTION_UNCHANGED


def _case_identity(case: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal dict identifying a case for reporting."""
    return {
        "id": case["id"],
        "cluster": case.get("cluster", "unknown"),
        "locale": case.get("locale", "en"),
        "failure_reasons": case.get("failure_reasons", []),
        "reply_preview": case.get("reply_preview", ""),
    }


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_reports(before_path: str, after_path: str) -> dict[str, Any]:
    """
    Compare two regression reports and return a detailed delta analysis.

    Args:
        before_path: Path to the "before" regression report JSON.
        after_path:  Path to the "after" regression report JSON.

    Returns:
        Dict with before/after snapshots, deltas, per-cluster changes,
        per-locale changes, failure changes, regressed/improved cases,
        and release gate evaluation.
    """
    before = _load_report(before_path)
    after = _load_report(after_path)

    # ------------------------------------------------------------------
    # Overall snapshot
    # ------------------------------------------------------------------
    b_ov = before["overall"]
    a_ov = after["overall"]

    before_summary = {
        "overall_pass_rate": b_ov["pass_rate"],
        "total_passed": b_ov["passed"],
        "total_failed": b_ov["failed"],
        "total_cases": b_ov["total"],
    }
    after_summary = {
        "overall_pass_rate": a_ov["pass_rate"],
        "total_passed": a_ov["passed"],
        "total_failed": a_ov["failed"],
        "total_cases": a_ov["total"],
    }

    delta_overall = {
        "overall_pass_rate": round(a_ov["pass_rate"] - b_ov["pass_rate"], 4),
        "total_passed": a_ov["passed"] - b_ov["passed"],
        "total_failed": a_ov["failed"] - b_ov["failed"],
    }

    # ------------------------------------------------------------------
    # By cluster
    # ------------------------------------------------------------------
    all_clusters = sorted(set(list(before["by_cluster"].keys()) + list(after["by_cluster"].keys())))
    by_cluster_delta: dict[str, dict[str, Any]] = {}

    for cluster in all_clusters:
        b_data = before["by_cluster"].get(cluster, {"total": 0, "passed": 0, "pass_rate": 0.0})
        a_data = after["by_cluster"].get(cluster, {"total": 0, "passed": 0, "pass_rate": 0.0})

        b_rate = b_data.get("pass_rate", _safe_rate(b_data.get("passed", 0), b_data.get("total", 0)))
        a_rate = a_data.get("pass_rate", _safe_rate(a_data.get("passed", 0), a_data.get("total", 0)))

        direction = _direction(b_rate, a_rate)
        by_cluster_delta[cluster] = {
            "before_rate": round(b_rate, 4),
            "after_rate": round(a_rate, 4),
            "delta": round(a_rate - b_rate, 4),
            "direction": direction,
            "before_passed": b_data.get("passed", 0),
            "after_passed": a_data.get("passed", 0),
            "before_total": b_data.get("total", 0),
            "after_total": a_data.get("total", 0),
        }

    # ------------------------------------------------------------------
    # By locale
    # ------------------------------------------------------------------
    all_locales = sorted(set(list(before["by_locale"].keys()) + list(after["by_locale"].keys())))
    by_locale_delta: dict[str, dict[str, Any]] = {}

    for locale in all_locales:
        b_data = before["by_locale"].get(locale, {"total": 0, "passed": 0, "pass_rate": 0.0})
        a_data = after["by_locale"].get(locale, {"total": 0, "passed": 0, "pass_rate": 0.0})

        b_rate = b_data.get("pass_rate", _safe_rate(b_data.get("passed", 0), b_data.get("total", 0)))
        a_rate = a_data.get("pass_rate", _safe_rate(a_data.get("passed", 0), a_data.get("total", 0)))

        by_locale_delta[locale] = {
            "before_rate": round(b_rate, 4),
            "after_rate": round(a_rate, 4),
            "delta": round(a_rate - b_rate, 4),
            "direction": _direction(b_rate, a_rate),
        }

    # ------------------------------------------------------------------
    # Failure changes
    # ------------------------------------------------------------------
    b_fail = before["failure_breakdown"]
    a_fail = after["failure_breakdown"]

    resolved: dict[str, int] = {}
    new_failures: dict[str, int] = {}
    remaining: dict[str, int] = {}

    for category in FAILURE_CATEGORIES:
        b_count = b_fail.get(category, 0)
        a_count = a_fail.get(category, 0)

        if b_count > a_count:
            resolved[category] = b_count - a_count
        elif a_count > b_count:
            new_failures[category] = a_count - b_count

        if a_count > 0:
            remaining[category] = a_count

    # ------------------------------------------------------------------
    # Case-level changes
    # ------------------------------------------------------------------
    b_cases = {c["id"]: c for c in before.get("cases", [])}
    a_cases = {c["id"]: c for c in after.get("cases", [])}

    all_case_ids = sorted(set(b_cases.keys()) | set(a_cases.keys()))

    regressed_cases: list[dict[str, Any]] = []   # passed before, failed after
    improved_cases: list[dict[str, Any]] = []    # failed before, passed after

    for case_id in all_case_ids:
        b_case = b_cases.get(case_id, {})
        a_case = a_cases.get(case_id, {})

        b_passed = b_case.get("passed", False) if b_case else False
        a_passed = a_case.get("passed", False) if a_case else False

        if b_passed and not a_passed:
            regressed_cases.append(_case_identity(a_case if a_case else b_case))
        elif not b_passed and a_passed:
            improved_cases.append(_case_identity(a_case if a_case else b_case))

    # ------------------------------------------------------------------
    # Release gate evaluation
    # ------------------------------------------------------------------
    overall_90 = a_ov["pass_rate"] >= RELEASE_GATES["overall_90_percent"]

    per_cluster_85 = True
    for cluster in ALL_CLUSTERS:
        a_data = after["by_cluster"].get(cluster, {})
        a_rate = a_data.get("pass_rate", 0.0)
        if a_rate < RELEASE_GATES["per_cluster_85_percent"]:
            per_cluster_85 = False
            break

    zero_structural = a_fail.get("structural_leak", 0) == 0
    zero_script = a_fail.get("script_leak", 0) == 0

    can_release = all([
        overall_90,
        per_cluster_85,
        zero_structural,
        zero_script,
    ])

    release_gate = {
        "overall_90_percent": overall_90,
        "per_cluster_85_percent": per_cluster_85,
        "zero_structural_leaks": zero_structural,
        "zero_script_leaks": zero_script,
        "can_release": can_release,
    }

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before": before_summary,
        "after": after_summary,
        "delta": delta_overall,
        "by_cluster": by_cluster_delta,
        "by_locale": by_locale_delta,
        "failure_changes": {
            "resolved": resolved,
            "new": new_failures,
            "remaining": remaining,
        },
        "regressed_cases": regressed_cases,
        "improved_cases": improved_cases,
        "release_gate": release_gate,
    }

    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_comparison(report: dict[str, Any]) -> None:
    """Print a formatted comparison table to stdout."""
    before = report["before"]
    after = report["after"]
    delta = report["delta"]

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _delta_signed(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.1f}%" if abs(v) < 1.0 else f"{sign}{v}"

    def _delta_int(v: int) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v}"

    # Header
    print()
    print("=" * 72)
    print("  DAISY REGRESSION REPORT COMPARISON")
    print(f"  Generated: {report['generated_at']}")
    print("=" * 72)

    # Overall summary
    print()
    print("-" * 72)
    print(f"  {'Metric':<30} {'Before':>12} {'After':>12} {'Delta':>12}")
    print("-" * 72)
    print(f"  {'Overall Pass Rate':<30} {_pct(before['overall_pass_rate']):>12} "
          f"{_pct(after['overall_pass_rate']):>12} {_delta_signed(delta['overall_pass_rate']):>12}")
    print(f"  {'Total Passed':<30} {before['total_passed']:>12} "
          f"{after['total_passed']:>12} {_delta_int(delta['total_passed']):>12}")
    print(f"  {'Total Failed':<30} {before['total_failed']:>12} "
          f"{after['total_failed']:>12} {_delta_int(delta['total_failed']):>12}")
    print(f"  {'Total Cases':<30} {before['total_cases']:>12} "
          f"{after['total_cases']:>12} "
          f"{_delta_int(after['total_cases'] - before['total_cases']):>12}")
    print("-" * 72)

    # Per-cluster breakdown
    print()
    print("  PER-CLUSTER BREAKDOWN")
    print("-" * 72)
    print(f"  {'Cluster':<15} {'Before':>10} {'After':>10} {'Delta':>10} {'Direction':>12}")
    print("-" * 72)
    for cluster, data in sorted(report["by_cluster"].items()):
        direction = data["direction"]
        dir_symbol = "\u2191" if direction == DIRECTION_IMPROVED else "\u2193" if direction == DIRECTION_REGRESSED else "\u2192"
        print(f"  {cluster:<15} {_pct(data['before_rate']):>10} "
              f"{_pct(data['after_rate']):>10} {_delta_signed(data['delta']):>10} "
              f"{direction:>10} {dir_symbol}")
    print("-" * 72)

    # Per-locale breakdown
    print()
    print("  PER-LOCALE BREAKDOWN")
    print("-" * 72)
    print(f"  {'Locale':<15} {'Before':>10} {'After':>10} {'Delta':>10} {'Direction':>12}")
    print("-" * 72)
    for locale, data in sorted(report["by_locale"].items()):
        direction = data["direction"]
        dir_symbol = "\u2191" if direction == DIRECTION_IMPROVED else "\u2193" if direction == DIRECTION_REGRESSED else "\u2192"
        print(f"  {locale:<15} {_pct(data['before_rate']):>10} "
              f"{_pct(data['after_rate']):>10} {_delta_signed(data['delta']):>10} "
              f"{direction:>10} {dir_symbol}")
    print("-" * 72)

    # Failure changes
    print()
    print("  FAILURE ANALYSIS")
    print("-" * 72)
    fc = report["failure_changes"]

    if fc["resolved"]:
        print(f"  Resolved (count reduced):")
        for cat, count in sorted(fc["resolved"].items(), key=lambda x: -x[1]):
            print(f"    {PASS_EMOJI} {cat}: -{count}")
    else:
        print(f"  Resolved: (none)")

    if fc["new"]:
        print(f"  New (count increased):")
        for cat, count in sorted(fc["new"].items(), key=lambda x: -x[1]):
            print(f"    {FAIL_EMOJI} {cat}: +{count}")
    else:
        print(f"  New: (none)")

    if fc["remaining"]:
        print(f"  Remaining (still present):")
        for cat, count in sorted(fc["remaining"].items(), key=lambda x: -x[1]):
            print(f"    {'  '}{cat}: {count}")
    else:
        print(f"  Remaining: (none)")
    print("-" * 72)

    # Case-level changes
    if report["regressed_cases"]:
        print()
        print(f"  REGRESSED CASES ({len(report['regressed_cases'])}): passed before, failed after")
        for case in report["regressed_cases"]:
            reasons = ", ".join(case["failure_reasons"]) if case["failure_reasons"] else "unknown"
            print(f"    {FAIL_EMOJI} {case['id']} ({case['cluster']}/{case['locale']}) — {reasons}")

    if report["improved_cases"]:
        print()
        print(f"  IMPROVED CASES ({len(report['improved_cases'])}): failed before, passed after")
        for case in report["improved_cases"]:
            print(f"    {PASS_EMOJI} {case['id']} ({case['cluster']}/{case['locale']})")

    # Release gates
    print()
    print("=" * 72)
    print("  RELEASE GATES")
    print("=" * 72)
    rg = report["release_gate"]
    gates = [
        ("Overall pass rate >= 90%", rg["overall_90_percent"]),
        ("Per-cluster pass rate >= 85%", rg["per_cluster_85_percent"]),
        ("Zero structural_leak failures", rg["zero_structural_leaks"]),
        ("Zero script_leak failures", rg["zero_script_leaks"]),
    ]
    for label, passed in gates:
        symbol = PASS_EMOJI if passed else FAIL_EMOJI
        print(f"    {symbol} {label}")

    print()
    can_release = rg["can_release"]
    release_symbol = PASS_EMOJI if can_release else FAIL_EMOJI
    release_text = "RELEASE APPROVED" if can_release else "DO NOT RELEASE"
    print(f"  {release_symbol} {release_text}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown(report: dict[str, Any]) -> str:
    """
    Generate a markdown report suitable for pasting into a GitHub PR.

    Args:
        report: The comparison report dict from compare_reports().

    Returns:
        A markdown-formatted string.
    """
    before = report["before"]
    after = report["after"]
    delta = report["delta"]
    rg = report["release_gate"]

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _delta_signed(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.1f}%" if abs(v) < 1.0 else f"{sign}{v}"

    def _delta_int(v: int) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v}"

    def _gate_emoji(passed: bool) -> str:
        return PASS_EMOJI if passed else FAIL_EMOJI

    lines: list[str] = []

    # Title
    lines.append("# Daisy Regression Report Comparison")
    lines.append("")
    lines.append(f"**Generated:** {report['generated_at']}Z")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("|--------|--------|-------|-------|")
    lines.append(
        f"| Overall Pass Rate | {_pct(before['overall_pass_rate'])} | "
        f"{_pct(after['overall_pass_rate'])} | {_delta_signed(delta['overall_pass_rate'])} |"
    )
    lines.append(
        f"| Total Passed | {before['total_passed']} | {after['total_passed']} | "
        f"{_delta_int(delta['total_passed'])} |"
    )
    lines.append(
        f"| Total Failed | {before['total_failed']} | {after['total_failed']} | "
        f"{_delta_int(delta['total_failed'])} |"
    )
    lines.append(
        f"| Total Cases | {before['total_cases']} | {after['total_cases']} | "
        f"{_delta_int(after['total_cases'] - before['total_cases'])} |"
    )
    lines.append("")

    # Per-cluster breakdown
    lines.append("## Per-Cluster Breakdown")
    lines.append("")
    lines.append("| Cluster | Before | After | Delta | Direction |")
    lines.append("|---------|--------|-------|-------|-----------|")
    for cluster, data in sorted(report["by_cluster"].items()):
        direction = data["direction"]
        dir_emoji = "\u2191" if direction == DIRECTION_IMPROVED else "\u2193" if direction == DIRECTION_REGRESSED else "\u2192"
        lines.append(
            f"| {cluster} | {_pct(data['before_rate'])} | {_pct(data['after_rate'])} | "
            f"{_delta_signed(data['delta'])} | {direction} {dir_emoji} |"
        )
    lines.append("")

    # Per-locale breakdown
    lines.append("## Per-Locale Breakdown")
    lines.append("")
    lines.append("| Locale | Before | After | Delta | Direction |")
    lines.append("|--------|--------|-------|-------|-----------|")
    for locale, data in sorted(report["by_locale"].items()):
        direction = data["direction"]
        dir_emoji = "\u2191" if direction == DIRECTION_IMPROVED else "\u2193" if direction == DIRECTION_REGRESSED else "\u2192"
        lines.append(
            f"| {locale} | {_pct(data['before_rate'])} | {_pct(data['after_rate'])} | "
            f"{_delta_signed(data['delta'])} | {direction} {dir_emoji} |"
        )
    lines.append("")

    # Failure analysis
    lines.append("## Failure Analysis")
    lines.append("")

    fc = report["failure_changes"]

    if fc["resolved"]:
        lines.append("### Resolved (count reduced)")
        lines.append("")
        for cat, count in sorted(fc["resolved"].items(), key=lambda x: -x[1]):
            lines.append(f"- {PASS_EMOJI} **{cat}**: -{count}")
        lines.append("")

    if fc["new"]:
        lines.append("### New (count increased)")
        lines.append("")
        for cat, count in sorted(fc["new"].items(), key=lambda x: -x[1]):
            lines.append(f"- {FAIL_EMOJI} **{cat}**: +{count}")
        lines.append("")

    lines.append("### Remaining (still present)")
    lines.append("")
    if fc["remaining"]:
        for cat, count in sorted(fc["remaining"].items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")
    else:
        lines.append("*No remaining failures*")
    lines.append("")

    # Regressed / improved cases
    if report["regressed_cases"]:
        lines.append("## Regressed Cases (passed before, failed after)")
        lines.append("")
        lines.append("| Case ID | Cluster | Locale | Failure Reasons |")
        lines.append("|---------|---------|--------|-----------------|")
        for case in report["regressed_cases"]:
            reasons = ", ".join(case["failure_reasons"]) if case["failure_reasons"] else "unknown"
            lines.append(f"| {case['id']} | {case['cluster']} | {case['locale']} | {reasons} |")
        lines.append("")

    if report["improved_cases"]:
        lines.append("## Improved Cases (failed before, passed after)")
        lines.append("")
        lines.append("| Case ID | Cluster | Locale |")
        lines.append("|---------|---------|--------|")
        for case in report["improved_cases"]:
            lines.append(f"| {case['id']} | {case['cluster']} | {case['locale']} |")
        lines.append("")

    # Release gate status
    lines.append("## Release Gate Status")
    lines.append("")
    lines.append("| Gate | Status |")
    lines.append("|------|--------|")
    lines.append(f"| Overall pass rate >= 90% | {_gate_emoji(rg['overall_90_percent'])} |")
    lines.append(f"| Per-cluster pass rate >= 85% | {_gate_emoji(rg['per_cluster_85_percent'])} |")
    lines.append(f"| Zero structural_leak failures | {_gate_emoji(rg['zero_structural_leaks'])} |")
    lines.append(f"| Zero script_leak failures | {_gate_emoji(rg['zero_script_leaks'])} |")
    lines.append("")

    can_release = rg["can_release"]
    if can_release:
        lines.append(f"## {_gate_emoji(True)} RELEASE APPROVED")
        lines.append("")
        lines.append("All release gates passed. The deployment is cleared for production cutover.")
    else:
        lines.append(f"## {_gate_emoji(False)} DO NOT RELEASE")
        lines.append("")
        lines.append("One or more release gates failed. Address the issues above before cutting over.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Daisy regression reports (before vs after) and produce a delta analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_regression_reports.py before.json after.json
  python compare_regression_reports.py before.json after.json --output delta.json
  python compare_regression_reports.py before.json after.json --format markdown --output report.md
        """,
    )
    parser.add_argument("before", help="Path to the 'before' regression report JSON")
    parser.add_argument("after", help="Path to the 'after' regression report JSON")
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Path to write the comparison output (default: stdout only)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown"],
        default="json",
        help="Output format for the written file (default: json). Console always shows pretty table.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (only write to file)",
    )
    args = parser.parse_args()

    # Run comparison
    report = compare_reports(args.before, args.after)

    # Always print pretty table to console (unless --quiet)
    if not args.quiet:
        print_comparison(report)

    # Write output file if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "markdown":
            content = generate_markdown(report)
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write(content)
            if not args.quiet:
                logger.info("Markdown report written to %s", out_path)
        else:
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            if not args.quiet:
                logger.info("JSON report written to %s", out_path)

    # Exit code reflects release gate status
    sys.exit(0 if report["release_gate"]["can_release"] else 1)


if __name__ == "__main__":
    main()
