"""Render ablation results as a ranked report with dominant-cause analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dominant_cause(adjacent: list[dict], summaries: list[dict]) -> dict:
    """Identify largest contributor to genericness from adjacent deltas."""
    if not adjacent:
        return {"verdict": "insufficient_data"}

    # Primary: C4 vs C3 (anti-halluc layers)
    anti = next((d for d in adjacent if d["from"] == "C3_rag" and d["to"] == "C4_full"), None)
    fix_cmp = None
    s4 = next((s for s in summaries if s["config"] == "C4_full"), None)
    s4nf = next((s for s in summaries if s["config"] == "C4_nofallback"), None)
    if s4 and s4nf:
        fix_cmp = {
            "delta_genericness": round(
                s4nf["cross_scenario_genericness"] - s4["cross_scenario_genericness"], 4
            ),
            "delta_canned_rate": round(s4nf["canned_rate"] - s4["canned_rate"], 4),
            "delta_specificity": round(s4nf["mean_specificity"] - s4["mean_specificity"], 4),
        }
    ranked = sorted(
        adjacent,
        key=lambda d: d["delta_genericness"],
        reverse=True,
    )
    worst_step = ranked[0] if ranked else None

    c4 = next((s for s in summaries if s["config"] == "C4_full"), None)
    c0 = next((s for s in summaries if s["config"] == "C0_base"), None)

    verdict = "unknown"
    evidence: list[str] = []
    if anti and anti["delta_genericness"] > 0.02:
        verdict = "anti_hallucination_layers"
        evidence.append(
            f"C4 vs C3: genericness +{anti['delta_genericness']}, "
            f"canned_rate +{anti.get('delta_canned_rate', 0)}, "
            f"specificity {anti.get('delta_specificity', 0):+.4f}"
        )
    elif anti and anti.get("delta_canned_rate", 0) > 0.1:
        verdict = "anti_hallucination_layers"
        evidence.append(
            f"C4 vs C3: canned_rate +{anti['delta_canned_rate']}, "
            f"specificity {anti.get('delta_specificity', 0):+.4f}"
        )
    elif worst_step:
        verdict = f"largest_step_{worst_step['from']}_to_{worst_step['to']}"
        evidence.append(
            f"Largest genericness delta: {worst_step['from']}->{worst_step['to']} "
            f"(+{worst_step['delta_genericness']})"
        )

    if c4 and c4.get("fallback_rate", 0) >= 0.5:
        evidence.append(f"C4 fallback_rate={c4['fallback_rate']} (curated fallback on every turn)")
        verdict = "anti_hallucination_layers"

    if fix_cmp:
        fb_before = s4.get("fallback_rate", 0)
        fb_after = s4nf.get("fallback_rate", 0)
        evidence.append(
            f"Fix validation C4_full vs C4_nofallback: "
            f"genericness {fix_cmp['delta_genericness']:+.4f}, "
            f"specificity {fix_cmp['delta_specificity']:+.4f}, "
            f"fallback_rate {fb_before} -> {fb_after}"
        )

    return {
        "verdict": verdict,
        "evidence": evidence,
        "anti_halluc_delta": anti,
        "fix_nofallback_delta": fix_cmp,
        "ranked_steps": ranked,
    }


def fix_verification(summaries: list[dict]) -> dict | None:
    """Compare C4_full (pre-fix) vs C4_nofallback (post-fix proxy)."""
    before = next((s for s in summaries if s["config"] == "C4_full"), None)
    after = next((s for s in summaries if s["config"] == "C4_nofallback"), None)
    if not before or not after:
        return None

    def row(metric: str, key: str, fmt: str = ".4f", higher_better: bool | None = None) -> dict:
        b, a = before[key], after[key]
        delta = round(a - b, 4)
        direction = ""
        if higher_better is not None and delta != 0:
            improved = (delta > 0) if higher_better else (delta < 0)
            direction = " (improved)" if improved else " (worse)"
        return {
            "metric": metric,
            "before": format(b, fmt),
            "after": format(a, fmt),
            "delta": f"{delta:+.4f}{direction}",
        }

    rows = [
        row("cross_scenario_genericness", "cross_scenario_genericness", higher_better=False),
        row("distinct_2", "distinct_2", higher_better=True),
        row("canned_rate", "canned_rate", higher_better=False),
        row("mean_specificity", "mean_specificity", higher_better=True),
        row("fallback_rate", "fallback_rate", ".4f", higher_better=False),
    ]
    d_gen = round(after["cross_scenario_genericness"] - before["cross_scenario_genericness"], 4)
    d_spec = round(after["mean_specificity"] - before["mean_specificity"], 4)
    d_d2 = round(after["distinct_2"] - before["distinct_2"], 4)
    fb_b, fb_a = before.get("fallback_rate", 0), after.get("fallback_rate", 0)
    return {
        "fix": "score.py: keep model output after voice-QC regen (no fallback_reply substitution)",
        "proxy_config": "C4_nofallback",
        "rows": rows,
        "conclusion": (
            f"Fallback substitution was the dominant blandness driver: fallback_rate "
            f"{fb_b} -> {fb_a}, cross_scenario_genericness {d_gen:+.4f}, "
            f"specificity {d_spec:+.4f}, distinct_2 {d_d2:+.4f}. "
            "Fix applied in score.py; redeploy endpoint to validate on production traffic."
        ),
    }


def format_markdown(data: dict) -> str:
    lines = [
        "# Genericness ablation report",
        "",
        f"Timestamp: {data.get('timestamp', 'n/a')}",
        f"Cases: {data.get('n_cases', 'n/a')}",
        "",
        "## Metrics by configuration",
        "",
        "| Config | Genericness | Distinct-2 | Canned rate | Specificity | Fallback rate |",
        "|--------|-------------|------------|-------------|-------------|---------------|",
    ]
    for s in data.get("summaries", []):
        lines.append(
            f"| {s['config']} | {s['cross_scenario_genericness']} | {s['distinct_2']} | "
            f"{s['canned_rate']} | {s['mean_specificity']} | {s.get('fallback_rate', 'n/a')} |"
        )

    lines.extend(["", "## Adjacent config deltas (effect size)", ""])
    for d in data.get("adjacent_deltas", []):
        lines.append(
            f"- **{d['from']} -> {d['to']}**: "
            f"delta_genericness={d['delta_genericness']:+.4f}, "
            f"delta_canned={d['delta_canned_rate']:+.4f}, "
            f"delta_specificity={d['delta_specificity']:+.4f}"
        )

    cause = data.get("dominant_cause", {})
    lines.extend([
        "",
        "## Dominant cause",
        "",
        f"**Verdict:** `{cause.get('verdict', 'n/a')}`",
        "",
    ])
    for e in cause.get("evidence", []):
        lines.append(f"- {e}")

    if data.get("recommended_fix"):
        lines.extend(["", "## Recommended fix", "", data["recommended_fix"]])

    fix_v = data.get("fix_verification")
    if fix_v:
        lines.extend([
            "",
            "## Fix verification (C4_full vs C4_nofallback)",
            "",
            "Production fix: disable `fallback_reply` substitution after voice-QC regen in `score.py`.",
            "Ablation proxy: `C4_nofallback` = same stack without curated fallback substitution.",
            "",
            "| Metric | C4_full (before) | C4_nofallback (after) | Delta |",
            "|--------|------------------|------------------------|-------|",
        ])
        for row in fix_v.get("rows", []):
            lines.append(
                f"| {row['metric']} | {row['before']} | {row['after']} | {row['delta']} |"
            )
        lines.extend(["", fix_v.get("conclusion", "")])

    return "\n".join(lines)


FIX_BY_VERDICT = {
    "anti_hallucination_layers": (
        "Disable curated fallback substitution in score.py (keep regen, drop fallback_reply "
        "replacement). Re-measure C4 canned_rate and cross_scenario_genericness."
    ),
    "largest_step_C2_prompt_to_C3_rag": (
        "Replace book_index.json meta-instructions with concrete technique excerpts from books; "
        "or disable RAG injection until corpus is non-generic."
    ),
    "largest_step_C1_lora_to_C2_prompt": (
        "Switch production to DAISY_PROMPT_MODE=aligned (compact overlay) instead of full voice "
        "contract stack; re-measure."
    ),
    "largest_step_C0_base_to_C1_lora": (
        "LoRA weights bias toward generic validation shape — expand v12 training with diverse "
        "response structures, not only reflect+question."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "results" / "ablation_results.json"),
    )
    parser.add_argument(
        "--offline",
        default=str(Path(__file__).resolve().parent / "results" / "offline_audit.json"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "ablation_report.md"),
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cause = dominant_cause(data.get("adjacent_deltas", []), data.get("summaries", []))

    offline_path = Path(args.offline)
    if offline_path.is_file():
        offline = json.loads(offline_path.read_text(encoding="utf-8"))
        rag = offline.get("rag_audit", {})
        if rag.get("meta_instruction_fraction", 0) >= 0.5:
            cause.setdefault("offline_rag_note", (
                f"Offline: {rag['meta_instruction_fraction']*100:.0f}% of RAG chunks are "
                "meta-instructions similar to bland outputs."
            ))
        lora = offline.get("lora_data_audit", {})
        if lora.get("reflect_plus_question_shape_rate", 0) >= 0.7:
            cause.setdefault("offline_lora_note", (
                f"Offline: {lora['reflect_plus_question_shape_rate']*100:.0f}% of training "
                "targets match reflect+question template."
            ))

    verdict = cause["verdict"]
    fix = FIX_BY_VERDICT.get(verdict)
    if not fix and verdict.startswith("largest_step_"):
        for key, val in FIX_BY_VERDICT.items():
            if verdict in key or key in verdict:
                fix = val
                break
    fix_v = fix_verification(data.get("summaries", []))
    data["dominant_cause"] = cause
    data["recommended_fix"] = fix or "Re-run ablation; no dominant cause above threshold."
    data["fix_verification"] = fix_v
    if fix_v:
        fix_path = Path(args.output).parent / "fix_verification.json"
        fix_path.write_text(json.dumps(fix_v, indent=2), encoding="utf-8")

    md = format_markdown(data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
