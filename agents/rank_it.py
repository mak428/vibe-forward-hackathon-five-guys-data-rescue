"""Agent 2 — Rank It: score each issue's audit severity using a PyMC Bayesian model."""

import json
from pathlib import Path

import cognee
import numpy as np
import pymc as pm

OUTPUT_DIR = Path(__file__).parent.parent / "output"
TOTAL_RECORDS = 5000

# Beta prior (alpha, beta) encodes our domain belief about how audit-critical each issue type is.
# Higher alpha / lower beta → higher expected severity.
#
# The six canonical data-quality issue classes:
#   1. exact_duplicate      — byte-for-byte copy, different record_id
#   2. near_duplicate       — same entity, case/whitespace variants
#   3. unit_format_drift    — silent firmware/system change altered measurement units
#   4. orphaned_reference   — customer ID absent from master file
#   5. decimal_shift        — systematic ×10 / ÷10 weight corruption
#   6. impossible_value     — physically or logically impossible field value
PRIORS: dict[str, tuple[float, float]] = {
    "unit_format_drift":  (9.0, 1.0),   # ~90 % — silent, systematic sensor/firmware change
    "decimal_shift":      (8.5, 1.5),   # ~85 % — systematic ×10 weight error poisons analytics
    "orphaned_reference": (8.0, 2.0),   # ~80 % — unknown customers = critical audit risk
    "impossible_value":   (7.0, 3.0),   # ~70 % — physically impossible data (dates, quantities)
    "near_duplicate":     (5.0, 5.0),   # ~50 % — case/whitespace variants break joins
    "exact_duplicate":    (4.0, 6.0),   # ~40 % — inflates counts but underlying data exists
}


def bayesian_severity(issue_type: str, affected_count: int) -> dict:
    """Return posterior mean + 95 % CI for audit severity using PyMC Beta conjugate.

    Update rule: volume evidence proportionally strengthens the alpha (criticality) parameter
    while leaving beta (non-criticality) fixed.  This preserves the domain-knowledge prior
    direction while letting high-volume issues earn higher scores.
    """
    alpha_prior, beta_prior = PRIORS.get(issue_type, (5.0, 5.0))

    # Volume evidence: scale 0→1 based on what fraction of the dataset is affected.
    # Cap at 1.0 so even a 100 % hit rate only doubles alpha.
    volume_fraction = affected_count / TOTAL_RECORDS
    evidence_boost = min(volume_fraction * 4, 1.0)   # 25 % of records → full boost

    alpha_post = alpha_prior * (1.0 + evidence_boost)
    beta_post = beta_prior   # leave beta fixed — our prior about non-criticality stands

    # pm.draw is fast — no MCMC required, just draws from the analytical distribution.
    samples = pm.draw(pm.Beta.dist(alpha=alpha_post, beta=beta_post), draws=4000, random_seed=42)

    return {
        "mean": float(np.mean(samples)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "alpha_posterior": round(alpha_post, 2),
        "beta_posterior": round(beta_post, 2),
    }


def priority_label(severity_mean: float) -> str:
    if severity_mean >= 0.75:
        return "CRITICAL"
    if severity_mean >= 0.55:
        return "HIGH"
    if severity_mean >= 0.35:
        return "MEDIUM"
    return "LOW"


async def rank_issues(issues: list[dict] | None = None) -> list[dict]:
    if issues is None:
        with open(OUTPUT_DIR / "findings.json") as f:
            issues = json.load(f)

    print("[Agent 2 — Rank It] Scoring severity with PyMC Beta posteriors …")

    ranked: list[dict] = []
    for issue in issues:
        sev = bayesian_severity(issue["type"], issue["count"])
        ranked.append({
            **issue,
            "severity": sev,
            "priority": priority_label(sev["mean"]),
        })

    ranked.sort(key=lambda x: x["severity"]["mean"], reverse=True)

    # Store in Cognee (vector-only: add without cognify — zero LLM calls)
    ranking_summary = (
        "Agent 2 (Rank It) severity rankings (PyMC Bayesian Beta posteriors): "
        + "; ".join(
            f"{r['type']}/{r['subtype']} → {r['priority']} ({r['severity']['mean']:.2f} "
            f"[{r['severity']['ci_low']:.2f}–{r['severity']['ci_high']:.2f}])"
            for r in ranked
        )
    )
    await cognee.add(ranking_summary, dataset_name="data_rescue")

    with open(OUTPUT_DIR / "rankings.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)

    print(f"[Agent 2 — Rank It] Ranked {len(ranked)} issue types:")
    for r in ranked:
        sev = r["severity"]
        print(
            f"  {r['priority']:8s}  {r['type']}/{r['subtype']}  "
            f"severity={sev['mean']:.2f} CI=[{sev['ci_low']:.2f}, {sev['ci_high']:.2f}]  "
            f"n={r['count']}"
        )

    return ranked
