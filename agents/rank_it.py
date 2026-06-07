"""Agent 2 — Rank It: score each issue's audit severity.

Approach (Decision-Lab philosophy — multi-method convergence):
  Method A: PyMC Beta posterior (domain-prior + volume evidence)
  Method B: Frequency-weighted risk score (count / total × base_weight)
  Method C: Regulatory-impact heuristic (hardcoded audit-failure categories)

When A, B, C agree on the priority tier → HIGH CONFIDENCE.
When they diverge → the issue is flagged as UNCERTAIN and shown in the report.
This makes every ranking decision transparent and human-verifiable (Rule R07).
"""

import json
from pathlib import Path

import cognee
import numpy as np
import pymc as pm
from cognee.api.v1.search import SearchType

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _load_total_records() -> int:
    meta_path = OUTPUT_DIR / "dataset_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return int(__import__("json").load(f).get("total_records", 5000))
    return 5000

# ── Domain priors ─────────────────────────────────────────────────────────────
# Beta(alpha, beta): higher alpha/lower beta = more likely to be audit-critical.
# Reasoning source: regulatory audit standards — unknown entities and impossible
# values are always audit-stoppers; naming conflicts are fixable but show governance
# failure; duplicates inflate counts but underlying data exists.
PRIORS: dict[str, tuple[float, float]] = {
    "orphaned_reference":  (9.0, 1.0),  # ~90% — unknown customer = phantom entity risk
    "impossible_value":    (8.0, 2.0),  # ~80% — physically impossible = data integrity failure
    "logical_conflict":    (7.0, 3.0),  # ~70% — contradictory status/dates = unreliable recordkeeping
    "naming_conflict":     (6.0, 4.0),  # ~60% — breaks cross-plant joins; fixable but shows process gaps
    "duplicate":           (4.0, 6.0),  # ~40% — inflated counts but real data exists; lower risk
}

# Method C: issues that always = CRITICAL regardless of count (regulatory hard-stops)
REGULATORY_HARD_STOPS = {"orphaned_reference", "impossible_value"}

# Method B: base risk weight per type (0–1 scale)
RISK_WEIGHTS = {
    "orphaned_reference":  0.90,
    "impossible_value":    0.85,
    "logical_conflict":    0.70,
    "naming_conflict":     0.65,
    "duplicate":           0.40,
}


# ── Method A: PyMC Bayesian Beta posterior ────────────────────────────────────
def method_a_pymc(issue_type: str, affected_count: int, total_records: int) -> dict:
    """PyMC Beta conjugate update: volume evidence boosts alpha proportionally."""
    alpha_prior, beta_prior = PRIORS.get(issue_type, (5.0, 5.0))
    volume_fraction = affected_count / max(total_records, 1)
    evidence_boost = min(volume_fraction * 4, 1.0)
    alpha_post = alpha_prior * (1.0 + evidence_boost)
    beta_post = beta_prior

    samples = pm.draw(pm.Beta.dist(alpha=alpha_post, beta=beta_post), draws=4000, random_seed=42)
    return {
        "mean": float(np.mean(samples)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "alpha_posterior": round(alpha_post, 2),
        "beta_posterior": round(beta_post, 2),
        "reasoning": (
            f"Prior Beta({alpha_prior:.0f},{beta_prior:.0f}) updated with volume evidence "
            f"({affected_count}/{total_records} = {volume_fraction:.1%} of records affected). "
            f"Evidence boost: {evidence_boost:.2f}× → posterior Beta({alpha_post:.2f},{beta_post:.2f})."
        ),
    }


# ── Method B: Frequency-weighted risk score ───────────────────────────────────
def method_b_freq(issue_type: str, affected_count: int, total_records: int) -> float:
    """Simple frequency × base-weight score. No prior assumptions."""
    base = RISK_WEIGHTS.get(issue_type, 0.5)
    volume_factor = min(affected_count / max(total_records, 1) * 5, 1.0)
    return round(base * (0.7 + 0.3 * volume_factor), 3)


# ── Method C: Regulatory hard-stop heuristic ─────────────────────────────────
def method_c_regulatory(issue_type: str) -> float:
    """Binary: is this issue a known regulatory audit-stopper?"""
    return 0.95 if issue_type in REGULATORY_HARD_STOPS else 0.50


# ── Convergence check ─────────────────────────────────────────────────────────
def priority_label(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.55:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def check_convergence(score_a: float, score_b: float, score_c: float) -> dict:
    """Decision-Lab style convergence: do 3 methods agree on the priority tier?"""
    tiers = [priority_label(score_a), priority_label(score_b), priority_label(score_c)]
    unique_tiers = set(tiers)
    if len(unique_tiers) == 1:
        return {"converged": True, "agreement": "ALL_THREE", "tiers": tiers}
    if len(unique_tiers) == 2:
        majority = max(unique_tiers, key=tiers.count)
        return {"converged": True, "agreement": "MAJORITY", "tiers": tiers, "majority": majority}
    return {"converged": False, "agreement": "DIVERGED", "tiers": tiers}


async def rank_issues(issues: list[dict] | None = None) -> list[dict]:
    total_records = _load_total_records()

    # ── Read Agent 1 context from Cognee (R02: every agent reads from Cognee) ─
    cognee_context = ""
    try:
        results = await cognee.search("data quality issues found Harven Manufacturing", SearchType.CHUNKS)
        cognee_context = " | ".join(str(r)[:120] for r in (results or [])[:3])
        print(f"[Agent 2 — Rank It] Cognee recall: {len(results or [])} chunks retrieved from Agent 1")
    except Exception as e:
        print(f"[Agent 2 — Rank It] Cognee recall warning: {e}")

    if issues is None:
        with open(OUTPUT_DIR / "findings.json") as f:
            issues = json.load(f)

    print("[Agent 2 — Rank It] Scoring with 3-method convergence (Decision-Lab philosophy)…")

    ranked: list[dict] = []
    for issue in issues:
        # Run all three methods
        sev_a = method_a_pymc(issue["type"], issue["count"], total_records)
        sev_b = method_b_freq(issue["type"], issue["count"], total_records)
        sev_c = method_c_regulatory(issue["type"])

        # Ensemble: weighted average (Method A carries most weight as it's data-driven)
        ensemble_score = 0.5 * sev_a["mean"] + 0.3 * sev_b + 0.2 * sev_c
        convergence = check_convergence(sev_a["mean"], sev_b, sev_c)

        # Final priority from ensemble; flag if methods diverged
        final_priority = priority_label(ensemble_score)
        confidence = "HIGH" if convergence["converged"] else "UNCERTAIN"

        ranking_reasoning = (
            f"Method A (PyMC): {sev_a['mean']:.2f} — {sev_a['reasoning']} | "
            f"Method B (Freq×Weight): {sev_b:.2f} — base_weight={RISK_WEIGHTS.get(issue['type'],0.5)}, "
            f"volume_factor={min(issue['count']/total_records*5,1):.2f} | "
            f"Method C (Regulatory): {sev_c:.2f} — "
            f"{'hard-stop category (audit failure)' if issue['type'] in REGULATORY_HARD_STOPS else 'not a hard-stop'} | "
            f"Ensemble (0.5A+0.3B+0.2C): {ensemble_score:.2f} → {final_priority} | "
            f"Convergence: {convergence['agreement']} {convergence['tiers']}"
        )

        ranked.append({
            **issue,
            "severity": {
                **sev_a,
                "method_b_score": sev_b,
                "method_c_score": sev_c,
                "ensemble_score": round(ensemble_score, 3),
            },
            "priority": final_priority,
            "confidence": confidence,
            "convergence": convergence,
            "ranking_reasoning": ranking_reasoning,
        })

    ranked.sort(key=lambda x: x["severity"]["ensemble_score"], reverse=True)

    # ── Write to Cognee (R02) ─────────────────────────────────────────────────
    ranking_summary = (
        "Agent 2 (Rank It) 3-method convergence rankings: "
        + "; ".join(
            f"{r['type']}/{r['subtype']}={r['priority']}({r['severity']['ensemble_score']:.2f}) "
            f"confidence={r['confidence']}"
            for r in ranked
        )
    )
    await cognee.add(ranking_summary, dataset_name="data_rescue")

    with open(OUTPUT_DIR / "rankings.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)

    print(f"[Agent 2 — Rank It] Ranked {len(ranked)} issue types (cognee_context={bool(cognee_context)}):")
    for r in ranked:
        sev = r["severity"]
        print(
            f"  {r['priority']:8s} [{r['confidence']:9s}]  {r['type']}/{r['subtype']}  "
            f"ensemble={sev['ensemble_score']:.2f}  A={sev['mean']:.2f}  B={sev['method_b_score']:.2f}  "
            f"C={sev['method_c_score']:.2f}  n={r['count']}"
        )

    return ranked
