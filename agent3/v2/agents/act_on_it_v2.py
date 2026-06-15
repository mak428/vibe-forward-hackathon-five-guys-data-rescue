"""
Agent 3 — Act On It (v2)
========================
Inputs
------
  findings.json   — Agent 1 output: list of issue dicts, each with subtype,
                    columns, count, record_ids, severity, evidence, description
  rankings.json   — Agent 2 output: same list enriched with a `severity` dict
                    (mean, ci_low, ci_high, alpha_posterior, beta_posterior)
                    and a `priority` label (CRITICAL / HIGH / MEDIUM / LOW)
  data/*.csv      — raw data files (primary + optional reference tables)

Outputs
-------
  output/track01_cleaned.csv  — remediated primary table
  output/audit_log.json       — per-issue action record with:
                                  • action taken (AUTO_FIXED / FLAGGED / ESCALATED / SKIPPED)
                                  • Bayesian confidence % + 94 % HDI
                                  • plain-English explanation

Design principles
-----------------
1. **No hardcoded domain priors.**  The Beta(α, β) prior for each issue is taken
   *directly* from Agent 2's posterior parameters (alpha_posterior, beta_posterior).
   Agent 2 already did the Bayesian severity scoring; we reuse that distribution
   as our prior and update it with structural evidence extracted from the data.

2. **Structural evidence updates the posterior.**  For each issue class we compute
   an evidence count (successes / trials) from the actual data — e.g. what fraction
   of a numeric column is positive, how tight the group spread is, how complete a
   foreign-key table is.  This shifts the posterior away from the prior when the
   data is informative.

3. **Common-sense rules gate actions without overriding the model.**  Rules that are
   domain-agnostic (non-negative columns should be non-negative; date A should
   precede date B if the data says so) feed the evidence computation rather than
   being hard-coded dispatch branches.  The model decides; the rules inform it.

4. **Confidence threshold is calibrated per action type.**
     AUTO_FIX  ≥ 0.90   (high bar — we are modifying data)
     FLAG      ≥ 0.45   (lower bar — we are annotating, not changing)
     ESCALATE  ≥ 0.55   (medium — sends to human, not destructive)

5. **pm.draw (no MCMC) for speed.**  Agent 2 already ran NUTS; here we need only
   analytical draws from the updated Beta to compute the posterior mean and HDI.
   This keeps Agent 3 fast even on large issue lists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

try:
    import cognee
except Exception:
    cognee = None

OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR   = Path(__file__).parent.parent / "data"

# Action confidence thresholds
THRESH_FIX      = 0.90
THRESH_ESCALATE = 0.55
THRESH_FLAG     = 0.45

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("act_on_it")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BayesianVerdict:
    """Posterior summary for one issue."""
    confidence: float       # posterior mean  (0–1)
    hdi_low: float          # 94 % HDI lower bound
    hdi_high: float         # 94 % HDI upper bound
    alpha_post: float
    beta_post: float
    evidence_successes: int
    evidence_trials: int
    evidence_note: str      # human-readable description of evidence used


@dataclass
class IssueResult:
    subtype: str
    columns: list[str]
    count: int
    priority: str
    action: str             # AUTO_FIXED | FLAGGED | ESCALATED | SKIPPED
    records_affected: int
    confidence_pct: float
    hdi_low_pct: float
    hdi_high_pct: float
    explanation: str
    fix_details: dict       # anything fix-specific (e.g. values swapped, rows dropped)


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian posterior computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_posterior(
    alpha_prior: float,
    beta_prior: float,
    successes: int,
    trials: int,
    draws: int = 4000,
    seed: int = 42,
) -> BayesianVerdict:
    """
    Update a Beta prior from Agent 2 with structural evidence (successes / trials)
    using the Beta-Binomial conjugate update:

        α_post = α_prior + successes
        β_post = β_prior + (trials - successes)

    Then draw from Beta(α_post, β_post) to get the posterior mean and HDI.
    pm.draw is used (no MCMC) — analytical Beta samples are essentially instant.
    """
    failures = max(trials - successes, 0)
    alpha_post = alpha_prior + successes
    beta_post  = beta_prior  + failures

    samples = pm.draw(
        pm.Beta.dist(alpha=alpha_post, beta=beta_post),
        draws=draws,
        random_seed=seed,
    )
    samples = np.asarray(samples).flatten()
    mean    = float(np.mean(samples))
    hdi     = np.percentile(samples, [3.0, 97.0])   # 94 % HDI

    return BayesianVerdict(
        confidence=mean,
        hdi_low=float(hdi[0]),
        hdi_high=float(hdi[1]),
        alpha_post=round(alpha_post, 3),
        beta_post=round(beta_post, 3),
        evidence_successes=successes,
        evidence_trials=trials,
        evidence_note="",   # filled in by callers
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evidence extractors — one per issue class
# Each returns (successes, trials, note) derived from the real data.
# successes = observations that SUPPORT the proposed fix being correct.
# trials    = total relevant observations inspected.
# ─────────────────────────────────────────────────────────────────────────────

def evidence_duplicate(df: pd.DataFrame, id_col: str) -> tuple[int, int, str]:
    """
    Evidence for dropping duplicates: how many rows are exact copies?
    The more redundant rows relative to total, the stronger the case to remove.
    """
    cols_no_id = [c for c in df.columns if c != id_col]
    n_dups = int(df.duplicated(subset=cols_no_id, keep="first").sum())
    trials     = len(df)
    successes  = n_dups          # every confirmed duplicate is a success for 'drop it'
    note = (
        f"Found {n_dups} exact-duplicate rows out of {trials} total "
        f"({n_dups/max(trials,1)*100:.1f}% of dataset). "
        f"High duplicate rate strengthens removal confidence."
    )
    return successes, trials, note


def evidence_format_drift(df: pd.DataFrame, col: str) -> tuple[int, int, str]:
    """
    Evidence for normalising a code/identifier column.
    successes = deviants (non-canonical spellings) — each one that clearly maps to
                a canonical form is strong evidence normalisation is correct.
    trials    = all non-null values in the column.
    """
    raw  = df[col].astype(str)
    norm = raw.str.replace(r"[-_\s]", "", regex=True).str.upper()
    mode_per_key = raw.groupby(norm).transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    deviants   = int((raw != mode_per_key).sum())
    total      = int(raw.notna().sum())
    # How many distinct canonical keys exist with a clear majority spelling?
    n_keys     = int(norm.nunique())
    n_resolved = int((raw.groupby(norm).transform("nunique") == 1).sum())
    note = (
        f"{deviants} of {total} '{col}' values deviate from the canonical spelling "
        f"across {n_keys} distinct keys. {n_resolved} values already have an "
        f"unambiguous canonical form to normalise to."
    )
    return deviants, total, note


def evidence_orphan(
    df: pd.DataFrame, fk_col: str, ref_values: set[str]
) -> tuple[int, int, str]:
    """
    Evidence for escalating orphaned references.
    trials    = total non-null FK values.
    successes = orphaned count (the anomaly *is* the evidence for escalation).
    High orphan rate → high confidence that escalation is correct.
    """
    vals    = df[fk_col].astype(str)
    orphans = int((~vals.isin(ref_values)).sum())
    total   = int(vals.notna().sum())
    known   = total - orphans
    note = (
        f"{orphans} of {total} '{fk_col}' values are absent from the reference table "
        f"({known} are valid). High unknown fraction increases escalation confidence."
    )
    return orphans, total, note


def evidence_non_positive(df: pd.DataFrame, col: str) -> tuple[int, int, str]:
    """
    Evidence for flagging negative/zero values in a column that should be positive.
    successes = values ≤ 0 (violations — each is direct evidence for the flag).
    trials    = all non-null numeric values.
    """
    s         = pd.to_numeric(df[col], errors="coerce").dropna()
    bad       = int((s <= 0).sum())
    total     = len(s)
    pos_frac  = (s > 0).mean()
    note = (
        f"{bad} of {total} '{col}' values are ≤ 0 while "
        f"{pos_frac*100:.1f}% of the column is positive. "
        f"Physical impossibility of negative values supports flagging."
    )
    return bad, total, note


def evidence_outlier(
    df: pd.DataFrame, col: str, group_col: str | None
) -> tuple[int, int, str]:
    """
    Evidence for flagging/fixing numeric outliers (decimal-shift / unit errors).
    Uses the same MAD-z + ratio logic as Agent 1, so evidence is consistent
    with what was reported.
    successes = confirmed outlier rows.
    trials    = all rows with a valid group median to compare against.
    """
    MAD_Z = 6.0
    RATIO = 4.0
    MIN_G = 4

    s = pd.to_numeric(df[col], errors="coerce")
    if group_col and group_col in df.columns:
        gkey = df[group_col].astype(str).str.replace(r"[-_\s]", "", regex=True).str.upper()
    else:
        gkey = pd.Series(["ALL"] * len(df), index=df.index)

    flagged, eligible = 0, 0
    for _, grp_idx in s.groupby(gkey).groups.items():
        grp = s.loc[grp_idx].dropna()
        if len(grp) < MIN_G or grp.median() == 0:
            continue
        eligible += len(grp)
        med = grp.median()
        mad = (grp - med).abs().median()
        if mad == 0:
            continue
        z     = (grp - med).abs() / (1.4826 * mad)
        ratio = grp / med
        flagged += int(((z > MAD_Z) & (ratio > RATIO)).sum())

    note = (
        f"{flagged} of {eligible} eligible '{col}' values are robust outliers "
        f"(>6 MAD-z and >4x group median). "
        f"Consistent with a decimal-shift or unit-conversion error."
    )
    return flagged, max(eligible, 1), note


def evidence_date_order(
    df: pd.DataFrame, col_a: str, col_b: str
) -> tuple[int, int, str]:
    """
    Evidence for fixing / flagging date-order violations.
    successes = violations (direct evidence that a fix is needed).
    trials    = rows where both dates are present.
    """
    da = pd.to_datetime(df[col_a], errors="coerce")
    db = pd.to_datetime(df[col_b], errors="coerce")
    both     = da.notna() & db.notna()
    total    = int(both.sum())
    viol     = int((da[both] > db[both]).sum())
    gap_days = (da[both & (da > db)] - db[both & (da > db)]).dt.days
    med_gap  = float(gap_days.median()) if len(gap_days) else 0.0
    note = (
        f"{viol} of {total} rows violate the expected '{col_a} ≤ {col_b}' ordering. "
        f"Median gap for violations: {med_gap:.0f} days. "
        f"Small gaps are consistent with field transpositions; large gaps suggest deeper corruption."
    )
    return viol, max(total, 1), note


# ─────────────────────────────────────────────────────────────────────────────
# Fix executors — each returns (df_modified, n_changed, detail_dict)
# ─────────────────────────────────────────────────────────────────────────────

def fix_exact_duplicates(
    df: pd.DataFrame, id_col: str
) -> tuple[pd.DataFrame, int, dict]:
    cols_no_id = [c for c in df.columns if c != id_col]
    drop_idx   = df[df.duplicated(subset=cols_no_id, keep="first")].index.tolist()
    df_out     = df.drop(index=drop_idx).reset_index(drop=True)
    return df_out, len(drop_idx), {"dropped_indices": drop_idx}


def fix_format_drift(
    df: pd.DataFrame, col: str, subtype: str
) -> tuple[pd.DataFrame, int, dict]:
    """Normalise separators and/or casing to the canonical (most-common) spelling."""
    raw  = df[col].astype(str)
    norm = raw.str.replace(r"[-_\s]", "", regex=True).str.upper()
    mode_per_key = raw.groupby(norm).transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    mask         = raw != mode_per_key
    before       = df.loc[mask, col].tolist()
    df[col]      = mode_per_key
    after        = df.loc[mask, col].tolist()
    n_changed    = int(mask.sum())
    return df, n_changed, {
        "sample_before": before[:10],
        "sample_after":  after[:10],
        "normalisation": "canonical mode per normalised key",
    }


def fix_date_transposition(
    df: pd.DataFrame, col_a: str, col_b: str, max_gap_days: int = 7
) -> tuple[pd.DataFrame, int, int, dict]:
    """
    Swap col_a / col_b where col_a > col_b AND the gap ≤ max_gap_days.
    Returns (df, n_swapped, n_unfixed, details).
    """
    da   = pd.to_datetime(df[col_a], errors="coerce")
    db   = pd.to_datetime(df[col_b], errors="coerce")
    viol = (da > db) & da.notna() & db.notna()
    gap  = (da - db).dt.days

    swappable = viol & (gap <= max_gap_days)
    unfixable = viol & (gap >  max_gap_days)

    df.loc[swappable, [col_a, col_b]] = (
        df.loc[swappable, [col_b, col_a]].values
    )
    return df, int(swappable.sum()), int(unfixable.sum()), {
        "swap_threshold_days": max_gap_days,
        "swapped": int(swappable.sum()),
        "still_violated": int(unfixable.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Explain builder — produces a single human-readable string
# ─────────────────────────────────────────────────────────────────────────────

def build_explanation(
    action: str,
    subtype: str,
    verdict: BayesianVerdict,
    fix_details: dict,
    issue: dict,
    threshold_used: float,
) -> str:
    conf = verdict.confidence * 100
    hdi  = f"{verdict.hdi_low*100:.1f}%–{verdict.hdi_high*100:.1f}%"
    prior_src = (
        f"Prior α={issue['severity']['alpha_posterior']:.2f}, "
        f"β={issue['severity']['beta_posterior']:.2f} (from Agent 2 Bayesian ranking)"
    )
    evidence_src = (
        f"Evidence: {verdict.evidence_successes} successes / "
        f"{verdict.evidence_trials} trials from data. "
        f"{verdict.evidence_note}"
    )
    posterior_src = (
        f"Posterior: α={verdict.alpha_post:.2f}, β={verdict.beta_post:.2f} → "
        f"confidence {conf:.1f}% (94% HDI {hdi})"
    )

    if action == "AUTO_FIXED":
        action_line = (
            f"FIX APPLIED (confidence {conf:.1f}% ≥ threshold {threshold_used*100:.0f}%). "
        )
    elif action == "FLAGGED":
        action_line = (
            f"FLAGGED for review (confidence {conf:.1f}% ≥ {threshold_used*100:.0f}%, "
            f"below AUTO_FIX threshold). "
        )
    elif action == "ESCALATED":
        action_line = (
            f"ESCALATED to compliance (confidence {conf:.1f}% ≥ {threshold_used*100:.0f}%). "
        )
    else:
        action_line = (
            f"SKIPPED — confidence {conf:.1f}% below all action thresholds. "
            f"Issue recorded but no change made. "
        )

    details_line = ""
    if fix_details:
        parts = []
        for k, v in fix_details.items():
            if isinstance(v, list) and v:
                parts.append(f"{k}: {v[:5]}")
            elif not isinstance(v, list):
                parts.append(f"{k}: {v}")
        if parts:
            details_line = "Details — " + "; ".join(parts) + ". "

    return (
        f"[{subtype}] {action_line}"
        f"{prior_src}. "
        f"{evidence_src} "
        f"{posterior_src}. "
        f"{details_line}"
        f"Agent 1 description: {issue.get('description', '')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Issue dispatcher — one branch per canonical subtype
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_issue(
    issue: dict,
    df: pd.DataFrame,
    id_col: str,
    ref_tables: dict[str, pd.DataFrame],
    indices_to_drop: set[int],
    flags: dict[str, list[str]],
) -> IssueResult:
    subtype  = issue["subtype"]
    columns  = issue.get("columns", [])
    count    = issue["count"]
    priority = issue.get("priority", "MEDIUM")
    rec_ids  = issue.get("record_ids", [])

    # Agent 2 always provides alpha_posterior / beta_posterior.
    # Fall back to flat Beta(5,5) if missing (shouldn't happen in practice).
    sev          = issue.get("severity", {})
    alpha_prior  = float(sev.get("alpha_posterior", 5.0))
    beta_prior   = float(sev.get("beta_posterior",  5.0))

    fix_details: dict = {}

    # ── 1. Exact / semantic duplicates ───────────────────────────────────────
    if subtype in ("exact_duplicate", "semantic_duplicate"):
        suc, tri, note = evidence_duplicate(df, id_col)
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        if verdict.confidence >= THRESH_FIX:
            drop_idx = df[df.duplicated(
                subset=[c for c in df.columns if c != id_col], keep="first"
            )].index.tolist()
            indices_to_drop.update(drop_idx)
            action, thr = "AUTO_FIXED", THRESH_FIX
            fix_details = {"rows_scheduled_for_drop": len(drop_idx)}
        elif verdict.confidence >= THRESH_FLAG:
            for rid in rec_ids:
                flags.setdefault(rid, []).append("DUPLICATE")
            action, thr = "FLAGGED", THRESH_FLAG
        else:
            action, thr = "SKIPPED", THRESH_FIX

    # ── 2. Near-duplicates / format variants / separator drift ───────────────
    elif subtype in (
        "near_duplicate_variant", "separator_mismatch",
        "malformed_part_number", "separator_drift", "format_variant",
    ):
        col = columns[0] if columns else None
        if col and col in df.columns:
            suc, tri, note = evidence_format_drift(df, col)
        else:
            suc, tri, note = count, count * 2, f"No specific column context for '{subtype}'"
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        if verdict.confidence >= THRESH_FIX and col and col in df.columns:
            df_new, n_changed, fd = fix_format_drift(df, col, subtype)
            df.update(df_new)
            df[col] = df_new[col]
            fix_details = fd
            action, thr = "AUTO_FIXED", THRESH_FIX
        elif verdict.confidence >= THRESH_FLAG:
            for rid in rec_ids:
                flags.setdefault(rid, []).append("FORMAT_ISSUE")
            action, thr = "FLAGGED", THRESH_FLAG
        else:
            action, thr = "SKIPPED", THRESH_FIX

    # ── 3. Orphaned references ────────────────────────────────────────────────
    elif subtype in ("unknown_reference", "unknown_customer_id"):
        col = columns[0] if columns else None
        # Find the reference set from the loaded ref tables
        ref_vals: set[str] = set()
        for rt in ref_tables.values():
            for rc in rt.columns:
                if rc == col or rt[rc].is_unique:
                    ref_vals = set(rt[rc].astype(str))
                    break
            if ref_vals:
                break

        if col and col in df.columns and ref_vals:
            suc, tri, note = evidence_orphan(df, col, ref_vals)
        else:
            suc, tri, note = count, count + 10, f"Reference table not available for '{col}'"
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        # Orphaned references are never auto-fixed — escalate or flag.
        if verdict.confidence >= THRESH_ESCALATE:
            for rid in rec_ids:
                flags.setdefault(rid, []).append("ORPHANED_REFERENCE")
            action, thr = "ESCALATED", THRESH_ESCALATE
        elif verdict.confidence >= THRESH_FLAG:
            for rid in rec_ids:
                flags.setdefault(rid, []).append("ORPHANED_REFERENCE")
            action, thr = "FLAGGED", THRESH_FLAG
        else:
            action, thr = "SKIPPED", THRESH_ESCALATE
        fix_details = {"unknown_ids": issue.get("unknown_ids", [])[:20]}

    # ── 4. Non-positive values (physically impossible) ────────────────────────
    elif subtype in ("non_positive_value", "non_positive_quantity"):
        col = columns[0] if columns else None
        if col and col in df.columns:
            suc, tri, note = evidence_non_positive(df, col)
        else:
            suc, tri, note = count, count * 2, "Column not found"
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        # Can't auto-correct sign without knowing the true value — always flag.
        for rid in rec_ids:
            flags.setdefault(rid, []).append("IMPOSSIBLE_VALUE")
        action = "FLAGGED" if verdict.confidence >= THRESH_FLAG else "SKIPPED"
        thr     = THRESH_FLAG

    # ── 5. Numeric outliers (decimal-shift / unit-conversion errors) ──────────
    elif subtype in ("numeric_outlier", "weight_inflation", "decimal_shift"):
        col       = columns[0] if columns else None
        group_col = columns[1] if len(columns) > 1 else None
        # Try to infer the best grouping column from evidence in the issue
        if col and col in df.columns:
            suc, tri, note = evidence_outlier(df, col, group_col)
        else:
            suc, tri, note = count, count * 2, "Column not found"
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        # Outliers are flagged (not auto-fixed) because we cannot determine the
        # true unit or magnitude without a domain expert.
        for rid in rec_ids:
            flags.setdefault(rid, []).append("OUTLIER_VALUE")
        action = "FLAGGED" if verdict.confidence >= THRESH_FLAG else "SKIPPED"
        thr     = THRESH_FLAG
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            fix_details = {
                "column_median": round(float(s.median()), 4),
                "column_mean":   round(float(s.mean()), 4),
                "note": "Auto-correction withheld: true unit/magnitude requires SME confirmation.",
            }

    # ── 6. Date-order violations ──────────────────────────────────────────────
    elif subtype in ("date_order_violation", "ship_before_production"):
        col_a = columns[0] if len(columns) > 0 else None
        col_b = columns[1] if len(columns) > 1 else None
        if col_a and col_b and col_a in df.columns and col_b in df.columns:
            suc, tri, note = evidence_date_order(df, col_a, col_b)
        else:
            suc, tri, note = count, count + 10, "Date columns not found"
        verdict = compute_posterior(alpha_prior, beta_prior, suc, tri)
        verdict.evidence_note = note

        if verdict.confidence >= THRESH_FIX and col_a and col_b:
            df_new, n_swap, n_unfixed, fd = fix_date_transposition(df, col_a, col_b)
            df[col_a] = df_new[col_a]
            df[col_b] = df_new[col_b]
            fix_details = fd
            if n_unfixed > 0:
                # Rows with large gaps still get flagged even after partial fix
                da = pd.to_datetime(df[col_a], errors="coerce")
                db = pd.to_datetime(df[col_b], errors="coerce")
                still_bad = df[(da > db) & da.notna() & db.notna()][id_col].astype(str).tolist()
                for rid in still_bad:
                    flags.setdefault(rid, []).append("DATE_INVERSION")
            action, thr = "AUTO_FIXED", THRESH_FIX
        elif verdict.confidence >= THRESH_FLAG:
            for rid in rec_ids:
                flags.setdefault(rid, []).append("DATE_INVERSION")
            action, thr = "FLAGGED", THRESH_FLAG
        else:
            action, thr = "SKIPPED", THRESH_FIX

    # ── Catch-all for any unrecognised subtype ────────────────────────────────
    else:
        verdict = compute_posterior(alpha_prior, beta_prior, count, count * 2)
        verdict.evidence_note = f"No structured evidence extractor for subtype '{subtype}'."
        for rid in rec_ids:
            flags.setdefault(rid, []).append("UNCLASSIFIED_ISSUE")
        action, thr = "FLAGGED", THRESH_FLAG

    explanation = build_explanation(action, subtype, verdict, fix_details, issue, thr)
    log.info(
        "  [%s] %s/%s  n=%d  conf=%.1f%%  HDI=[%.1f%%,%.1f%%]",
        action, issue.get("type","?"), subtype, count,
        verdict.confidence*100, verdict.hdi_low*100, verdict.hdi_high*100,
    )

    return IssueResult(
        subtype=subtype,
        columns=columns,
        count=count,
        priority=priority,
        action=action,
        records_affected=count,
        confidence_pct=round(verdict.confidence * 100, 2),
        hdi_low_pct=round(verdict.hdi_low * 100, 2),
        hdi_high_pct=round(verdict.hdi_high * 100, 2),
        explanation=explanation,
        fix_details=fix_details,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tables() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    frames = {f.name: pd.read_csv(f, dtype=str, keep_default_na=False) for f in csvs}
    primary_name = max(frames, key=lambda k: len(frames[k]))
    primary = frames[primary_name]
    refs = {k: v for k, v in frames.items() if k != primary_name}
    log.info("Primary table: %s (%d rows × %d cols)", primary_name, *primary.shape)
    return primary, refs


def _infer_id_col(df: pd.DataFrame) -> str:
    ID_HINT = re.compile(r"(^id$|_id$|^id_|record|uuid|guid|key$|_key$)", re.I)
    unique_cols = [c for c in df.columns if df[c].is_unique]
    named = [c for c in unique_cols if ID_HINT.search(c)]
    if named:
        return named[0]
    if unique_cols:
        return unique_cols[0]
    return df.columns[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def act_on_issues(ranked: list[dict] | None = None) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load Agent 2 output (rankings with Bayesian severity already computed)
    if ranked is None:
        with open(OUTPUT_DIR / "rankings.json") as f:
            ranked = json.load(f)
    log.info("Loaded %d ranked issues from Agent 2", len(ranked))

    # Load data
    df, ref_tables = _load_tables()
    id_col = _infer_id_col(df)
    log.info("ID column inferred as '%s'", id_col)

    indices_to_drop: set[int] = set()
    flags: dict[str, list[str]] = {}
    results: list[IssueResult] = []

    # Process issues in priority order (CRITICAL first, then severity mean desc)
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    ordered = sorted(
        ranked,
        key=lambda x: (
            priority_order.get(x.get("priority", "LOW"), 4),
            -x.get("severity", {}).get("mean", 0),
        ),
    )

    for issue in ordered:
        log.info(
            "Processing: %s/%s  priority=%s  n=%d",
            issue.get("type","?"), issue.get("subtype","?"),
            issue.get("priority","?"), issue.get("count", 0),
        )
        result = dispatch_issue(issue, df, id_col, ref_tables, indices_to_drop, flags)
        results.append(result)

    # ── Apply pending row drops ───────────────────────────────────────────────
    n_dropped = len(indices_to_drop)
    if indices_to_drop:
        df = df.drop(index=list(indices_to_drop)).reset_index(drop=True)

    # ── Annotate with audit columns ───────────────────────────────────────────
    df["audit_flags"] = df[id_col].map(
        lambda r: "; ".join(flags.get(str(r), [])) or ""
    )
    df["audit_status"] = df[id_col].map(
        lambda r: (
            "ESCALATED" if "ORPHANED_REFERENCE" in flags.get(str(r), [])
            else ("FLAGGED" if flags.get(str(r)) else "CLEAN")
        )
    )

    # ── Save cleaned CSV ──────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "track01_cleaned.csv"
    df.to_csv(out_csv, index=False)
    log.info("Saved cleaned CSV → %s  (%d rows)", out_csv, len(df))

    # ── Build and save audit log ──────────────────────────────────────────────
    n_escalated = int((df["audit_status"] == "ESCALATED").sum())
    n_flagged   = int((df["audit_status"] == "FLAGGED").sum())
    n_clean     = int((df["audit_status"] == "CLEAN").sum())

    audit_log = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_input_records": len(df) + n_dropped,
        "duplicates_removed": n_dropped,
        "output_records": len(df),
        "clean": n_clean,
        "flagged": n_flagged,
        "escalated": n_escalated,
        "confidence_thresholds": {
            "auto_fix": THRESH_FIX,
            "escalate": THRESH_ESCALATE,
            "flag": THRESH_FLAG,
        },
        "issues": [asdict(r) for r in results],
    }
    with open(OUTPUT_DIR / "audit_log.json", "w") as f:
        json.dump(audit_log, f, indent=2, default=str)
    log.info("Audit log → %s", OUTPUT_DIR / "audit_log.json")

    # ── Cognee (guarded) ──────────────────────────────────────────────────────
    summary_text = (
        f"Agent 3 (Act On It v2) results: "
        f"input={len(df)+n_dropped}, removed={n_dropped} duplicates, "
        f"output={len(df)} — {n_clean} CLEAN / {n_flagged} FLAGGED / {n_escalated} ESCALATED. "
        + "; ".join(
            f"{r.action} {r.records_affected} ({r.subtype}) conf={r.confidence_pct:.1f}%"
            for r in results
        )
    )
    if cognee is not None:
        try:
            await cognee.add(summary_text, dataset_name="data_rescue")
        except Exception as e:
            log.warning("Cognee write skipped: %s", e)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  AGENT 3 — ACT ON IT (v2)  |  Bayesian confidence-gated remediation")
    print(f"{'='*72}")
    print(f"  Input records  : {len(df) + n_dropped}")
    print(f"  Dups removed   : {n_dropped}")
    print(f"  Output records : {len(df)}")
    print(f"  CLEAN={n_clean}  FLAGGED={n_flagged}  ESCALATED={n_escalated}")
    print(f"\n  {'PRIORITY':<10} {'ACTION':<12} {'SUBTYPE':<30} {'CONF%':>6}  {'HDI':<18}  n")
    print(f"  {'-'*85}")
    for r in results:
        hdi = f"[{r.hdi_low_pct:.1f}–{r.hdi_high_pct:.1f}%]"
        print(
            f"  {r.priority:<10} {r.action:<12} {r.subtype:<30} "
            f"{r.confidence_pct:>5.1f}%  {hdi:<18}  {r.count}"
        )
    print(f"\n  Full explanations in output/audit_log.json")
    print(f"{'='*72}\n")

    return audit_log


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point (sync wrapper so you can run: python act_on_it_v2.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(act_on_issues())
