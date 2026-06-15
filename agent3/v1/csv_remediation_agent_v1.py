"""
CSV Data Remediation Agent (Agent 3)
======================================
Inputs:
  - Raw CSV files (one or more)
  - Agent 1 output : anomaly detection report  (JSON)
  - Agent 2 output : ranked priority report     (JSON)

Output:
  - Remediated CSV files
  - Remediation report (JSON) with Bayesian confidence % and fix explanations

Bayesian modeling is performed via PyMC (pymc-labs/pymc-modeling).
For each anomaly, a Beta-Binomial model is fitted using:
  - a prior derived from the anomaly's severity/type
  - likelihood evidence drawn from the surrounding data context
The posterior mean is reported as the confidence level in each fix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("remediation_agent")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """Single anomaly record produced by Agent 1."""
    anomaly_id: str
    file: str
    row: int
    column: str
    anomaly_type: str          # e.g. "missing", "outlier", "type_mismatch", "duplicate"
    original_value: Any
    description: str
    severity: float = 0.5      # 0–1, higher = worse


@dataclass
class PriorityItem:
    """Single priority record produced by Agent 2."""
    anomaly_id: str
    rank: int                  # 1 = most critical
    priority_score: float      # 0–1


@dataclass
class RemediationResult:
    """Result of remediating one anomaly."""
    anomaly_id: str
    file: str
    row: int
    column: str
    anomaly_type: str
    original_value: Any
    fixed_value: Any
    confidence_pct: float      # Bayesian posterior mean × 100
    hdi_low_pct: float         # 94 % HDI lower bound × 100
    hdi_high_pct: float        # 94 % HDI upper bound × 100
    fix_applied: bool
    explanation: str
    priority_rank: int | None = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csvs(paths: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for p in paths:
        name = Path(p).name
        frames[name] = pd.read_csv(p)
        log.info("Loaded CSV '%s'  (%d rows × %d cols)", name, *frames[name].shape)
    return frames


def load_json(path: str, label: str) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    log.info("Loaded %s report: %d items", label, len(data))
    return data


def parse_anomalies(raw: list[dict]) -> list[Anomaly]:
    out = []
    for item in raw:
        out.append(Anomaly(
            anomaly_id=str(item["anomaly_id"]),
            file=item["file"],
            row=int(item["row"]),
            column=item["column"],
            anomaly_type=item["anomaly_type"].lower(),
            original_value=item.get("original_value"),
            description=item.get("description", ""),
            severity=float(item.get("severity", 0.5)),
        ))
    return out


def parse_priorities(raw: list[dict]) -> dict[str, PriorityItem]:
    out: dict[str, PriorityItem] = {}
    for item in raw:
        aid = str(item["anomaly_id"])
        out[aid] = PriorityItem(
            anomaly_id=aid,
            rank=int(item["rank"]),
            priority_score=float(item.get("priority_score", 0.5)),
        )
    return out


# ---------------------------------------------------------------------------
# Bayesian confidence estimation
# ---------------------------------------------------------------------------

# Alpha / Beta priors (Beta distribution) per anomaly type.
# The Beta(α, β) prior encodes our baseline belief that a candidate fix
# is correct *before* we look at the data context.
#
#   α  = pseudo-count of "this fix would be correct"
#   β  = pseudo-count of "this fix would be wrong"
#
# Type priors are calibrated domain-heuristics; tune them for your data.

TYPE_PRIORS: dict[str, tuple[float, float]] = {
    "missing":        (8.0, 2.0),   # imputing missing values is usually safe
    "outlier":        (5.0, 3.0),   # trimming outliers — moderate confidence
    "type_mismatch":  (9.0, 1.0),   # coercing types is nearly always correct
    "duplicate":      (7.0, 2.0),   # dropping duplicates is generally safe
    "format_error":   (8.0, 2.0),   # reformatting is high confidence
    "range_violation":(6.0, 3.0),   # clamping to range — moderate
    "default":        (5.0, 5.0),   # flat / uncertain prior
}


def _context_evidence(
    df: pd.DataFrame,
    col: str,
    row: int,
    anomaly_type: str,
) -> tuple[int, int]:
    """
    Derive (successes, failures) from the surrounding column context.

    These counts augment the prior so the posterior reflects both our
    domain belief AND empirical evidence from the actual data.

    Returns
    -------
    successes : int  — evidence supporting the fix
    failures  : int  — evidence against the fix
    """
    series = df[col] if col in df.columns else pd.Series([], dtype=object)

    successes, failures = 0, 0

    if anomaly_type in ("missing",):
        non_null = series.dropna()
        # More non-null neighbours → imputing is more defensible
        successes = min(int(len(non_null) * 0.3), 20)
        failures  = max(0, 5 - successes)

    elif anomaly_type in ("outlier", "range_violation"):
        if pd.api.types.is_numeric_dtype(series):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            inliers = series[(series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)]
            frac_inlier = len(inliers) / max(len(series), 1)
            # High inlier fraction → outlier fix is more confident
            successes = int(frac_inlier * 15)
            failures  = int((1 - frac_inlier) * 10)
        else:
            successes, failures = 3, 3

    elif anomaly_type == "type_mismatch":
        # Count how many other cells in the column already have the right type
        expected_type = series.dropna().dtype
        correct = series.dropna().apply(
            lambda v: isinstance(v, (int, float)) if pd.api.types.is_numeric_dtype(expected_type)
            else isinstance(v, str)
        )
        successes = int(correct.sum() * 0.2)
        failures  = max(0, 3 - successes)

    elif anomaly_type == "duplicate":
        dup_frac = series.duplicated().mean()
        # Low duplication overall → removing this duplicate is very safe
        successes = int((1 - dup_frac) * 15)
        failures  = int(dup_frac * 10)

    else:
        successes, failures = 3, 3

    return max(successes, 1), max(failures, 1)


def bayesian_confidence(
    anomaly: Anomaly,
    df: pd.DataFrame,
    n_samples: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """
    Fit a Beta-Binomial model in PyMC and return the posterior mean
    and 94 % HDI as (mean, hdi_low, hdi_high) — all in [0, 1].

    Model
    -----
    θ  ~ Beta(α_prior, β_prior)          # prior over P(fix is correct)
    y  ~ Binomial(n_context, θ)          # context observations
    """
    atype = anomaly.anomaly_type if anomaly.anomaly_type in TYPE_PRIORS else "default"
    alpha_prior, beta_prior = TYPE_PRIORS[atype]

    # Boost / shrink prior by severity (high severity → we must act → prior shifts up)
    severity_boost = anomaly.severity * 2.0
    alpha_prior = alpha_prior + severity_boost

    suc, fail = _context_evidence(df, anomaly.column, anomaly.row, anomaly.anomaly_type)

    with pm.Model() as model:                               # noqa: F841
        theta = pm.Beta("theta", alpha=alpha_prior, beta=beta_prior)
        n_obs = suc + fail
        _ = pm.Binomial("y_obs", n=n_obs, p=theta, observed=suc)

        # Use NUTS sampler (No-U-Turn Sampler) — PyMC default
        idata = pm.sample(
            draws=n_samples,
            tune=500,
            chains=2,
            progressbar=False,
            random_seed=random_seed,
            target_accept=0.9,
        )

    posterior_theta = idata.posterior["theta"].values.flatten()
    mean_conf = float(np.mean(posterior_theta))
    try:
        hdi = pm.hdi(posterior_theta, hdi_prob=0.94)
    except TypeError:
        hdi = pm.hdi(posterior_theta, prob=0.94)
    hdi_low, hdi_high = float(hdi[0]), float(hdi[1])

    return mean_conf, hdi_low, hdi_high


# ---------------------------------------------------------------------------
# Fix strategies
# ---------------------------------------------------------------------------

def _fix_missing(df: pd.DataFrame, row: int, col: str) -> tuple[Any, str]:
    series = df[col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        fill = round(float(series.median()), 4)
        explanation = (
            f"Missing value imputed with column median ({fill}). "
            "Median is robust to outliers and preferred over mean for skewed distributions."
        )
    else:
        fill = series.mode().iloc[0] if not series.mode().empty else "UNKNOWN"
        explanation = (
            f"Missing string value imputed with column mode ('{fill}'). "
            "Mode represents the most frequent observed category."
        )
    return fill, explanation


def _fix_outlier(df: pd.DataFrame, row: int, col: str, original: Any) -> tuple[Any, str]:
    series = df[col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clamped = float(np.clip(float(original), lower, upper))
        explanation = (
            f"Outlier value ({original}) clamped to IQR fence "
            f"[{lower:.4f}, {upper:.4f}] → {clamped:.4f}. "
            "Winsorization preserves row while bringing value within expected range."
        )
        return clamped, explanation
    return original, "Outlier in non-numeric column — value left unchanged."


def _fix_type_mismatch(df: pd.DataFrame, row: int, col: str, original: Any) -> tuple[Any, str]:
    series = df[col].dropna()
    if pd.api.types.is_numeric_dtype(series):
        try:
            fixed = float(str(original).replace(",", "").strip())
            explanation = (
                f"Type mismatch: '{original}' coerced to numeric {fixed}. "
                "String representation of a number converted to float."
            )
            return fixed, explanation
        except ValueError:
            return np.nan, f"Type mismatch: could not coerce '{original}' to numeric — set to NaN."
    return str(original), f"Type coerced to string: '{original}'."


def _fix_duplicate(df: pd.DataFrame, row: int, col: str) -> tuple[Any, str]:
    return None, (
        "Duplicate row flagged for removal. "
        "The entire row will be dropped during post-processing. "
        "Keeping only the first occurrence preserves data integrity."
    )


def _fix_format_error(df: pd.DataFrame, row: int, col: str, original: Any) -> tuple[Any, str]:
    val = str(original).strip()
    # Attempt basic date normalisation as an example
    for fmt in ("%d/%m/%Y", "%m-%d-%Y", "%Y%m%d", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(val, fmt)
            fixed = parsed.strftime("%Y-%m-%d")
            return fixed, (
                f"Date format error: '{original}' normalised to ISO-8601 '{fixed}'. "
                "Standard date format prevents downstream parsing failures."
            )
        except ValueError:
            continue
    # Fallback: strip whitespace / special chars
    fixed = val.strip().replace("\n", " ").replace("\t", " ")
    return fixed, f"Format error: whitespace/control characters stripped from '{original}' → '{fixed}'."


def _fix_range_violation(df: pd.DataFrame, row: int, col: str, original: Any) -> tuple[Any, str]:
    return _fix_outlier(df, row, col, original)


FIX_DISPATCH = {
    "missing":         _fix_missing,
    "outlier":         _fix_outlier,
    "type_mismatch":   _fix_type_mismatch,
    "duplicate":       _fix_duplicate,
    "format_error":    _fix_format_error,
    "range_violation": _fix_range_violation,
}


def apply_fix(
    anomaly: Anomaly,
    df: pd.DataFrame,
) -> tuple[Any, str]:
    """Dispatch to the correct fix strategy and return (fixed_value, explanation)."""
    fn = FIX_DISPATCH.get(anomaly.anomaly_type)
    if fn is None:
        return anomaly.original_value, (
            f"No fix strategy defined for anomaly type '{anomaly.anomaly_type}'. "
            "Original value retained."
        )
    if anomaly.anomaly_type in ("missing", "duplicate"):
        return fn(df, anomaly.row, anomaly.column)
    return fn(df, anomaly.row, anomaly.column, anomaly.original_value)


# ---------------------------------------------------------------------------
# Core remediation loop
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60   # Only apply a fix if Bayesian posterior mean ≥ 60 %


def remediate(
    frames: dict[str, pd.DataFrame],
    anomalies: list[Anomaly],
    priorities: dict[str, PriorityItem],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    n_samples: int = 1000,
) -> tuple[dict[str, pd.DataFrame], list[RemediationResult]]:
    """
    Main remediation loop.

    Iterates anomalies in priority order, computes Bayesian confidence for each
    candidate fix, applies the fix if confidence ≥ threshold, and records results.
    """
    # Sort by priority rank (rank 1 = most important), unknown priorities last
    def sort_key(a: Anomaly) -> int:
        p = priorities.get(a.anomaly_id)
        return p.rank if p else 999_999

    sorted_anomalies = sorted(anomalies, key=sort_key)

    results: list[RemediationResult] = []
    rows_to_drop: dict[str, list[int]] = {f: [] for f in frames}

    for anomaly in sorted_anomalies:
        fname = anomaly.file
        if fname not in frames:
            log.warning("File '%s' not loaded — skipping anomaly %s", fname, anomaly.anomaly_id)
            continue

        df = frames[fname]
        if anomaly.row >= len(df):
            log.warning("Row %d out of bounds in '%s' — skipping", anomaly.row, fname)
            continue

        log.info(
            "Processing anomaly %s  [%s | row %d | col '%s' | type '%s']",
            anomaly.anomaly_id, fname, anomaly.row, anomaly.column, anomaly.anomaly_type,
        )

        # --- Bayesian confidence estimation ---
        mean_conf, hdi_low, hdi_high = bayesian_confidence(anomaly, df, n_samples=n_samples)
        log.info(
            "  Confidence: %.1f%%  (94%% HDI: %.1f%% – %.1f%%)",
            mean_conf * 100, hdi_low * 100, hdi_high * 100,
        )

        # --- Candidate fix ---
        fixed_value, explanation = apply_fix(anomaly, df)

        # --- Decision ---
        fix_applied = mean_conf >= confidence_threshold

        if fix_applied:
            if anomaly.anomaly_type == "duplicate":
                rows_to_drop[fname].append(anomaly.row)
                log.info("  ✓ Row %d flagged for duplicate removal", anomaly.row)
            elif anomaly.column in df.columns:
                try:
                    df.at[anomaly.row, anomaly.column] = fixed_value
                except (TypeError, ValueError):
                    # Column dtype is too restrictive — cast to object, write,
                    # then try to recover a clean numeric dtype afterwards.
                    df[anomaly.column] = df[anomaly.column].astype(object)
                    df.at[anomaly.row, anomaly.column] = fixed_value
                    try:
                        df[anomaly.column] = pd.to_numeric(df[anomaly.column], errors="ignore")
                    except Exception:
                        pass
                log.info("  ✓ Fix applied: %r → %r", anomaly.original_value, fixed_value)
        else:
            log.info(
                "  ✗ Fix NOT applied (confidence %.1f%% < threshold %.1f%%)",
                mean_conf * 100, confidence_threshold * 100,
            )
            explanation = (
                f"[FIX WITHHELD — confidence {mean_conf*100:.1f}% below "
                f"threshold {confidence_threshold*100:.1f}%] " + explanation
            )

        p = priorities.get(anomaly.anomaly_id)
        results.append(RemediationResult(
            anomaly_id=anomaly.anomaly_id,
            file=fname,
            row=anomaly.row,
            column=anomaly.column,
            anomaly_type=anomaly.anomaly_type,
            original_value=anomaly.original_value,
            fixed_value=fixed_value if fix_applied else anomaly.original_value,
            confidence_pct=round(mean_conf * 100, 2),
            hdi_low_pct=round(hdi_low * 100, 2),
            hdi_high_pct=round(hdi_high * 100, 2),
            fix_applied=fix_applied,
            explanation=explanation,
            priority_rank=p.rank if p else None,
        ))

    # Drop duplicate rows (in reverse order to preserve indices)
    for fname, drop_rows in rows_to_drop.items():
        if drop_rows:
            frames[fname] = frames[fname].drop(index=sorted(set(drop_rows), reverse=True)).reset_index(drop=True)
            log.info("Dropped %d duplicate rows from '%s'", len(drop_rows), fname)

    return frames, results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_remediated_csvs(frames: dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for fname, df in frames.items():
        stem = Path(fname).stem
        out_path = Path(output_dir) / f"{stem}_remediated.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved remediated CSV → %s", out_path)


def save_report(results: list[RemediationResult], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report_path = Path(output_dir) / "remediation_report.json"

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_anomalies_processed": len(results),
        "fixes_applied": sum(1 for r in results if r.fix_applied),
        "fixes_withheld": sum(1 for r in results if not r.fix_applied),
        "confidence_threshold_pct": CONFIDENCE_THRESHOLD * 100,
        "results": [asdict(r) for r in results],
    }
    with open(report_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("Remediation report → %s", report_path)
    return str(report_path)


def print_summary(results: list[RemediationResult]) -> None:
    print("\n" + "=" * 72)
    print("  REMEDIATION SUMMARY")
    print("=" * 72)
    print(f"  Total anomalies processed : {len(results)}")
    print(f"  Fixes applied             : {sum(1 for r in results if r.fix_applied)}")
    print(f"  Fixes withheld            : {sum(1 for r in results if not r.fix_applied)}")
    print(f"  Confidence threshold      : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print("-" * 72)
    header = f"{'ID':<12} {'File':<20} {'Row':>4} {'Col':<15} {'Type':<16} {'Conf%':>6} {'Applied'}"
    print(header)
    print("-" * 72)
    for r in sorted(results, key=lambda x: x.priority_rank or 999):
        applied_str = "✓ YES" if r.fix_applied else "✗ NO "
        print(
            f"{r.anomaly_id:<12} {r.file:<20} {r.row:>4} {r.column:<15} "
            f"{r.anomaly_type:<16} {r.confidence_pct:>5.1f}%  {applied_str}"
        )
    print("=" * 72)
    print("\nDETAILED EXPLANATIONS")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x.priority_rank or 999):
        status = "FIXED" if r.fix_applied else "WITHHELD"
        print(f"\n[{r.anomaly_id}] {r.file} | row {r.row} | col '{r.column}' | {r.anomaly_type.upper()}")
        print(f"  Status     : {status}")
        print(f"  Original   : {r.original_value!r}")
        print(f"  Fixed to   : {r.fixed_value!r}")
        print(f"  Confidence : {r.confidence_pct:.2f}%  (94% HDI: {r.hdi_low_pct:.1f}% – {r.hdi_high_pct:.1f}%)")
        print(f"  Explanation: {r.explanation}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agent 3 — Bayesian CSV Data Remediation using PyMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Minimal usage
python csv_remediation_agent.py \\
    --csvs data/sales.csv data/inventory.csv \\
    --anomalies agent1_output.json \\
    --priorities agent2_output.json

# Custom output directory and lower confidence threshold
python csv_remediation_agent.py \\
    --csvs data/*.csv \\
    --anomalies agent1_output.json \\
    --priorities agent2_output.json \\
    --output-dir results/ \\
    --confidence-threshold 0.70 \\
    --n-samples 2000

Input JSON Schemas
------------------
agent1_output.json  (list of anomaly objects)
[
  {
    "anomaly_id": "A001",
    "file": "sales.csv",
    "row": 14,
    "column": "revenue",
    "anomaly_type": "outlier",       // missing|outlier|type_mismatch|duplicate|format_error|range_violation
    "original_value": 9999999,
    "description": "Revenue 100x higher than column median",
    "severity": 0.85                 // 0–1
  },
  ...
]

agent2_output.json  (list of priority objects)
[
  {
    "anomaly_id": "A001",
    "rank": 1,                       // 1 = highest priority
    "priority_score": 0.95
  },
  ...
]
""",
    )
    p.add_argument("--csvs", nargs="+", required=True, help="Paths to raw CSV files")
    p.add_argument("--anomalies", required=True, help="Path to Agent 1 anomaly JSON")
    p.add_argument("--priorities", required=True, help="Path to Agent 2 priority JSON")
    p.add_argument("--output-dir", default="remediated_output", help="Output directory (default: remediated_output/)")
    p.add_argument(
        "--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Minimum Bayesian confidence to apply a fix (default: {CONFIDENCE_THRESHOLD})",
    )
    p.add_argument(
        "--n-samples", type=int, default=1000,
        help="MCMC draws per anomaly (default: 1000; reduce for speed, increase for accuracy)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence_threshold

    log.info("=== CSV Remediation Agent (Agent 3) — PyMC Bayesian ===")
    log.info("Confidence threshold : %.0f%%", args.confidence_threshold * 100)
    log.info("MCMC draws per anomaly: %d", args.n_samples)

    # Load inputs
    frames = load_csvs(args.csvs)
    raw_anomalies = load_json(args.anomalies, "anomaly")
    raw_priorities = load_json(args.priorities, "priority")

    anomalies = parse_anomalies(raw_anomalies)
    priorities = parse_priorities(raw_priorities)

    # Remediate
    remediated_frames, results = remediate(
        frames, anomalies, priorities,
        confidence_threshold=args.confidence_threshold,
        n_samples=args.n_samples,
    )

    # Persist outputs
    save_remediated_csvs(remediated_frames, args.output_dir)
    report_path = save_report(results, args.output_dir)

    # Human-readable summary
    print_summary(results)
    print(f"Full JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
