"""Agent 1 — Find It: detect all 6 data-quality issue classes.

Uses schema_map.json (from Agent 0 / schema_detect) so this works on ANY dataset —
column names are resolved at runtime, not hardcoded.

Classes (from dataset taxonomy):
  1. exact_duplicate          — identical records, different record_id
  2. near_duplicate_variant   — same part after case/whitespace normalisation
  3. unit_format_drift        — separator change clustering in time (firmware artefact)
  4. orphaned_reference       — entity reference absent from a companion lookup file
  5. decimal_shift_weight     — weight ≈ 10× or 100× part-number median (per-part z-score)
  6. impossible_value         — negative qty, ship-before-production, status/date contradiction
"""

import json
import re
from datetime import datetime
from pathlib import Path

import cognee
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _normalize_pn(pn: str) -> str:
    return re.sub(r"[\s_\-]+", "-", str(pn).strip().upper()).strip("-")


def _find_companion_lookup(data_dir: Path, customer_col: str) -> set[str]:
    """Find a companion CSV that lists valid values for the customer/entity column."""
    # Try any CSV that isn't the primary data file and has a matching column name fragment
    for f in sorted(data_dir.glob("*.csv")):
        try:
            df_lookup = pd.read_csv(f)
            # Check if any column looks like a customer_id column
            for col in df_lookup.columns:
                if re.search(r"customer.?id|cust.?id|client.?id|account.?id", col, re.I):
                    return set(df_lookup[col].str.strip())
                # Or: column that overlaps significantly with values in customer_col
            # Fallback: if filename suggests it's a reference table
            if re.search(r"customer|client|account|entity", f.stem, re.I):
                first_col = df_lookup.columns[0]
                return set(df_lookup[first_col].astype(str).str.strip())
        except Exception:
            continue
    return set()


async def find_issues() -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load schema map (set by Agent 0; fallback to column-name guessing) ────
    schema_path = OUTPUT_DIR / "schema_map.json"
    schema: dict = {}
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)

    def col(concept: str, fallback: str | None = None) -> str | None:
        return schema.get(concept) or fallback

    # ── Discover primary CSV ──────────────────────────────────────────────────
    csv_files = [f for f in DATA_DIR.glob("*.csv")
                 if not re.search(r"customer|client|entity|lookup", f.name, re.I)]
    if not csv_files:
        csv_files = list(DATA_DIR.glob("*.csv"))
    primary_csv = csv_files[0]
    df = pd.read_csv(primary_csv)
    total = len(df)

    # ── Column bindings from schema ───────────────────────────────────────────
    C_ID       = col("record_id")
    C_PART     = col("part_number")
    C_CUSTOMER = col("customer_id")
    C_PROD     = col("production_date")
    C_SHIP     = col("ship_date")
    C_QTY      = col("quantity")
    C_WEIGHT   = col("weight")
    C_STATUS   = col("status")

    # ── Parse dates ───────────────────────────────────────────────────────────
    if C_PROD and C_PROD in df.columns:
        df["_prod"] = pd.to_datetime(df[C_PROD], errors="coerce")
    else:
        df["_prod"] = pd.NaT

    if C_SHIP and C_SHIP in df.columns:
        df["_ship"] = pd.to_datetime(df[C_SHIP], errors="coerce")
    else:
        df["_ship"] = pd.NaT

    audit_cutoff = datetime.now()

    if C_PART and C_PART in df.columns:
        df["_pn_norm"] = df[C_PART].apply(_normalize_pn)
    else:
        df["_pn_norm"] = ""

    issues: list[dict] = []

    def _ids(mask) -> list:
        if C_ID and C_ID in df.columns:
            return df.loc[mask, C_ID].tolist()
        return df.index[mask].tolist()

    # ── Class 1: Exact duplicates ─────────────────────────────────────────────
    id_cols = [c for c in df.columns if not c.startswith("_") and c != C_ID]
    extra_dupes = df[df.duplicated(subset=id_cols, keep="first")]
    if len(extra_dupes):
        issues.append({
            "type": "duplicate",
            "subtype": "exact_duplicate",
            "count": len(extra_dupes),
            "record_ids": _ids(extra_dupes.index),
            "description": (
                f"{len(extra_dupes)} records are exact copies of another record "
                f"(all columns identical except {C_ID}) — double-entry during migration."
            ),
        })

    # ── Class 2: Near-duplicate variants (case / whitespace / separator) ──────
    if C_PART and C_PART in df.columns:
        near_dup_cols = ["_pn_norm"]
        if C_CUSTOMER and C_CUSTOMER in df.columns:
            near_dup_cols.append(C_CUSTOMER)
        if C_PROD and C_PROD in df.columns:
            near_dup_cols.append(C_PROD)
        if C_QTY and C_QTY in df.columns:
            near_dup_cols.append(C_QTY)

        near_dups = df[
            df.duplicated(subset=near_dup_cols, keep=False) &
            ~df.index.isin(extra_dupes.index)
        ]
        near_dups = near_dups[
            near_dups[C_PART].apply(_normalize_pn) != near_dups[C_PART].str.strip().str.upper()
        ]
        extra_near = near_dups[near_dups.duplicated(subset=near_dup_cols, keep="first")]
        if len(extra_near):
            issues.append({
                "type": "duplicate",
                "subtype": "near_duplicate_variant",
                "count": len(extra_near),
                "record_ids": _ids(extra_near.index),
                "description": (
                    f"{len(extra_near)} near-duplicate records: same normalised part, customer, "
                    f"dates, and quantity — differ only in casing/whitespace/separator."
                ),
            })

    # ── Class 3: Unit-format drift (temporal separator clustering) ────────────
    if C_PART and C_PART in df.columns:
        pn_has_both = {
            k for k, v in df.groupby("_pn_norm")[C_PART].apply(
                lambda s: set(str(p).strip().count("_") > 0 for p in s)
            ).items() if True in v and False in v
        }
        if pn_has_both:
            us_mask = df[C_PART].str.strip().str.contains("_", regex=False) & df["_pn_norm"].isin(pn_has_both)
            hy_mask = ~df[C_PART].str.strip().str.contains("_", regex=False) & df["_pn_norm"].isin(pn_has_both)
            underscore_df = df[us_mask]
            hyphen_df = df[hy_mask]

            drift_detected = False
            drift_note = ""
            if len(underscore_df) >= 5 and len(hyphen_df) >= 5 and df["_prod"].notna().any():
                u_med = underscore_df["_prod"].dropna().median()
                h_med = hyphen_df["_prod"].dropna().median()
                if pd.notna(u_med) and pd.notna(h_med):
                    gap = abs((u_med - h_med).days)
                    if gap > 30:
                        drift_detected = True
                        drift_note = (
                            f"Temporal gap {gap}d between format variants suggests "
                            f"a firmware/ETL configuration change."
                        )

            subtype = "unit_format_drift" if drift_detected else "separator_mismatch"
            if len(underscore_df):
                issues.append({
                    "type": "naming_conflict",
                    "subtype": subtype,
                    "count": len(underscore_df),
                    "record_ids": _ids(underscore_df.index),
                    "description": (
                        f"{len(underscore_df)} records use underscore separator for part numbers "
                        f"that elsewhere use hyphens — {len(pn_has_both)} affected part numbers. "
                        + (drift_note if drift_detected else "Likely acquired-plant naming convention.")
                    ),
                })

    # ── Class 2 extension: residual malformed part numbers ───────────────────
    if C_PART and C_PART in df.columns:
        handled_ids = set(
            rid for iss in issues for rid in iss.get("record_ids", [])
        )
        handled_idx = df[df[C_ID].isin(handled_ids)].index if C_ID else pd.Index([])
        malformed = df[
            (df[C_PART] != df[C_PART].str.strip().str.upper()) &
            ~df.index.isin(handled_idx)
        ]
        if len(malformed):
            issues.append({
                "type": "naming_conflict",
                "subtype": "malformed_part_number",
                "count": len(malformed),
                "record_ids": _ids(malformed.index),
                "description": (
                    f"{len(malformed)} records have {C_PART} values with whitespace or incorrect casing."
                ),
            })

    # ── Class 4: Orphaned entity references ──────────────────────────────────
    if C_CUSTOMER and C_CUSTOMER in df.columns:
        valid_ids = _find_companion_lookup(DATA_DIR, C_CUSTOMER)
        if valid_ids:
            orphaned = df[~df[C_CUSTOMER].str.strip().isin(valid_ids)]
            if len(orphaned):
                issues.append({
                    "type": "orphaned_reference",
                    "subtype": "unknown_customer_id",
                    "count": len(orphaned),
                    "record_ids": _ids(orphaned.index),
                    "unknown_ids": orphaned[C_CUSTOMER].str.strip().unique().tolist(),
                    "description": (
                        f"{len(orphaned)} records reference {C_CUSTOMER} values absent from "
                        f"the companion lookup file "
                        f"({len(orphaned[C_CUSTOMER].unique())} distinct IDs)."
                    ),
                })

    # ── Class 5: Decimal-shift weights (per-part z-score + ratio) ────────────
    if C_WEIGHT and C_WEIGHT in df.columns and C_PART and C_PART in df.columns:
        valid_w = df[df[C_WEIGHT] > 0].copy()
        if len(valid_w) > 10:
            part_stats = valid_w.groupby("_pn_norm")[C_WEIGHT].agg(["median", "std", "count"])
            part_stats.columns = ["w_med", "w_std", "n"]
            df = df.join(part_stats, on="_pn_norm")

            df["_zscore"] = df.apply(
                lambda r: abs(r[C_WEIGHT] - r["w_med"]) / r["w_std"]
                if r.get("n", 0) >= 4 and r.get("w_std", 0) > 0 else 0.0,
                axis=1,
            )
            df["_ratio"] = df.apply(
                lambda r: r[C_WEIGHT] / r["w_med"] if r.get("w_med", 0) > 0 else 1.0, axis=1
            )

            d10 = df[(df["_zscore"] > 3.5) & df["_ratio"].between(7, 13) & (df.get("n", 0) >= 4)]
            d100 = df[(df["_zscore"] > 3.5) & df["_ratio"].between(70, 130) & (df.get("n", 0) >= 4)]
            d10_ids = set(_ids(d10.index))
            d100_ids = set(_ids(d100.index))
            other_inf = df[
                (df["_zscore"] > 3.5) &
                ~df.index.isin(d10.index) &
                ~df.index.isin(d100.index) &
                (df.get("n", pd.Series(0, index=df.index)) >= 4)
            ]

            for subtype, subset, desc in [
                ("decimal_shift_10x", d10,
                 f"{len(d10)} records: {C_WEIGHT} ≈ 10× part median — decimal-point input error likely."),
                ("decimal_shift_100x", d100,
                 f"{len(d100)} records: {C_WEIGHT} ≈ 100× part median — wrong unit scale likely."),
                ("weight_inflation", other_inf,
                 f"{len(other_inf)} records: {C_WEIGHT} >3.5σ above part median — unit conversion error likely."),
            ]:
                if len(subset):
                    issues.append({
                        "type": "impossible_value",
                        "subtype": subtype,
                        "count": len(subset),
                        "record_ids": _ids(subset.index),
                        "description": desc,
                    })

    # ── Class 6a: Ship before production ─────────────────────────────────────
    if df["_ship"].notna().any() and df["_prod"].notna().any():
        date_inv = df[df["_ship"].notna() & df["_prod"].notna() & (df["_ship"] < df["_prod"])]
        if len(date_inv):
            issues.append({
                "type": "impossible_value",
                "subtype": "ship_before_production",
                "count": len(date_inv),
                "record_ids": _ids(date_inv.index),
                "description": f"{len(date_inv)} records: {C_SHIP} < {C_PROD} — field transposition likely.",
            })

    # ── Class 6b: Negative / zero quantity ───────────────────────────────────
    if C_QTY and C_QTY in df.columns:
        neg = df[df[C_QTY] <= 0]
        if len(neg):
            issues.append({
                "type": "impossible_value",
                "subtype": "non_positive_quantity",
                "count": len(neg),
                "record_ids": _ids(neg.index),
                "description": f"{len(neg)} records with {C_QTY} ≤ 0 — non-positive stock is impossible.",
            })

    # ── Class 6c: Status/date contradiction ──────────────────────────────────
    if C_STATUS and C_STATUS in df.columns and df["_ship"].notna().any():
        completed_terms = {"completed", "shipped", "delivered", "closed", "done"}
        mask_status = df[C_STATUS].str.lower().str.strip().isin(completed_terms)
        conflict = df[mask_status & df["_ship"].notna() & (df["_ship"] > pd.Timestamp(audit_cutoff))]
        if len(conflict):
            issues.append({
                "type": "logical_conflict",
                "subtype": "status_future_ship_date",
                "count": len(conflict),
                "record_ids": _ids(conflict.index),
                "description": (
                    f"{len(conflict)} records marked completed/shipped with a future {C_SHIP}."
                ),
            })

    # ── Class 6d: Future production date ─────────────────────────────────────
    fut_prod = df[df["_prod"].notna() & (df["_prod"] > pd.Timestamp(audit_cutoff))]
    if len(fut_prod):
        issues.append({
            "type": "logical_conflict",
            "subtype": "future_production_date",
            "count": len(fut_prod),
            "record_ids": _ids(fut_prod.index),
            "description": f"{len(fut_prod)} records with {C_PROD} in the future.",
        })

    # ── Cognee: write findings ────────────────────────────────────────────────
    total_affected = sum(i["count"] for i in issues)
    await cognee.add(
        f"Agent 1 (Find It): {total} records, {len(issues)} issue classes, "
        f"{total_affected} affected. Classes: "
        + ", ".join(f"{i['type']}/{i['subtype']}" for i in issues),
        dataset_name="data_rescue",
    )
    for issue in issues:
        await cognee.add(
            f"ISSUE [{issue['type']}/{issue['subtype']}] n={issue['count']}: {issue['description']}",
            dataset_name="data_rescue",
        )

    await cognee.cognify()  # vectorise staged data so downstream agents can search

    with open(OUTPUT_DIR / "findings.json", "w") as f:
        json.dump(issues, f, indent=2, default=str)
    with open(OUTPUT_DIR / "dataset_meta.json", "w") as f:
        json.dump({"total_records": total, "primary_csv": str(primary_csv)}, f)

    print(f"[Agent 1 — Find It] {total} records → {len(issues)} issue classes, {total_affected} affected")
    for iss in issues:
        print(f"  • {iss['type']}/{iss['subtype']}: {iss['count']}")

    return issues
