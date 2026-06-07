"""Agent 1 — Find It: scan the dataset and classify every data-quality issue."""

import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cognee
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
AUDIT_CUTOFF = datetime(2026, 6, 7)  # today


async def find_issues() -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "track01_data_rescue.csv")
    customers = pd.read_csv(DATA_DIR / "track01_customers.csv")
    valid_customer_ids = set(customers["customer_id"].str.strip())

    total = len(df)
    issues: list[dict] = []

    # ── 1. Semantic duplicates (same data, different record_id) ──────────────
    cols_no_id = [c for c in df.columns if c != "record_id"]
    dup_mask = df.duplicated(subset=cols_no_id, keep=False)
    dups_df = df[dup_mask]
    # keep first occurrence; extras are the issues
    extra_dupes = df[df.duplicated(subset=cols_no_id, keep="first")]
    if len(extra_dupes):
        issues.append({
            "type": "duplicate",
            "subtype": "semantic_duplicate",
            "count": len(extra_dupes),
            "record_ids": extra_dupes["record_id"].tolist(),
            "description": (
                f"{len(extra_dupes)} records are exact copies of another record "
                f"(same data, different record_id) — likely double-entry during migration"
            ),
        })

    # ── 2. Part-number separator conflict: BOLT_119 vs BOLT-119 ─────────────
    def normalize_pn(pn: str) -> str:
        return re.sub(r"[_\-]", "-", pn.strip().upper())

    pn_normalized: dict[str, set] = defaultdict(set)
    for pn in df["part_number"]:
        pn_normalized[normalize_pn(pn)].add(pn.strip())

    sep_conflict_bases = {k: v for k, v in pn_normalized.items() if len(v) > 1}
    underscore_records = df[df["part_number"].str.strip().str.contains("_", regex=False) &
                            df["part_number"].str.strip().apply(
                                lambda p: normalize_pn(p) in sep_conflict_bases)]
    if len(underscore_records):
        issues.append({
            "type": "naming_conflict",
            "subtype": "separator_mismatch",
            "count": len(underscore_records),
            "record_ids": underscore_records["record_id"].tolist(),
            "description": (
                f"{len(underscore_records)} records use underscore separator (BOLT_119) "
                f"for part numbers that elsewhere appear with hyphens (BOLT-119). "
                f"Affects {len(sep_conflict_bases)} distinct part numbers — likely acquired-plant convention."
            ),
        })

    # ── 3. Malformed part numbers (whitespace / lowercase) ───────────────────
    malformed = df[df["part_number"] != df["part_number"].str.strip().str.upper()]
    # Exclude records already caught by separator conflict
    sep_ids = set(underscore_records["record_id"]) if len(underscore_records) else set()
    malformed_extra = malformed[~malformed["record_id"].isin(sep_ids)]
    if len(malformed_extra):
        issues.append({
            "type": "naming_conflict",
            "subtype": "malformed_part_number",
            "count": len(malformed_extra),
            "record_ids": malformed_extra["record_id"].tolist(),
            "description": (
                f"{len(malformed_extra)} records have part numbers with leading/trailing "
                f"whitespace or incorrect casing (e.g. 'shft-121 ') — likely manual data entry errors."
            ),
        })

    # ── 4. Orphaned customer references (CX- IDs not in master list) ────────
    orphaned = df[~df["customer_id"].isin(valid_customer_ids)]
    if len(orphaned):
        issues.append({
            "type": "orphaned_reference",
            "subtype": "unknown_customer_id",
            "count": len(orphaned),
            "record_ids": orphaned["record_id"].tolist(),
            "unknown_ids": orphaned["customer_id"].unique().tolist(),
            "description": (
                f"{len(orphaned)} records reference customer IDs not in the master customer file "
                f"({len(orphaned['customer_id'].unique())} distinct IDs, all with 'CX-' prefix). "
                f"May indicate phantom customers or a migration from the acquired plants' CRM."
            ),
        })

    # ── 5. Weight inflation: >5× median for the same part number ────────────
    df["_pn_norm"] = df["part_number"].apply(normalize_pn)
    part_medians = df.groupby("_pn_norm")["weight_kg"].median()
    df["_weight_ratio"] = df.apply(
        lambda r: r["weight_kg"] / part_medians[r["_pn_norm"]]
        if part_medians[r["_pn_norm"]] > 0 else 1.0, axis=1
    )
    inflated = df[(df["_weight_ratio"] > 5) & (df.groupby("_pn_norm")["weight_kg"].transform("count") >= 4)]
    if len(inflated):
        issues.append({
            "type": "impossible_value",
            "subtype": "weight_inflation",
            "count": len(inflated),
            "record_ids": inflated["record_id"].tolist(),
            "description": (
                f"{len(inflated)} records have weight_kg more than 5× the median for that part number. "
                f"Consistent with a unit-conversion error (e.g., lbs entered as kg) on specific records."
            ),
        })

    # ── 6. Impossible dates: ship_date before production_date ───────────────
    df["_prod"] = pd.to_datetime(df["production_date"])
    df["_ship"] = pd.to_datetime(df["ship_date"])
    date_inv = df[df["_ship"] < df["_prod"]]
    if len(date_inv):
        issues.append({
            "type": "impossible_value",
            "subtype": "ship_before_production",
            "count": len(date_inv),
            "record_ids": date_inv["record_id"].tolist(),
            "description": (
                f"{len(date_inv)} records have a ship_date that precedes the production_date — "
                f"physically impossible; likely a transposition error during data entry."
            ),
        })

    # ── 7. Negative or zero quantity ─────────────────────────────────────────
    neg_qty = df[df["quantity"] <= 0]
    if len(neg_qty):
        issues.append({
            "type": "impossible_value",
            "subtype": "non_positive_quantity",
            "count": len(neg_qty),
            "record_ids": neg_qty["record_id"].tolist(),
            "description": (
                f"{len(neg_qty)} records have a quantity ≤ 0 "
                f"(values: {sorted(neg_qty['quantity'].tolist())}). "
                f"Non-positive quantities are physically impossible for warehouse stock."
            ),
        })

    # ── 8. Status / date logical conflicts ───────────────────────────────────
    status_conflict = df[
        df["status"].isin(["completed", "shipped"]) & (df["_ship"] > pd.Timestamp(AUDIT_CUTOFF))
    ]
    if len(status_conflict):
        issues.append({
            "type": "logical_conflict",
            "subtype": "status_future_ship_date",
            "count": len(status_conflict),
            "record_ids": status_conflict["record_id"].tolist(),
            "description": (
                f"{len(status_conflict)} records are marked 'completed' or 'shipped' "
                f"but have a ship_date in the future — status and date contradict each other."
            ),
        })

    # ── 9. Future production dates ────────────────────────────────────────────
    future_prod = df[df["_prod"] > pd.Timestamp(AUDIT_CUTOFF)]
    if len(future_prod):
        issues.append({
            "type": "logical_conflict",
            "subtype": "future_production_date",
            "count": len(future_prod),
            "record_ids": future_prod["record_id"].tolist(),
            "description": (
                f"{len(future_prod)} records have a production_date in the future. "
                f"These cannot appear in an audit of current stock."
            ),
        })

    # ── Store in Cognee (vector-only: add without cognify — zero LLM calls) ──
    total_affected = sum(i["count"] for i in issues)
    summary = (
        f"Agent 1 (Find It) completed scan of Harven Manufacturing dataset "
        f"({total} records, 3 plants). "
        f"Identified {len(issues)} issue categories affecting {total_affected} records. "
        f"Issue types: {', '.join(i['type'] + '/' + i['subtype'] for i in issues)}."
    )
    await cognee.add(summary, dataset_name="data_rescue")
    for issue in issues:
        await cognee.add(
            f"ISSUE [{issue['type']}/{issue['subtype']}] count={issue['count']}: {issue['description']}",
            dataset_name="data_rescue",
        )

    # Local file for inter-agent reliability
    with open(OUTPUT_DIR / "findings.json", "w") as f:
        json.dump(issues, f, indent=2, default=str)

    print(f"[Agent 1 — Find It] {len(issues)} issue types, {total_affected} total affected records")
    for iss in issues:
        print(f"  • {iss['type']}/{iss['subtype']}: {iss['count']} records")

    return issues
