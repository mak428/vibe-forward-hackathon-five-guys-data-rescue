"""Agent 3 — Act On It: fix, flag, or escalate every issue with a logged justification."""

import json
import re
from pathlib import Path

import cognee
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def normalize_pn(pn: str) -> str:
    return re.sub(r"[_]", "-", pn.strip().upper())


async def act_on_issues(ranked: list[dict] | None = None) -> dict:
    if ranked is None:
        with open(OUTPUT_DIR / "rankings.json") as f:
            ranked = json.load(f)

    df = pd.read_csv(DATA_DIR / "track01_data_rescue.csv")
    customers = pd.read_csv(DATA_DIR / "track01_customers.csv")
    valid_ids = set(customers["customer_id"])

    # Load Geodo validation results if the team has filled them in
    geodo_path = OUTPUT_DIR / "geodo_results.json"
    geodo_data: dict = {}
    if geodo_path.exists():
        with open(geodo_path) as f:
            geodo_data = json.load(f)

    df["production_date"] = pd.to_datetime(df["production_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])

    audit_log: list[dict] = []
    flags: dict[str, list[str]] = {}   # record_id → list of issue tags
    indices_to_drop: set[int] = set()

    def flag(record_ids: list[str], tag: str):
        for rid in record_ids:
            flags.setdefault(rid, []).append(tag)

    for issue in ranked:
        itype = issue["type"]
        isubtype = issue["subtype"]
        record_ids = issue.get("record_ids", [])
        count = issue["count"]

        # ── Semantic duplicates → AUTO-FIX ───────────────────────────────────
        if isubtype == "semantic_duplicate":
            cols_no_id = [c for c in df.columns if c != "record_id"]
            extra_idx = df[df.duplicated(subset=cols_no_id, keep="first")].index.tolist()
            indices_to_drop.update(extra_idx)
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "AUTO_FIXED",
                "records_affected": count,
                "justification": (
                    "Exact duplicate records removed; canonical first occurrence retained. "
                    "Duplicates arose from double-entry during plant migration."
                ),
            })

        # ── Separator mismatch → AUTO-FIX ────────────────────────────────────
        elif isubtype == "separator_mismatch":
            mask = df["part_number"].str.strip().str.contains("_", regex=False)
            df.loc[mask, "part_number"] = df.loc[mask, "part_number"].apply(
                lambda p: re.sub(r"[_]", "-", p.strip().upper())
            )
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "AUTO_FIXED",
                "records_affected": count,
                "justification": (
                    f"Normalised {count} part numbers from underscore (BOLT_119) to "
                    f"hyphen (BOLT-119) convention. Acquired plants B and C used underscore; "
                    f"Plant A (original) uses hyphen. Standardised to Plant A canonical format."
                ),
            })

        # ── Malformed part numbers (whitespace / lowercase) → AUTO-FIX ───────
        elif isubtype == "malformed_part_number":
            df["part_number"] = df["part_number"].str.strip().str.upper()
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "AUTO_FIXED",
                "records_affected": count,
                "justification": (
                    f"Stripped whitespace and uppercased {count} part numbers. "
                    f"Consistent with manual data-entry errors or inconsistent ETL pipelines."
                ),
            })

        # ── Orphaned customer IDs (CX-) → ESCALATE with Geodo context ────────
        elif isubtype == "unknown_customer_id":
            unknown_ids = issue.get("unknown_ids", [])
            geodo_verified = [cid for cid in unknown_ids if geodo_data.get(cid, {}).get("verified")]
            unverified = [cid for cid in unknown_ids if cid not in geodo_verified]

            flag(record_ids, "ORPHANED_CUSTOMER_ID")
            geodo_note = (
                f"Geodo validation: {len(geodo_verified)} IDs confirmed as real entities; "
                f"{len(unverified)} remain unverified."
            ) if geodo_data else (
                "Geodo validation pending — use output/geodo_lookup_list.json to research IDs on geodo.ai."
            )
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "ESCALATED",
                "records_affected": count,
                "justification": (
                    f"{count} records reference CX-prefixed customer IDs absent from the master "
                    f"customer list. {geodo_note} "
                    f"All {count} records escalated to compliance officer — unknown customers "
                    f"are a direct audit failure risk."
                ),
                "geodo_verified_count": len(geodo_verified),
                "unverified_ids": unverified,
            })

        # ── Weight inflation → FLAG ────────────────────────────────────────────
        elif isubtype == "weight_inflation":
            flag(record_ids, "WEIGHT_INFLATION")
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records have weight_kg >5× the median for their part number. "
                    f"Consistent with a lbs-entered-as-kg error on individual records from recently "
                    f"acquired plants. Cannot auto-correct without confirmed unit; flagged for SME review."
                ),
            })

        # ── Ship before production → AUTO-FIX if ≤7 days, else FLAG ──────────
        elif isubtype == "ship_before_production":
            bad = df[df["ship_date"] < df["production_date"]].copy()
            gap = (bad["production_date"] - bad["ship_date"]).dt.days
            fixable = bad[gap <= 7]
            unfixable = bad[gap > 7]

            if len(fixable):
                df.loc[fixable.index, ["production_date", "ship_date"]] = (
                    bad.loc[fixable.index, ["ship_date", "production_date"]].values
                )
                audit_log.append({
                    "issue": f"{itype}/{isubtype}",
                    "action": "AUTO_FIXED",
                    "records_affected": len(fixable),
                    "justification": (
                        f"Swapped production_date and ship_date for {len(fixable)} records "
                        f"where the gap was ≤7 days — consistent with a field transposition typo."
                    ),
                })
            if len(unfixable):
                flag(unfixable["record_id"].tolist(), "DATE_INVERSION")
                audit_log.append({
                    "issue": f"{itype}/{isubtype}",
                    "action": "FLAGGED",
                    "records_affected": len(unfixable),
                    "justification": (
                        f"{len(unfixable)} records have ship_date >7 days before production_date. "
                        f"Gap too large to be a transposition — likely data corruption or backdated entry."
                    ),
                })

        # ── Negative quantity → FLAG ───────────────────────────────────────────
        elif isubtype == "non_positive_quantity":
            flag(record_ids, "IMPOSSIBLE_QUANTITY")
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records have quantity ≤ 0. "
                    f"Negative stock is physically impossible; may indicate reversed-sign returns "
                    f"entered in the wrong system. Requires source-document review."
                ),
            })

        # ── Status / future-date conflicts → FLAG ─────────────────────────────
        elif isubtype in ("status_future_ship_date", "future_production_date"):
            flag(record_ids, "DATE_STATUS_CONFLICT")
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records have a logical contradiction between status and date fields. "
                    f"Auditors will flag these as evidence of unreliable record-keeping."
                ),
            })

    # ── Apply fixes to dataframe ──────────────────────────────────────────────
    df_clean = df.drop(index=list(indices_to_drop)).copy()
    df_clean["audit_flags"] = df_clean["record_id"].map(
        lambda r: "; ".join(flags.get(r, [])) or ""
    )
    df_clean["audit_status"] = df_clean["record_id"].map(
        lambda r: "ESCALATED" if "ORPHANED_CUSTOMER_ID" in flags.get(r, [])
        else ("FLAGGED" if flags.get(r) else "CLEAN")
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    df_clean.to_csv(OUTPUT_DIR / "track01_cleaned.csv", index=False)

    n_escalated = (df_clean["audit_status"] == "ESCALATED").sum()
    n_flagged = (df_clean["audit_status"] == "FLAGGED").sum()
    n_clean = (df_clean["audit_status"] == "CLEAN").sum()

    action_summary = {
        "total_input_records": len(df),
        "duplicates_removed": len(indices_to_drop),
        "output_records": len(df_clean),
        "clean": int(n_clean),
        "flagged": int(n_flagged),
        "escalated": int(n_escalated),
        "audit_log": audit_log,
    }
    with open(OUTPUT_DIR / "audit_log.json", "w") as f:
        json.dump(action_summary, f, indent=2, default=str)

    # Store in Cognee (vector-only: add without cognify — zero LLM calls)
    cognee_summary = (
        f"Agent 3 (Act On It) results: "
        f"input={len(df)} records, removed {len(indices_to_drop)} duplicates, "
        f"output={len(df_clean)} records — "
        f"{n_clean} CLEAN / {n_flagged} FLAGGED / {n_escalated} ESCALATED. "
        f"Actions: "
        + "; ".join(f"{a['action']} {a['records_affected']} ({a['issue']})" for a in audit_log)
    )
    await cognee.add(cognee_summary, dataset_name="data_rescue")

    print(f"[Agent 3 — Act On It]  in={len(df)}  removed={len(indices_to_drop)}  out={len(df_clean)}")
    print(f"  CLEAN={n_clean}  FLAGGED={n_flagged}  ESCALATED={n_escalated}")
    for entry in audit_log:
        print(f"  [{entry['action']:12s}] {entry['issue']}  n={entry['records_affected']}")

    return action_summary
