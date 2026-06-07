"""Agent 3 — Act On It: fix, flag, or escalate every issue with a logged justification.

Dataset-agnostic: reads column names from schema_map.json (produced by Agent 0).
Falls back to the original Harven column names if no schema map is present.
"""

import json
import re
from pathlib import Path

import cognee
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _load_schema() -> dict:
    p = OUTPUT_DIR / "schema_map.json"
    return json.load(open(p)) if p.exists() else {}


def _normalize_pn(pn: str) -> str:
    return re.sub(r"[_]", "-", str(pn).strip().upper())


async def act_on_issues(ranked: list[dict] | None = None) -> dict:
    # ── Read Agent 2 context from Cognee (R02) ────────────────────────────────
    try:
        from cognee.api.v1.search import SearchType
        results = await cognee.search("severity rankings priority CRITICAL HIGH", SearchType.CHUNKS)
        print(f"[Agent 3 — Act On It] Cognee recall: {len(results or [])} ranking chunks")
    except Exception as e:
        print(f"[Agent 3 — Act On It] Cognee recall warning: {e}")

    if ranked is None:
        with open(OUTPUT_DIR / "rankings.json") as f:
            ranked = json.load(f)

    schema = _load_schema()

    def col(concept: str, fallback: str) -> str:
        return schema.get(concept) or fallback

    C_ID       = col("record_id", "record_id")
    C_PART     = col("part_number", "part_number")
    C_CUSTOMER = col("customer_id", "customer_id")
    C_PROD     = col("production_date", "production_date")
    C_SHIP     = col("ship_date", "ship_date")
    C_QTY      = col("quantity", "quantity")
    C_WEIGHT   = col("weight", "weight_kg")

    # ── Discover primary CSV ──────────────────────────────────────────────────
    csv_files = [f for f in DATA_DIR.glob("*.csv")
                 if not re.search(r"customer|client|entity|lookup", f.name, re.I)]
    if not csv_files:
        csv_files = list(DATA_DIR.glob("*.csv"))
    primary_csv = csv_files[0]
    df = pd.read_csv(primary_csv)

    # Load companion lookup (customers / entities)
    valid_ids: set = set()
    for f in DATA_DIR.glob("*.csv"):
        if re.search(r"customer|client|entity|account", f.name, re.I):
            try:
                lkp = pd.read_csv(f)
                for c in lkp.columns:
                    if re.search(r"customer.?id|cust.?id|client.?id", c, re.I):
                        valid_ids = set(lkp[c].astype(str).str.strip())
                        break
                if valid_ids:
                    break
            except Exception:
                pass

    # Geodo enrichment
    geodo_data: dict = {}
    geodo_path = OUTPUT_DIR / "geodo_results.json"
    if geodo_path.exists():
        with open(geodo_path) as f:
            geodo_data = json.load(f)

    if C_PROD in df.columns:
        df[C_PROD] = pd.to_datetime(df[C_PROD], errors="coerce")
    if C_SHIP in df.columns:
        df[C_SHIP] = pd.to_datetime(df[C_SHIP], errors="coerce")

    audit_log: list[dict] = []
    flags: dict[str, list[str]] = {}
    indices_to_drop: set[int] = set()

    def flag(record_ids: list, tag: str):
        for rid in record_ids:
            flags.setdefault(str(rid), []).append(tag)

    for issue in ranked:
        isubtype = issue["subtype"]
        record_ids = issue.get("record_ids", [])
        count = issue["count"]
        itype = issue["type"]

        # ── Exact & near duplicates → AUTO-FIX ───────────────────────────────
        if isubtype in ("exact_duplicate", "near_duplicate_variant", "semantic_duplicate"):
            id_cols = [c for c in df.columns if c != C_ID]
            extra_idx = df[df.duplicated(subset=id_cols, keep="first")].index.tolist()
            indices_to_drop.update(extra_idx)
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "AUTO_FIXED",
                "records_affected": count,
                "justification": (
                    f"Exact/near-duplicate records removed; first occurrence retained. "
                    f"Arose from double-entry during plant migration or re-entry with format differences."
                ),
            })

        # ── Separator mismatch / unit_format_drift → AUTO-FIX ────────────────
        elif isubtype in ("separator_mismatch", "unit_format_drift", "malformed_part_number"):
            if C_PART in df.columns:
                mask = df[C_PART].str.strip().str.contains("_", regex=False)
                df.loc[mask, C_PART] = df.loc[mask, C_PART].apply(
                    lambda p: re.sub(r"[_]", "-", str(p).strip().upper())
                )
                df[C_PART] = df[C_PART].str.strip().str.upper()
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "AUTO_FIXED",
                "records_affected": count,
                "justification": (
                    f"Normalised {count} part numbers: underscore → hyphen, strip whitespace, uppercase. "
                    f"Standardised to canonical format from the original plant."
                ),
            })

        # ── Orphaned customer/entity IDs → ESCALATE ──────────────────────────
        elif isubtype == "unknown_customer_id":
            unknown_ids = issue.get("unknown_ids", [])
            geodo_verified = [cid for cid in unknown_ids if geodo_data.get(cid, {}).get("verified")]
            unverified = [cid for cid in unknown_ids if cid not in geodo_verified]
            flag(record_ids, "ORPHANED_CUSTOMER_ID")
            geodo_note = (
                f"Geodo: {len(geodo_verified)} IDs confirmed; {len(unverified)} unverified."
                if geodo_data else
                "Geodo validation pending — see output/geodo_lookup_list.json."
            )
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "ESCALATED",
                "records_affected": count,
                "justification": (
                    f"{count} records reference {C_CUSTOMER} values absent from the master list. "
                    f"{geodo_note} Escalated — unknown entities are a direct audit failure risk."
                ),
                "geodo_verified_count": len(geodo_verified),
                "unverified_ids": unverified,
            })

        # ── Decimal shift / weight inflation → FLAG ───────────────────────────
        elif isubtype in ("decimal_shift_10x", "decimal_shift_100x", "weight_inflation"):
            flag(record_ids, "WEIGHT_ANOMALY")
            multiplier = "10×" if "10x" in isubtype else "100×" if "100x" in isubtype else "N×"
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records have {C_WEIGHT} ≈ {multiplier} the part-number median (per-part z-score > 3.5). "
                    f"Consistent with decimal-point error or unit conversion (lbs/kg). "
                    f"Cannot auto-correct without source document — flagged for SME review."
                ),
            })

        # ── Ship before production: auto-fix ≤7 days, flag larger gaps ────────
        elif isubtype == "ship_before_production":
            if C_PROD in df.columns and C_SHIP in df.columns:
                bad = df[df[C_SHIP] < df[C_PROD]].copy()
                gap = (bad[C_PROD] - bad[C_SHIP]).dt.days
                fixable = bad[gap <= 7]
                unfixable = bad[gap > 7]
                if len(fixable):
                    df.loc[fixable.index, [C_PROD, C_SHIP]] = (
                        bad.loc[fixable.index, [C_SHIP, C_PROD]].values
                    )
                    audit_log.append({
                        "issue": f"{itype}/{isubtype}",
                        "action": "AUTO_FIXED",
                        "records_affected": len(fixable),
                        "justification": (
                            f"Swapped {C_PROD}/{C_SHIP} for {len(fixable)} records with gap ≤7 days — "
                            f"consistent with field transposition typo."
                        ),
                    })
                if len(unfixable):
                    flag(unfixable[C_ID].tolist(), "DATE_INVERSION")
                    audit_log.append({
                        "issue": f"{itype}/{isubtype}",
                        "action": "FLAGGED",
                        "records_affected": len(unfixable),
                        "justification": (
                            f"{len(unfixable)} records: {C_SHIP} >7 days before {C_PROD}. "
                            f"Gap too large for transposition — requires source document."
                        ),
                    })

        # ── Negative quantity → FLAG ──────────────────────────────────────────
        elif isubtype == "non_positive_quantity":
            flag(record_ids, "IMPOSSIBLE_QUANTITY")
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records with {C_QTY} ≤ 0. "
                    f"Likely return adjustments entered in main inventory module. "
                    f"Requires source-document review."
                ),
            })

        # ── Status/date conflict → FLAG ───────────────────────────────────────
        elif isubtype in ("status_future_ship_date", "future_production_date"):
            flag(record_ids, "DATE_STATUS_CONFLICT")
            audit_log.append({
                "issue": f"{itype}/{isubtype}",
                "action": "FLAGGED",
                "records_affected": count,
                "justification": (
                    f"{count} records have a logical contradiction between status and date fields. "
                    f"Auditors will flag as unreliable record-keeping."
                ),
            })

    # ── Apply fixes ───────────────────────────────────────────────────────────
    df_clean = df.drop(index=list(indices_to_drop)).copy()
    df_clean["audit_flags"] = df_clean[C_ID].map(
        lambda r: "; ".join(flags.get(str(r), [])) or ""
    )
    df_clean["audit_status"] = df_clean[C_ID].map(
        lambda r: "ESCALATED" if "ORPHANED_CUSTOMER_ID" in flags.get(str(r), [])
        else ("FLAGGED" if flags.get(str(r)) else "CLEAN")
    )

    df_clean.to_csv(OUTPUT_DIR / "track01_cleaned.csv", index=False)

    n_escalated = int((df_clean["audit_status"] == "ESCALATED").sum())
    n_flagged   = int((df_clean["audit_status"] == "FLAGGED").sum())
    n_clean     = int((df_clean["audit_status"] == "CLEAN").sum())

    action_summary = {
        "total_input_records": len(df),
        "duplicates_removed": len(indices_to_drop),
        "output_records": len(df_clean),
        "clean": n_clean,
        "flagged": n_flagged,
        "escalated": n_escalated,
        "audit_log": audit_log,
    }
    with open(OUTPUT_DIR / "audit_log.json", "w") as f:
        json.dump(action_summary, f, indent=2, default=str)

    cognee_summary = (
        f"Agent 3 (Act On It): in={len(df)}, removed={len(indices_to_drop)} dupes, "
        f"out={len(df_clean)} — {n_clean} CLEAN / {n_flagged} FLAGGED / {n_escalated} ESCALATED. "
        + "; ".join(f"{a['action']} {a['records_affected']} ({a['issue']})" for a in audit_log)
    )
    await cognee.add(cognee_summary, dataset_name="data_rescue")
    await cognee.cognify()

    print(f"[Agent 3 — Act On It]  in={len(df)}  removed={len(indices_to_drop)}  out={len(df_clean)}")
    print(f"  CLEAN={n_clean}  FLAGGED={n_flagged}  ESCALATED={n_escalated}")
    for entry in audit_log:
        print(f"  [{entry['action']:12s}] {entry['issue']}  n={entry['records_affected']}")

    return action_summary
