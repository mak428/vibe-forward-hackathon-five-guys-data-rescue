"""Geodo bridge — exports unknown entity IDs for manual research on geodo.ai.

Dataset-agnostic: reads schema_map.json to find the customer/entity column name.

Integration workflow:
  1. Run export_geodo_lookup() → writes output/geodo_lookup_list.json
  2. Team member opens geodo.ai, searches each unknown ID, fills in the template
  3. Team member saves results to output/geodo_results.json
  4. Agent 3 reads geodo_results.json automatically

Template for output/geodo_results.json:
{
  "CX-A001": { "verified": true,  "company_name": "Acme Corp",  "notes": "confirmed subsidiary" },
  "CX-A002": { "verified": false, "company_name": "",           "notes": "not found in any registry" }
}
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _load_schema() -> dict:
    p = OUTPUT_DIR / "schema_map.json"
    return json.load(open(p)) if p.exists() else {}


def export_geodo_lookup() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    schema = _load_schema()
    C_CUSTOMER = schema.get("customer_id", "customer_id")

    # Load findings.json if available — use unknown_ids from Agent 1
    findings_path = OUTPUT_DIR / "findings.json"
    if findings_path.exists():
        with open(findings_path) as f:
            findings = json.load(f)
        for iss in findings:
            if iss.get("subtype") == "unknown_customer_id" and "unknown_ids" in iss:
                orphaned_ids = sorted(iss["unknown_ids"])
                _write_lookup(orphaned_ids)
                return _build_payload(orphaned_ids)

    # Fallback: recompute from CSV
    csv_files = [f for f in DATA_DIR.glob("*.csv")
                 if not re.search(r"customer|client|entity|lookup", f.name, re.I)]
    if not csv_files:
        csv_files = list(DATA_DIR.glob("*.csv"))
    df = pd.read_csv(csv_files[0])

    valid_ids: set = set()
    for f in DATA_DIR.glob("*.csv"):
        if re.search(r"customer|client|entity|account", f.name, re.I):
            lkp = pd.read_csv(f)
            for c in lkp.columns:
                if re.search(r"customer.?id|cust.?id|client.?id", c, re.I):
                    valid_ids = set(lkp[c].astype(str).str.strip())
                    break
            if valid_ids:
                break

    if C_CUSTOMER not in df.columns:
        print(f"[Geodo Bridge] Warning: column '{C_CUSTOMER}' not found — skipping lookup export")
        return {}

    orphaned_ids = sorted(df[~df[C_CUSTOMER].astype(str).str.strip().isin(valid_ids)][C_CUSTOMER].unique())
    _write_lookup(orphaned_ids)
    return _build_payload(orphaned_ids)


def _build_payload(orphaned_ids: list) -> dict:
    return {
        "unknown_customer_ids": orphaned_ids,
        "count": len(orphaned_ids),
        "instructions": (
            "Open https://www.geodo.ai in your browser. "
            "For each ID below, search for the company. "
            "Verify whether it is a real entity. "
            "Save findings to output/geodo_results.json using the template format."
        ),
        "results_template": {
            cid: {"verified": False, "company_name": "", "notes": ""}
            for cid in orphaned_ids
        },
    }


def _write_lookup(orphaned_ids: list) -> None:
    lookup_path = OUTPUT_DIR / "geodo_lookup_list.json"
    with open(lookup_path, "w") as f:
        json.dump(_build_payload(orphaned_ids), f, indent=2)
    print(f"[Geodo Bridge] {len(orphaned_ids)} unknown IDs → {lookup_path}")
    print("  → Research on geodo.ai, save results to output/geodo_results.json")
