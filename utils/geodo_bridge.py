"""Geodo bridge — exports a lookup list of unknown customer IDs for manual research
on geodo.ai, then reads back any results the team has filled in.

Geodo is a no-code web platform (no Python API), so the integration is:
  1. Run export_geodo_lookup() → writes output/geodo_lookup_list.json
  2. Team member opens geodo.ai, searches each CX- ID, fills in the template
  3. Team member saves results to output/geodo_results.json
  4. Agent 3 reads geodo_results.json automatically

Template for output/geodo_results.json:
{
  "CX-A001": { "verified": true,  "company_name": "Acme Corp",  "notes": "confirmed subsidiary" },
  "CX-A002": { "verified": false, "company_name": "",           "notes": "not found in any registry" }
}
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def export_geodo_lookup() -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "track01_data_rescue.csv")
    customers = pd.read_csv(DATA_DIR / "track01_customers.csv")
    valid_ids = set(customers["customer_id"].str.strip())

    orphaned_ids = sorted(df[~df["customer_id"].isin(valid_ids)]["customer_id"].unique())

    payload = {
        "unknown_customer_ids": orphaned_ids,
        "count": len(orphaned_ids),
        "instructions": (
            "Open https://www.geodo.ai in your browser. "
            "For each customer ID below, search for the company name or ID. "
            "Verify whether it is a real entity (possibly from an acquired plant's CRM). "
            "Save your findings to output/geodo_results.json using the template format above."
        ),
        "results_template": {
            cid: {"verified": False, "company_name": "", "notes": ""}
            for cid in orphaned_ids
        },
    }

    lookup_path = OUTPUT_DIR / "geodo_lookup_list.json"
    with open(lookup_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Geodo Bridge] Exported {len(orphaned_ids)} unknown customer IDs → {lookup_path}")
    print("  → Open geodo.ai, research each ID, save results to output/geodo_results.json")
    return payload
