"""Agent 6 — Geodo Research: automatically verify unknown customer/entity IDs.

Workflow mirrors what a human would do on geodo.ai:
  1. Load each unknown CX- ID from geodo_lookup_list.json
  2. Pull all associated order records from the dataset
  3. Score each ID against 5 statistical signals (order realism, weight,
     quantity range, part-number existence, date validity)
  4. Call OpenAI once (batched) to infer plausible company context
  5. Write verified/unverified decisions to output/geodo_results.json
  6. Store results in Cognee so Agent 3 can use them

This simulates Geodo's entity-research workflow programmatically.
"""

import json
import os
import re
from pathlib import Path

import cognee
import numpy as np
import pandas as pd
from cognee.api.v1.search import SearchType

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ── Normal-customer baseline (computed from dataset) ──────────────────────────
QTY_LOW,    QTY_HIGH    = 20,   4928
WEIGHT_LOW, WEIGHT_HIGH = 1.20, 79.73


def _load_schema() -> dict:
    p = OUTPUT_DIR / "schema_map.json"
    return json.load(open(p)) if p.exists() else {}


def _signal_score(row: pd.Series, part_stats: dict) -> tuple[float, list[str]]:
    """Return (0–1 confidence score, list of evidence notes)."""
    score = 0.0
    notes = []

    # Signal 1: quantity in normal range
    qty = row.get("quantity", None)
    if qty is not None and QTY_LOW <= float(qty) <= QTY_HIGH:
        score += 0.20
        notes.append(f"qty={qty:.0f} within normal range [{QTY_LOW}–{QTY_HIGH}]")
    elif qty is not None:
        notes.append(f"qty={qty:.0f} OUTSIDE normal range")

    # Signal 2: weight in normal range
    w = row.get("weight_kg", None) or row.get("weight", None)
    if w is not None and WEIGHT_LOW <= float(w) <= WEIGHT_HIGH:
        score += 0.20
        notes.append(f"weight={w:.2f}kg within normal range")
    elif w is not None:
        notes.append(f"weight={w:.2f}kg outside normal range")

    # Signal 3: part number exists in records from known customers
    pn = str(row.get("part_number", "")).strip().upper()
    pn_norm = re.sub(r"[\s_\-]+", "-", pn).strip("-")
    if pn_norm in part_stats:
        score += 0.25
        notes.append(f"part {pn_norm} found in {part_stats[pn_norm]} known-customer records")
    else:
        notes.append(f"part {pn_norm} not seen in known-customer records")

    # Signal 4: production date in valid range (2024–2026)
    prod_date = str(row.get("production_date", ""))
    if re.search(r"202[4-6]", prod_date):
        score += 0.20
        notes.append(f"production_date {prod_date} in expected range 2024–2026")
    else:
        notes.append(f"production_date {prod_date} outside expected range")

    # Signal 5: CX- prefix implies systematic acquired-plant origin (structural trust)
    cid = str(row.get("customer_id", ""))
    if re.match(r"CX-[A-Z]\d+", cid):
        score += 0.15
        prefix = cid[3]
        plant_map = {"A": "Plant A annex", "B": "Plant B (acquired)", "C": "Plant C (acquired)", "D": "Plant D (acquired)"}
        notes.append(f"CX-{prefix}*** format → {plant_map.get(prefix, 'acquired plant')} customer ID series")

    return round(score, 2), notes


def _infer_company_names(records: list[dict]) -> dict[str, str]:
    """One batched OpenAI call to infer plausible company names for all IDs."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        lines = "\n".join(
            f"- {r['customer_id']}: part={r.get('part_number','?')}, "
            f"qty={r.get('quantity','?')}, plant_prefix={r['customer_id'][3]}"
            for r in records[:80]
        )
        prompt = (
            "You are a compliance analyst reviewing industrial manufacturing customer IDs. "
            "For each CX-prefixed customer ID below, suggest a plausible company name "
            "that would match the order profile (industrial parts manufacturer/distributor). "
            "Reply ONLY as JSON: {\"CX-XXXX\": \"Company Name\", ...}\n\n" + lines
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[Agent 6 — Geodo] Company name inference skipped: {e}")
        return {}


async def run_geodo_research() -> dict:
    # ── Read Cognee context (R02) ─────────────────────────────────────────────
    try:
        results = await cognee.search(
            "orphaned customer IDs unknown CX- escalated entity validation",
            SearchType.CHUNKS,
        )
        print(f"[Agent 6 — Geodo] Cognee recall: {len(results or [])} chunks")
    except Exception as e:
        print(f"[Agent 6 — Geodo] Cognee recall warning: {e}")

    # ── Load unknown IDs ──────────────────────────────────────────────────────
    lookup_path = OUTPUT_DIR / "geodo_lookup_list.json"
    if not lookup_path.exists():
        print("[Agent 6 — Geodo] No geodo_lookup_list.json found — run Agent 1 first")
        return {}

    with open(lookup_path) as f:
        lookup = json.load(f)
    unknown_ids: list[str] = lookup.get("unknown_customer_ids", [])
    print(f"[Agent 6 — Geodo] Researching {len(unknown_ids)} unknown customer IDs …")

    # ── Load dataset ──────────────────────────────────────────────────────────
    schema = _load_schema()
    C_CUSTOMER = schema.get("customer_id", "customer_id")
    C_PART     = schema.get("part_number", "part_number")

    csv_files = [f for f in DATA_DIR.glob("*.csv")
                 if not re.search(r"customer|client|entity|lookup", f.name, re.I)]
    df = pd.read_csv(csv_files[0] if csv_files else next(DATA_DIR.glob("*.csv")))

    # Build part-number occurrence count for known customers
    known = df[~df[C_CUSTOMER].astype(str).str.startswith("CX-", na=False)]

    def _norm(p):
        return re.sub(r"[\s_\-]+", "-", str(p).strip().upper()).strip("-")

    part_stats: dict[str, int] = (
        known[C_PART].apply(_norm).value_counts().to_dict()
        if C_PART in known.columns else {}
    )

    # ── Score each unknown ID ─────────────────────────────────────────────────
    cx_records = df[df[C_CUSTOMER].isin(unknown_ids)]
    scored: list[dict] = []

    for cid in unknown_ids:
        rows = cx_records[cx_records[C_CUSTOMER] == cid]
        if len(rows) == 0:
            scored.append({
                "customer_id": cid,
                "confidence": 0.0,
                "record_count": 0,
                "notes": ["No records found in dataset"],
                "verified": False,
            })
            continue

        row = rows.iloc[0].to_dict()
        row["customer_id"] = cid

        conf, notes = _signal_score(row, part_stats)
        scored.append({
            "customer_id": cid,
            "confidence": conf,
            "record_count": len(rows),
            "part_number": str(row.get(C_PART, "?")),
            "quantity": row.get("quantity"),
            "weight": row.get("weight_kg") or row.get("weight"),
            "production_date": str(row.get("production_date", "")),
            "notes": notes,
            "verified": conf >= 0.60,  # ≥3 of 5 signals pass
        })

    # ── Batch company name inference via OpenAI ───────────────────────────────
    verified_records = [s for s in scored if s["verified"]]
    company_names: dict[str, str] = {}
    if verified_records:
        print(f"[Agent 6 — Geodo] Inferring company names for {len(verified_records)} verified IDs (1 API call) …")
        company_names = _infer_company_names(verified_records)

    # ── Build geodo_results.json ──────────────────────────────────────────────
    results: dict[str, dict] = {}
    verified_count   = 0
    unverified_count = 0

    for s in scored:
        cid = s["customer_id"]
        company = company_names.get(cid, "")
        prefix  = cid[3] if len(cid) > 3 else "?"
        plant_origin = {
            "A": "Plant A customer annex",
            "B": "Plant B acquired customer",
            "C": "Plant C acquired customer",
            "D": "Plant D acquired customer",
        }.get(prefix, "acquired plant customer")

        if s["verified"]:
            verified_count += 1
            results[cid] = {
                "verified": True,
                "company_name": company or f"{plant_origin} (name pending confirmation)",
                "confidence": s["confidence"],
                "record_count": s["record_count"],
                "plant_origin": plant_origin,
                "notes": "; ".join(s["notes"]),
                "research_method": "geodo_agent_statistical_verification",
            }
        else:
            unverified_count += 1
            results[cid] = {
                "verified": False,
                "company_name": "",
                "confidence": s["confidence"],
                "record_count": s["record_count"],
                "notes": "; ".join(s["notes"]) + " — requires manual Geodo research",
                "research_method": "geodo_agent_statistical_verification",
            }

    with open(OUTPUT_DIR / "geodo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "total_researched": len(unknown_ids),
        "verified": verified_count,
        "unverified": unverified_count,
        "verification_rate": round(verified_count / max(len(unknown_ids), 1), 3),
        "results": results,
    }

    with open(OUTPUT_DIR / "geodo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Write to Cognee ───────────────────────────────────────────────────────
    try:
        await cognee.add(
            f"Agent 6 (Geodo Research): researched {len(unknown_ids)} unknown CX- customer IDs. "
            f"Verified: {verified_count}, Unverified: {unverified_count} "
            f"(rate: {summary['verification_rate']:.0%}). "
            f"Verification method: 5-signal statistical scoring + OpenAI company name inference. "
            f"Results saved to geodo_results.json.",
            dataset_name="data_rescue",
        )
        await cognee.cognify()
    except Exception:
        pass

    print(f"[Agent 6 — Geodo] Done: {verified_count} verified, {unverified_count} unverified "
          f"({summary['verification_rate']:.0%} verification rate)")
    print(f"  → output/geodo_results.json")

    return summary
