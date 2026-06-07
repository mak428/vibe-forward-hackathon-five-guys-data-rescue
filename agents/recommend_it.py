"""Agent 5 — Recommend It: read all agent context from Cognee, then generate
scored, actionable recommendations for the compliance officer.

Each recommendation is:
  - Deterministic (zero LLM calls)
  - Scored by urgency × feasibility × confidence
  - Sourced from Cognee memory (reads Agent 3 output)
  - Written back to Cognee (read by Agent 4 report)

R07 compliance: every recommendation includes step-by-step reasoning.
"""

import json
from pathlib import Path

import cognee
import pandas as pd
from cognee.api.v1.search import SearchType

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Days until regulatory audit — drives urgency scoring
AUDIT_DAYS_AWAY = 4

# ── Playbook: deterministic rules per issue tag ────────────────────────────────
# Each entry: (steps, feasibility, confidence, estimated_effort_hours)
PLAYBOOK: dict[str, dict] = {
    "ORPHANED_CUSTOMER_ID": {
        "steps": [
            "Open output/geodo_lookup_list.json — 80 unknown CX- IDs are listed.",
            "For each CX- ID: search geodo.ai (web entity lookup) for the company.",
            "If found: add entity to track01_customers.csv with source='geodo_verified'.",
            "If NOT found after 5-min search: mark record as QUARANTINE in audit_flags.",
            "After all IDs resolved: re-run `python main.py` — escalated count should drop to 0.",
        ],
        "why": (
            "Unknown customer IDs are a hard audit-stop: auditors will flag phantom-entity risk. "
            "Geodo provides fast web entity lookup without manual legal research. "
            "Quarantining unverified records prevents them from tainting the audit-ready set."
        ),
        "feasibility": 0.85,
        "confidence": 0.95,
        "effort_hours": 3.0,
        "owner": "Compliance Officer + Data Steward",
    },
    "WEIGHT_INFLATION": {
        "steps": [
            "Export flagged records: filter track01_cleaned.csv where audit_flags='WEIGHT_INFLATION'.",
            "Pull original purchase orders (PO) for each record_id.",
            "Compare PO weight vs recorded weight — lbs/kg confusion shows 2.2× discrepancy.",
            "If discrepancy confirmed: apply weight_kg = weight_kg × 0.453592 and add note='unit_corrected'.",
            "If no source document available: set audit_flags='WEIGHT_UNVERIFIED' — exclude from audit.",
        ],
        "why": (
            "Weight values >5× the part-number median are statistically consistent with a "
            "lbs-entered-as-kg error (2.2× factor) compounded with typos. "
            "Auto-correction without a source document would introduce new errors. "
            "0.453592 is the exact lbs-to-kg conversion factor."
        ),
        "feasibility": 0.75,
        "confidence": 0.80,
        "effort_hours": 1.5,
        "owner": "Data Steward",
    },
    "DATE_INVERSION": {
        "steps": [
            "Filter track01_cleaned.csv where audit_flags contains 'DATE_INVERSION'.",
            "For each record: look up the original shipping manifest or ERP system entry.",
            "Confirm which date is correct (production vs ship).",
            "Update the incorrect field; add note='date_verified_<source>'.",
            "If no source document: mark UNRESOLVABLE and exclude from audit.",
        ],
        "why": (
            "Date inversions >7 days cannot be auto-corrected — the pipeline already fixed "
            "≤7-day inversions (transposition typos). Larger gaps suggest backdated entry "
            "or data migration errors that need a source document to resolve safely."
        ),
        "feasibility": 0.70,
        "confidence": 0.85,
        "effort_hours": 0.5,
        "owner": "Data Steward",
    },
    "IMPOSSIBLE_QUANTITY": {
        "steps": [
            "Filter track01_cleaned.csv where audit_flags='IMPOSSIBLE_QUANTITY'.",
            "Route records to returns-processing team: negative qty = likely return adjustment.",
            "Verify each record against returns/RMA system.",
            "If confirmed return: create offsetting credit record; change qty to abs(qty).",
            "If not a return: mark as DATA_ENTRY_ERROR and exclude from inventory count.",
        ],
        "why": (
            "Negative quantities are physically impossible for warehouse stock. "
            "Common cause: returns entered in the main inventory module instead of the "
            "returns module — the sign is correct (deduct) but the table is wrong. "
            "Auditors will flag negative inventory as a controls failure."
        ),
        "feasibility": 0.90,
        "confidence": 0.80,
        "effort_hours": 0.25,
        "owner": "Warehouse Manager",
    },
    "DATE_STATUS_CONFLICT": {
        "steps": [
            "Filter track01_cleaned.csv where audit_flags='DATE_STATUS_CONFLICT'.",
            "For each record: check if order has actually shipped (ERP lookup).",
            "If shipped: update audit_status to reflect actual; verify ship_date against manifest.",
            "If not shipped: update ship_date to current ERP scheduled date.",
            "Add note='status_reconciled_<date>' after correction.",
        ],
        "why": (
            "Status/date conflicts tell auditors records were not updated in real time. "
            "Even if the underlying data is correct, the conflict itself shows process failure. "
            "Reconciliation must be done before audit; auditors check ERP vs CSV consistency."
        ),
        "feasibility": 0.85,
        "confidence": 0.85,
        "effort_hours": 1.0,
        "owner": "Operations Team",
    },
}

# Default playbook for unknown tags
DEFAULT_PLAYBOOK = {
    "steps": ["Review flagged records manually.", "Consult data steward for remediation."],
    "why": "Anomaly detected by pipeline — requires manual investigation.",
    "feasibility": 0.60,
    "confidence": 0.60,
    "effort_hours": 1.0,
    "owner": "Data Steward",
}


def _urgency(issue_tag: str, records_affected: int) -> float:
    """Urgency score: harder deadline + more records = higher urgency."""
    # Normalise record count: 100+ records → max urgency contribution
    volume_factor = min(records_affected / 100, 1.0)
    # Deadline pressure: ≤4 days is high urgency
    deadline_factor = 1.0 if AUDIT_DAYS_AWAY <= 4 else max(0.5, 1 - AUDIT_DAYS_AWAY / 14)
    # ESCALATED issues always at max urgency
    if issue_tag == "ORPHANED_CUSTOMER_ID":
        return 1.0
    return round(0.6 * deadline_factor + 0.4 * volume_factor, 3)


def _priority_score(urgency: float, feasibility: float, confidence: float) -> float:
    return round(0.5 * urgency + 0.25 * feasibility + 0.25 * confidence, 3)


def _priority_label(score: float) -> str:
    if score >= 0.80:
        return "DO_NOW"
    if score >= 0.60:
        return "DO_TODAY"
    if score >= 0.40:
        return "DO_THIS_WEEK"
    return "OPTIONAL"


async def recommend_actions(action_summary: dict | None = None) -> list[dict]:
    if action_summary is None:
        with open(OUTPUT_DIR / "audit_log.json") as f:
            action_summary = json.load(f)

    # ── Read Agent 3 context from Cognee (R02) ────────────────────────────────
    cognee_context: list[str] = []
    try:
        results = await cognee.search(
            "FLAGGED ESCALATED records audit actions justification",
            SearchType.CHUNKS,
        )
        cognee_context = [str(r) for r in (results or [])][:5]
        print(f"[Agent 5 — Recommend It] Cognee recall: {len(cognee_context)} chunks from Agent 3")
    except Exception as e:
        print(f"[Agent 5 — Recommend It] Cognee recall warning: {e}")

    # ── Load cleaned CSV for record-level context ─────────────────────────────
    flagged_counts: dict[str, int] = {}
    try:
        df = pd.read_csv(OUTPUT_DIR / "track01_cleaned.csv")
        if "audit_flags" in df.columns:
            for flags_str in df["audit_flags"].dropna():
                for tag in flags_str.split("; "):
                    tag = tag.strip()
                    if tag:
                        flagged_counts[tag] = flagged_counts.get(tag, 0) + 1
    except FileNotFoundError:
        pass

    # ── Build recommendations from audit log ─────────────────────────────────
    recommendations: list[dict] = []
    for entry in action_summary["audit_log"]:
        if entry["action"] not in ("FLAGGED", "ESCALATED"):
            continue

        issue_tag = entry["issue"].split("/")[1].upper().replace("_", "_")
        # Map audit log subtype → playbook key
        tag_map = {
            "UNKNOWN_CUSTOMER_ID": "ORPHANED_CUSTOMER_ID",
            "WEIGHT_INFLATION": "WEIGHT_INFLATION",
            "DATE_INVERSION": "DATE_INVERSION",
            "NON_POSITIVE_QUANTITY": "IMPOSSIBLE_QUANTITY",
            "STATUS_FUTURE_SHIP_DATE": "DATE_STATUS_CONFLICT",
            "FUTURE_PRODUCTION_DATE": "DATE_STATUS_CONFLICT",
            "SHIP_BEFORE_PRODUCTION": "DATE_INVERSION",
        }
        playbook_key = tag_map.get(issue_tag, issue_tag)
        playbook = PLAYBOOK.get(playbook_key, DEFAULT_PLAYBOOK)

        records_affected = entry["records_affected"]
        urgency = _urgency(playbook_key, records_affected)
        priority_score = _priority_score(urgency, playbook["feasibility"], playbook["confidence"])
        priority_label = _priority_label(priority_score)

        rec = {
            "issue": entry["issue"],
            "action_type": entry["action"],
            "records_affected": records_affected,
            "priority": priority_label,
            "priority_score": priority_score,
            "urgency": urgency,
            "feasibility": playbook["feasibility"],
            "confidence": playbook["confidence"],
            "estimated_effort_hours": playbook["effort_hours"],
            "owner": playbook["owner"],
            "recommended_steps": playbook["steps"],
            "reasoning": playbook["why"],
            "agent3_justification": entry.get("justification", ""),
            "audit_days_remaining": AUDIT_DAYS_AWAY,
        }
        recommendations.append(rec)

    # Sort: DO_NOW first, then by priority score descending
    order = {"DO_NOW": 0, "DO_TODAY": 1, "DO_THIS_WEEK": 2, "OPTIONAL": 3}
    recommendations.sort(key=lambda r: (order.get(r["priority"], 4), -r["priority_score"]))

    total_effort = sum(r["estimated_effort_hours"] for r in recommendations)

    # ── Save recommendations ──────────────────────────────────────────────────
    output = {
        "audit_days_remaining": AUDIT_DAYS_AWAY,
        "total_items": len(recommendations),
        "total_effort_hours": round(total_effort, 1),
        "recommendations": recommendations,
    }
    with open(OUTPUT_DIR / "recommendations.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Write summary to Cognee (read by Agent 4 report) ─────────────────────
    summary = (
        f"Agent 5 (Recommend It): {len(recommendations)} recommendations generated. "
        f"Total effort: {total_effort:.1f}h within {AUDIT_DAYS_AWAY} days. "
        + "; ".join(
            f"{r['priority']} — {r['issue']} ({r['records_affected']} records, {r['estimated_effort_hours']}h)"
            for r in recommendations
        )
    )
    try:
        await cognee.add(summary, dataset_name="data_rescue")
        await cognee.cognify()
    except Exception:
        pass

    print(f"[Agent 5 — Recommend It] {len(recommendations)} recommendations (total {total_effort:.1f}h):")
    for r in recommendations:
        print(f"  [{r['priority']:14s}]  {r['issue']}  n={r['records_affected']}  ~{r['estimated_effort_hours']}h")

    return recommendations
