# VibeForward M-Agents — Track 01: Data Rescue
**Team:** Five Guys with Sixty Percent Confidence | VibeForward M-2 Hackathon @ Fordham University

Harven Manufacturing's warehouse data is corrupted 4 days before a regulatory audit.  
This pipeline finds, ranks, fixes, and explains every data-quality issue using a 4-agent architecture connected through **Cognee** memory.

---

## Architecture

```
track01_data_rescue.csv
        │
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 1 — FIND IT  (agents/find_it.py)                      │
   │  Detects: duplicates · separator conflicts · weight inflation  │
   │           orphaned CX- customer IDs · impossible values       │
   │  → stores findings summary in Cognee                         │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 2 — RANK IT  (agents/rank_it.py)  [PyMC]             │
   │  Scores each issue type via PyMC Beta posterior              │
   │  → stores severity rankings in Cognee                        │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory  +  output/geodo_lookup_list.json → Geodo
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 3 — ACT ON IT  (agents/act_on_it.py)  [Geodo]        │
   │  AUTO-FIX: duplicates, separator conflicts, malformed names  │
   │  FLAG:     weight inflation, impossible dates/quantities      │
   │  ESCALATE: unknown CX- customers (validated via Geodo)       │
   │  → stores action log in Cognee                               │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory recall
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 4 — EXPLAIN IT  (agents/explain_it.py)  [Claude]     │
   │  Recalls all findings from Cognee, asks Claude to write      │
   │  a plain-English audit readiness report                      │
   │  → output/audit_report.html                                  │
   └───────────────────────────────────────────────────────────────┘
```

**Stack:** Cognee (memory) · PyMC (Bayesian severity) · Geodo (entity validation) · Claude (narrative)

---

## Quickstart

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and LLM_API_KEY (OpenAI, for Cognee)

# 3. Run the pipeline
python main.py
```

### Geodo step (between Agent 1 and Agent 3)

After Agent 1 runs, `output/geodo_lookup_list.json` contains 80 unknown CX- customer IDs.

1. Open [geodo.ai](https://www.geodo.ai) in your browser
2. Research each CX- ID (search by company name / region)
3. Save your findings to `output/geodo_results.json`:

```json
{
  "CX-A228": { "verified": true,  "company_name": "Acme Parts Ltd", "notes": "confirmed subsidiary of Plant B acquiree" },
  "CX-A630": { "verified": false, "company_name": "",               "notes": "not found in any registry" }
}
```

Agent 3 reads this file automatically and factors it into its ESCALATE decisions.

---

## Outputs

| File | Description |
|---|---|
| `output/audit_report.html` | Plain-English compliance report (open in browser) |
| `output/track01_cleaned.csv` | Cleaned dataset with `audit_status` + `audit_flags` columns |
| `output/audit_log.json` | Machine-readable action log (every decision justified) |
| `output/findings.json` | Raw issue list from Agent 1 |
| `output/rankings.json` | PyMC severity scores from Agent 2 |
| `output/geodo_lookup_list.json` | CX- IDs to validate on Geodo |

---

## Issues Found (5,000 records, ~850 seeded)

| Issue | Count | Action |
|---|---|---|
| Part number separator conflict (BOLT_119 vs BOLT-119) | 1,252 | AUTO-FIXED |
| Semantic duplicates (same data, different record_id) | 130 | AUTO-FIXED |
| Malformed part numbers (whitespace / lowercase) | 63 | AUTO-FIXED |
| Unknown CX- customer IDs (orphaned references) | 80 | ESCALATED |
| Weight inflation > 5× median (unit error) | 35 | FLAGGED |
| Ship date before production date | 10 | AUTO-FIXED (2) / FLAGGED (8) |
| Status/date logical conflict | 17 | FLAGGED |
| Future production dates | 5 | FLAGGED |
| Negative quantities | 5 | FLAGGED |

