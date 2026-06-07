# VibeForward M-Agents — Track 01: Data Rescue
**Team:** Five Guys with Sixty Percent Confidence | VibeForward M-2 Hackathon @ Fordham University

Harven Manufacturing's warehouse data is corrupted 4 days before a regulatory audit.  
This pipeline finds, ranks, fixes, explains, and guides remediation of every data-quality issue — using a 6-agent architecture connected through **Cognee** memory.

Works on **any dataset** — columns are auto-discovered at runtime.

---

## Architecture

```
Any CSV dataset
        │
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 0 — SCHEMA DETECT  (utils/schema_detect.py)           │
   │  Auto-maps column names to semantic concepts                  │
   │  (record_id, part_number, customer_id, weight, dates…)        │
   │  → schema_map.json  (used by all downstream agents)          │
   └────┬──────────────────────────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 1 — FIND IT  (agents/find_it.py)                      │
   │  6 issue classes (per-part statistical baselining):           │
   │    • exact_duplicate / near_duplicate_variant                 │
   │    • unit_format_drift  (temporal separator clustering)       │
   │    • orphaned_reference (unknown customer/entity IDs)         │
   │    • decimal_shift_weight (×10 / ×100 per-part z-score)      │
   │    • impossible_value (dates, quantity, status conflicts)     │
   │  → writes findings to Cognee                                  │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory  +  geodo_lookup_list.json → Geodo
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 2 — RANK IT  (agents/rank_it.py)  [PyMC]             │
   │  3-method convergence (Decision-Lab philosophy):              │
   │    Method A: PyMC Beta posterior (Bayesian, data-driven)     │
   │    Method B: Frequency × base-weight score                   │
   │    Method C: Regulatory hard-stop heuristic                  │
   │  Ensemble + convergence check → CRITICAL / HIGH / MEDIUM / LOW│
   │  → writes rankings to Cognee                                  │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 3 — ACT ON IT  (agents/act_on_it.py)  [Geodo]        │
   │  AUTO-FIX: duplicates, separator/format conflicts            │
   │  AUTO-FIX: date transpositions ≤7 days                       │
   │  FLAG:     decimal-shift weights, impossible values           │
   │  ESCALATE: unknown entity IDs (enriched via Geodo)           │
   │  → writes action log to Cognee                                │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 5 — RECOMMEND IT  (agents/recommend_it.py)            │
   │  Scores every flagged/escalated issue:                        │
   │    urgency × feasibility × confidence                         │
   │  → DO_NOW / DO_TODAY / DO_THIS_WEEK action plan              │
   │  → step-by-step instructions per issue type                  │
   │  → writes recommendations to Cognee                           │
   └────┬──────────────────────────────────────────────────────────┘
        │ Cognee memory recall (all agents)
   ┌────▼──────────────────────────────────────────────────────────┐
   │  Agent 4 — EXPLAIN IT  (agents/explain_it.py)                │
   │  Robinhood-style dark dashboard (output/audit_report.html):  │
   │    • Verdict banner (RED / YELLOW / GREEN)                   │
   │    • Stat cards + animated issue bar chart                    │
   │    • 5 tabs: Dashboard · Issues · Action Plan ·              │
   │              Decision Log · Benchmark                         │
   │    • Embedded compliance chatbot (no backend needed)          │
   │    • Download buttons: HTML report + clean CSV + audit log   │
   └───────────────────────────────────────────────────────────────┘
```

**Stack:** Cognee (memory) · PyMC v5 (Bayesian severity) · Decision-Lab (convergence reasoning) · Geodo (entity validation) · Trupeer (demo video) · LLM API calls at runtime: **0**

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key (.env already set if running locally)
cp .env.example .env        # add OPENAI_API_KEY + LLM_API_KEY (same key, for Cognee)

# 3. Run the full pipeline (~2 min)
python main.py

# 4. Open the dashboard
open output/audit_report.html
```

**That's it.** The pipeline auto-detects your dataset's columns — no config needed.

---

## Geodo Step (optional enrichment)

After Agent 1 runs, `output/geodo_lookup_list.json` contains all unknown entity IDs.

1. Open [geodo.ai](https://www.geodo.ai) in your browser
2. Search each ID; verify whether it's a real company
3. Save findings to `output/geodo_results.json` before Agent 3 runs:

```json
{
  "CX-A228": { "verified": true,  "company_name": "Acme Parts Ltd", "notes": "confirmed subsidiary" },
  "CX-A630": { "verified": false, "company_name": "",               "notes": "not found" }
}
```

Agent 3 reads this automatically and uses it in ESCALATE decisions.

---

## Output Files

| File | Description |
|---|---|
| `output/audit_report.html` | **Open this** — Robinhood dark dashboard with chatbot |
| `output/track01_cleaned.csv` | Cleaned dataset with `audit_status` + `audit_flags` |
| `output/audit_log.json` | Every action with justification (R07) |
| `output/recommendations.json` | Prioritized compliance officer action plan |
| `output/findings.json` | Raw issue list from Agent 1 |
| `output/rankings.json` | PyMC severity scores + reasoning from Agent 2 |
| `output/schema_map.json` | Auto-detected column mapping |
| `output/geodo_lookup_list.json` | Entity IDs to validate on Geodo |

---

## Issue Classes Detected

6 classes across the dataset taxonomy (naive range filters only find tier 1; subtle classes require per-part statistical baselining):

| Class | Detection Method | Typical Action |
|---|---|---|
| Exact duplicate | All-column deduplication | AUTO-FIXED |
| Near-duplicate variant | Normalised key match (case/whitespace/separator) | AUTO-FIXED |
| Unit-format drift | Temporal clustering of separator variants (firmware artefact) | AUTO-FIXED |
| Orphaned reference | Companion lookup file diff | ESCALATED |
| Decimal-shift weight | Per-part z-score + ratio bucketing (×10 / ×100) | FLAGGED |
| Impossible value | Date inversion, negative qty, status contradiction | AUTO-FIXED / FLAGGED |

---

## Hackathon Rules Compliance

| Rule | How |
|---|---|
| **R01** Cognee is the memory layer | Every agent calls `cognee.add()` (write) then `cognee.search()` (read) |
| **R02** Every agent reads from Cognee | Agents 2–4 each call `cognee.search()` before acting |
| **R05** Product Brief submitted | `PRODUCT_BRIEF.md` — drafted from website + dataset only |
| **R07** Every decision has visible reason | `justification` in every audit_log entry; `ranking_reasoning` in every ranking |
| **R08** Agent 4 output downloadable | HTML + CSV + JSON download buttons in dashboard |
| **Benchmark** | `utils/benchmark.py` scores recall/precision vs ~850 seeded Kaggle issues |
