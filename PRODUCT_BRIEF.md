# Product Brief — Step 0
## Team: Five Guys with Sixty Percent Confidence
## Hackathon: VibeForward M² · Track 01: Data Rescue

---

## Who Is This For?

**Primary user:** A compliance officer at Harven Manufacturing — non-technical, no database skills, 4 days before a regulatory audit.

**Their problem:** The warehouse records across Plants A, B, and C (two recently acquired) cannot be trusted. The dataset has duplicates, naming inconsistencies, impossible values, and unknown customer references. The compliance officer cannot tell what is safe to submit to auditors.

**Secondary user:** The data steward who must act on flagged and escalated records before the audit.

---

## What Does the Product Do?

A four-agent automated pipeline that:

1. **Finds** every data-quality issue in the warehouse dataset (duplicates, unit conflicts, impossible values, orphaned references)
2. **Ranks** each issue by audit severity using a Bayesian probabilistic model (PyMC) with visible confidence intervals
3. **Acts** — auto-fixes what is safe to fix, flags what needs a human, and escalates what would cause an audit failure
4. **Explains** — produces a plain-English downloadable audit readiness report with a RED / YELLOW / GREEN verdict and a full decision log

All agent findings are stored in and retrieved from **Cognee** — every agent reads what the previous agent wrote.

---

## Success Conditions (Judged Against This Document)

| # | Condition | Measurable Target |
|---|-----------|-------------------|
| S1 | All seeded data-quality issues identified | ≥ 850 flagged issues across 9 categories |
| S2 | Clean dataset produced | Duplicates removed; part numbers normalised; clean CSV exported |
| S3 | Every decision has a logged reason | `audit_log.json` has `justification` field for every action |
| S4 | Bayesian severity visible | PyMC posterior mean + 95% CI shown per issue in report |
| S5 | Geodo validation integrated | 80 unknown CX- customer IDs exported; research workflow documented |
| S6 | Agent 4 output downloadable | HTML report + CSV downloadable directly from product interface |
| S7 | Compliance officer can use product | Judge runs `python main.py` and reads report; no engineering knowledge needed |
| S8 | Cognee memory demonstrable | Each agent's Cognee write and read is logged in terminal output |

---

## Scope

**Dataset:** `track01_data_rescue.csv` — 5,000 warehouse records, Plants A / B / C  
**Companion:** `track01_customers.csv` — customer master list  
**Track:** 01 — Data Rescue  
**Kaggle benchmark:** Used; findings compared against ~850 seeded issues

**In scope:**
- All 9 data-quality issue categories discoverable in the dataset
- Automated fixes (duplicates, separator normalisation, malformed names, date transpositions)
- Flagging (weight anomalies, impossible values, logical conflicts)
- Escalation (unknown customer IDs — requires Geodo entity research)
- Downloadable HTML report + clean CSV

**Out of scope:**
- Real-time streaming ingestion
- Multi-user access control
- ERP system integration

---

## Stack

| Tool | Role |
|------|------|
| **Cognee** | Memory layer — every agent writes findings; every subsequent agent reads them |
| **PyMC** | Bayesian Beta posterior severity scoring with visible confidence intervals |
| **Decision Lab (dlab-cli)** | Multi-approach convergence reasoning for Agent 2 ranking |
| **Decision Hub (dhub)** | `pymc-labs/pymc-modeling` skill validates our Bayesian approach |
| **Geodo** | Domain Expert researches 80 unknown CX- customer IDs for escalation decisions |
| **Trupeer** | 5-minute demo video showing pipeline running on real data |

---

## Agent Architecture

```
Dataset → Agent 1 (Find It) → [Cognee write] →
          Agent 2 (Rank It / PyMC / Decision Lab) → [Cognee read+write] →
          Agent 3 (Act On It / Geodo) → [Cognee read+write] →
          Agent 4 (Explain It) → [Cognee read] → downloadable HTML report + CSV
```

---

*Brief drafted from hackathon rules at vibe-forward.vercel.app and dataset README_track01.md.*
