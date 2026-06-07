"""VibeForward M-Agents Hackathon — Track 01: Data Rescue
Team: Five Guys with Sixty Percent Confidence

Pipeline:
  Agent 1 (Find It)  → Agent 2 (Rank It / PyMC)
                     → Agent 3 (Act On It / Geodo)
                     → Agent 4 (Explain It / Claude)

Memory layer: Cognee (every agent reads/writes via cognee.remember / cognee.recall)
"""

import asyncio
import os
from pathlib import Path

import cognee
from dotenv import load_dotenv

from agents.find_it import find_issues
from agents.rank_it import rank_issues
from agents.act_on_it import act_on_issues
from agents.explain_it import explain_findings
from utils.geodo_bridge import export_geodo_lookup

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


async def main():
    print("=" * 64)
    print("  VibeForward M-Agents  |  Track 01: Data Rescue")
    print("  Harven Manufacturing — 4-day pre-audit pipeline")
    print("=" * 64)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Cognee setup ───────────────────────────────────────────────────────────
    # Cognee reads LLM_API_KEY / LLM_PROVIDER from environment (.env).
    # Default: OpenAI gpt-4o-mini + text-embedding-3-small.
    # Override in .env with LLM_PROVIDER=anthropic etc. (see .env.example).
    await cognee.forget(everything=True)  # start fresh for this run
    # Use cognee.add (vector ingestion only) — zero LLM calls
    await cognee.add(
        "Starting Harven Manufacturing Data Rescue pipeline. "
        "Dataset: 5,000 warehouse records across Plants A, B, C. "
        "Regulatory audit in 4 days. Goal: identify and remediate all data-quality issues.",
        dataset_name="data_rescue",
    )

    # ── Agent 1 — Find It ──────────────────────────────────────────────────────
    print("\n[1/4] Agent 1 — FIND IT")
    issues = await find_issues()

    # Export Geodo lookup list immediately after finding orphaned customer IDs
    print("\n[Geodo] Exporting unknown customer IDs for manual validation …")
    export_geodo_lookup()

    # ── Agent 2 — Rank It (PyMC) ───────────────────────────────────────────────
    print("\n[2/4] Agent 2 — RANK IT  (PyMC Bayesian severity scoring)")
    ranked = await rank_issues(issues)

    # ── Agent 3 — Act On It ────────────────────────────────────────────────────
    print("\n[3/4] Agent 3 — ACT ON IT")
    print("      (If you ran Geodo research, save results to output/geodo_results.json first)")
    action_summary = await act_on_issues(ranked)

    # ── Agent 4 — Explain It (Claude) ─────────────────────────────────────────
    print("\n[4/4] Agent 4 — EXPLAIN IT  (Claude narrative via Cognee memory recall)")
    report_path = await explain_findings(action_summary)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Pipeline complete!")
    print(f"  Audit report  : {report_path}")
    print(f"  Cleaned data  : {OUTPUT_DIR}/track01_cleaned.csv")
    print(f"  Audit log     : {OUTPUT_DIR}/audit_log.json")
    print(f"  Geodo lookup  : {OUTPUT_DIR}/geodo_lookup_list.json")
    print("=" * 64)

    # Final Cognee memory (vector-only, no LLM)
    await cognee.add(
        f"Pipeline complete. Report: {report_path}. "
        f"Clean={action_summary['clean']} Flagged={action_summary['flagged']} "
        f"Escalated={action_summary['escalated']}.",
        dataset_name="data_rescue",
    )


if __name__ == "__main__":
    asyncio.run(main())
