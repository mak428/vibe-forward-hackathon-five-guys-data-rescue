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
from agents.recommend_it import recommend_actions
from agents.explain_it import explain_findings
from utils.geodo_bridge import export_geodo_lookup
from utils.schema_detect import detect_schema, save_schema

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


async def main():
    print("=" * 64)
    print("  VibeForward M-Agents  |  Track 01: Data Rescue")
    print("  Harven Manufacturing — 4-day pre-audit pipeline")
    print("=" * 64)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Agent 0 — Schema Discovery (dataset-agnostic) ─────────────────────────
    print("\n[0/5] Schema auto-discovery (works for any dataset)")
    _data_dir = Path(__file__).parent / "data"
    _csv_files = list(_data_dir.glob("*.csv"))
    if not _csv_files:
        raise FileNotFoundError(f"No CSV found in {_data_dir}")
    _primary_csv = next((f for f in _csv_files if "customer" not in f.name.lower()), _csv_files[0])
    _df_sample = __import__("pandas").read_csv(_primary_csv)
    schema = detect_schema(_df_sample)
    save_schema(schema)
    print(f"  Dataset    : {_primary_csv.name}  ({len(_df_sample):,} rows × {len(_df_sample.columns)} cols)")
    print(f"  Mapped     : record_id={schema.record_id}, part={schema.part_number}, "
          f"customer={schema.customer_id}, qty={schema.quantity}, weight={schema.weight}")
    print(f"  Dates      : production={schema.production_date}, ship={schema.ship_date}")
    print(f"  Unmapped   : {schema.unmapped}")

    # ── Cognee setup ───────────────────────────────────────────────────────────
    # Cognee reads LLM_API_KEY / LLM_PROVIDER from environment (.env).
    # Default: OpenAI gpt-4o-mini + text-embedding-3-small.
    # Override in .env with LLM_PROVIDER=anthropic etc. (see .env.example).
    await cognee.forget(everything=True)  # start fresh for this run
    # Use cognee.add (vector ingestion only) — zero LLM calls
    await cognee.add(
        f"Starting Data Rescue pipeline. "
        f"Dataset: {len(_df_sample):,} records detected. "
        f"Schema: {', '.join(f'{k}={v}' for k,v in {'record_id': schema.record_id, 'part': schema.part_number, 'customer': schema.customer_id}.items() if v)}. "
        f"Goal: detect all data-quality issues, rank by severity, remediate, and report.",
        dataset_name="data_rescue",
    )

    # ── Agent 1 — Find It ──────────────────────────────────────────────────────
    print("\n[1/5] Agent 1 — FIND IT  (6 issue classes, per-part statistical baselining)")
    issues = await find_issues()

    # Export Geodo lookup list immediately after finding orphaned customer IDs
    print("\n[Geodo] Exporting unknown entity IDs for manual validation …")
    export_geodo_lookup()

    # ── Agent 2 — Rank It (PyMC) ───────────────────────────────────────────────
    print("\n[2/5] Agent 2 — RANK IT  (PyMC Bayesian 3-method convergence)")
    ranked = await rank_issues(issues)

    # ── Agent 3 — Act On It ────────────────────────────────────────────────────
    print("\n[3/5] Agent 3 — ACT ON IT")
    print("      (Save Geodo results to output/geodo_results.json before this step for enrichment)")
    action_summary = await act_on_issues(ranked)

    # ── Agent 5 — Recommend It ────────────────────────────────────────────────
    print("\n[4/5] Agent 5 — RECOMMEND IT  (compliance officer decision guide)")
    recommendations = await recommend_actions(action_summary)

    # ── Agent 4 — Explain It ──────────────────────────────────────────────────
    print("\n[5/5] Agent 4 — EXPLAIN IT  (downloadable audit report via Cognee memory recall)")
    report_path = await explain_findings(action_summary, recommendations)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Pipeline complete!")
    print(f"  Audit report     : {report_path}")
    print(f"  Cleaned data     : {OUTPUT_DIR}/track01_cleaned.csv")
    print(f"  Audit log        : {OUTPUT_DIR}/audit_log.json")
    print(f"  Recommendations  : {OUTPUT_DIR}/recommendations.json")
    print(f"  Geodo lookup     : {OUTPUT_DIR}/geodo_lookup_list.json")
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
