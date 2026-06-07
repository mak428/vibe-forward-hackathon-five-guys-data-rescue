"""Agent 4 — Explain It: recall context from Cognee via vector search, then render
a fully deterministic HTML audit report — zero LLM API calls."""

import json
from datetime import datetime
from pathlib import Path

import cognee
from cognee.api.v1.search import SearchType

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def explain_findings(action_summary: dict | None = None) -> str:
    if action_summary is None:
        with open(OUTPUT_DIR / "audit_log.json") as f:
            action_summary = json.load(f)
    with open(OUTPUT_DIR / "findings.json") as f:
        findings = json.load(f)
    with open(OUTPUT_DIR / "rankings.json") as f:
        rankings = json.load(f)

    # ── Pull agent narratives from Cognee via vector search (no LLM) ─────────
    cognee_snippets: list[str] = []
    try:
        results = await cognee.search(
            "data quality issues actions severity Harven Manufacturing",
            SearchType.CHUNKS,
        )
        cognee_snippets = [str(r) for r in (results or [])][:6]
    except Exception:
        pass

    # ── Deterministic issue table ─────────────────────────────────────────────
    priority_colour = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#2980b9", "LOW": "#27ae60"}
    issue_rows = "".join(
        f"""<tr>
          <td><span style="color:{priority_colour.get(r['priority'],'#333')};font-weight:bold">{r['priority']}</span></td>
          <td>{r['type']}/{r['subtype']}</td>
          <td style="text-align:right">{r['count']:,}</td>
          <td style="text-align:right">{r['severity']['mean']:.2f}</td>
          <td>[{r['severity']['ci_low']:.2f} – {r['severity']['ci_high']:.2f}]</td>
        </tr>"""
        for r in rankings
    )

    # ── Deterministic action table ────────────────────────────────────────────
    action_colour = {"AUTO_FIXED": "#27ae60", "FLAGGED": "#e67e22", "ESCALATED": "#c0392b"}
    action_rows = "".join(
        f"""<tr>
          <td><span style="color:{action_colour.get(a['action'],'#333')};font-weight:bold">{a['action']}</span></td>
          <td>{a['issue']}</td>
          <td style="text-align:right">{a['records_affected']:,}</td>
          <td style="font-size:0.85em;color:#555">{a['justification'][:120]}…</td>
        </tr>"""
        for a in action_summary["audit_log"]
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if action_summary["escalated"] > 0:
        verdict, verdict_cls, verdict_reason = (
            "RED",
            "red",
            f"{action_summary['escalated']} records reference customer IDs that cannot be verified. "
            "Auditors will flag these as missing entity validation — manual resolution required before audit.",
        )
    elif action_summary["flagged"] > 10:
        verdict, verdict_cls, verdict_reason = (
            "YELLOW",
            "yellow",
            f"{action_summary['flagged']} records are flagged for human review. "
            "Data is usable but some anomalies remain unresolved.",
        )
    else:
        verdict, verdict_cls, verdict_reason = (
            "GREEN",
            "green",
            "All critical issues resolved. Dataset is audit-ready.",
        )

    # ── Cognee memory snippets sidebar ────────────────────────────────────────
    cognee_html = ""
    if cognee_snippets:
        items = "".join(f"<li>{s[:200]}</li>" for s in cognee_snippets)
        cognee_html = f"""
<h2>Cognee Memory — Agent Context (vector recall)</h2>
<ul style="font-size:0.85em;color:#555;line-height:1.7">{items}</ul>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Harven Manufacturing — Audit Readiness Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 980px; margin: 40px auto; padding: 0 24px; color: #222; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }}
  h2   {{ color: #16213e; margin-top: 32px; }}
  p    {{ line-height: 1.6; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.9em; }}
  th   {{ background: #0f3460; color: #fff; padding: 9px 12px; text-align: left; }}
  td   {{ border: 1px solid #ddd; padding: 8px 12px; vertical-align: top; }}
  tr:nth-child(even) td {{ background: #f5f7fa; }}
  .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .stat-card {{ background: #f0f4ff; border-radius: 8px; padding: 16px 24px; flex: 1; min-width: 140px; text-align:center; }}
  .stat-num  {{ font-size: 2.2em; font-weight: bold; color: #0f3460; }}
  .stat-lbl  {{ font-size: 0.82em; color: #666; margin-top: 4px; }}
  .verdict   {{ border-radius: 8px; padding: 20px 24px; margin: 24px 0; color: #fff; }}
  .red       {{ background: #c0392b; }}
  .yellow    {{ background: #e67e22; }}
  .green     {{ background: #27ae60; }}
  .verdict h2 {{ color: #fff; border:none; margin-top:0; }}
  .pipeline  {{ background: #eef2ff; border-left: 4px solid #4361ee; padding: 14px 18px;
                border-radius: 4px; font-size: 0.87em; color: #444; margin-top: 40px; }}
</style>
</head>
<body>

<h1>Harven Manufacturing — Audit Readiness Report</h1>
<p>
  <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  <strong>Dataset:</strong> track01_data_rescue.csv &nbsp;|&nbsp;
  <strong>Records in:</strong> {action_summary['total_input_records']:,} &nbsp;|&nbsp;
  <strong>Records out:</strong> {action_summary['output_records']:,}
</p>

<div class="stat-grid">
  <div class="stat-card"><div class="stat-num">{action_summary['total_input_records']:,}</div><div class="stat-lbl">Input records</div></div>
  <div class="stat-card"><div class="stat-num">{action_summary['duplicates_removed']}</div><div class="stat-lbl">Duplicates removed</div></div>
  <div class="stat-card"><div class="stat-num" style="color:#27ae60">{action_summary['clean']:,}</div><div class="stat-lbl">Audit-ready (CLEAN)</div></div>
  <div class="stat-card"><div class="stat-num" style="color:#e67e22">{action_summary['flagged']}</div><div class="stat-lbl">Need review (FLAGGED)</div></div>
  <div class="stat-card"><div class="stat-num" style="color:#c0392b">{action_summary['escalated']}</div><div class="stat-lbl">Must resolve (ESCALATED)</div></div>
</div>

<h2>Executive Summary</h2>
<p>
  Harven Manufacturing's warehouse dataset contained <strong>{sum(i['count'] for i in findings):,} data-quality issues
  across {len(findings)} categories</strong> in {action_summary['total_input_records']:,} records from Plants A, B, and C.
  The pipeline automatically resolved {sum(1 for a in action_summary['audit_log'] if a['action'] == 'AUTO_FIXED')} issue
  types ({action_summary['duplicates_removed'] + sum(a['records_affected'] for a in action_summary['audit_log'] if a['action'] == 'AUTO_FIXED' and 'duplicate' not in a['issue']):,} records fixed),
  leaving <strong>{action_summary['clean']:,} clean, audit-ready records</strong>.
  <strong>{action_summary['escalated']} records</strong> with unverified customer IDs must be resolved before the audit.
</p>

<h2>What Was Fixed Automatically</h2>
<p>The following issues were corrected without human intervention:</p>
<ul>
  {"".join(f"<li><strong>{a['issue']}</strong> ({a['records_affected']:,} records) — {a['justification'][:180]}</li>" for a in action_summary['audit_log'] if a['action'] == 'AUTO_FIXED')}
</ul>

<h2>What Needs Human Review Before the Audit (FLAGGED)</h2>
<p>These records have anomalies that cannot be auto-corrected. A data steward must review each one:</p>
<table>
  <tr><th>Issue</th><th>Records</th><th>Why It Matters</th></tr>
  {"".join(f"<tr><td>{a['issue']}</td><td style='text-align:right'>{a['records_affected']:,}</td><td>{a['justification'][:200]}</td></tr>" for a in action_summary['audit_log'] if a['action'] == 'FLAGGED')}
</table>

<h2>What Must Be Resolved or the Audit Will Fail (ESCALATED)</h2>
{"".join(f"<p><strong>{a['issue']}</strong> — {a['justification']}</p>" for a in action_summary['audit_log'] if a['action'] == 'ESCALATED') or "<p>None.</p>"}

<h2>Issue Severity Rankings (PyMC Bayesian Beta Posteriors)</h2>
<table>
  <tr><th>Priority</th><th>Issue Type</th><th>Count</th><th>Severity</th><th>95% CI</th></tr>
  {issue_rows}
</table>

<h2>Full Action Log</h2>
<table>
  <tr><th>Action</th><th>Issue</th><th>Records</th><th>Justification</th></tr>
  {action_rows}
</table>

{cognee_html}

<div class="verdict {verdict_cls}">
  <h2>Audit Readiness Verdict: {verdict}</h2>
  <p>{verdict_reason}</p>
</div>

<div class="pipeline">
  <strong>Pipeline:</strong>
  Agent 1 — Find It &nbsp;→&nbsp;
  Agent 2 — Rank It (PyMC Bayesian severity · dhub skill: pymc-labs/pymc-modeling) &nbsp;→&nbsp;
  Agent 3 — Act On It (Geodo entity validation) &nbsp;→&nbsp;
  Agent 4 — Explain It (deterministic · zero LLM calls) &nbsp;|&nbsp;
  Memory layer: <strong>Cognee</strong> (vector search · add/search API)
  &nbsp;|&nbsp; LLM API calls: <strong>0</strong>
</div>

</body>
</html>"""

    report_path = OUTPUT_DIR / "audit_report.html"
    with open(report_path, "w") as f:
        f.write(html)

    try:
        await cognee.add(
            f"Agent 4 (Explain It): Audit report generated. Verdict: {verdict}. "
            f"Report saved to {report_path}.",
            dataset_name="data_rescue",
        )
    except Exception:
        pass

    print(f"[Agent 4 — Explain It] Report written → {report_path}  (verdict: {verdict})")
    return str(report_path)
