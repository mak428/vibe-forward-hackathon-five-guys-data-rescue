"""Agent 4 — Explain It: Robinhood-style dark dashboard with embedded compliance chatbot.

R07: full decision transparency in every view.
R08: HTML report + CSV downloadable directly from product interface.
"""

import base64
import json
from datetime import datetime
from pathlib import Path

import cognee
from cognee.api.v1.search import SearchType

OUTPUT_DIR = Path(__file__).parent.parent / "output"


async def explain_findings(action_summary: dict | None = None, recommendations: list[dict] | None = None) -> str:
    if action_summary is None:
        with open(OUTPUT_DIR / "audit_log.json") as f:
            action_summary = json.load(f)
    if recommendations is None:
        rec_path = OUTPUT_DIR / "recommendations.json"
        if rec_path.exists():
            with open(rec_path) as f:
                recommendations = json.load(f).get("recommendations", [])
        else:
            recommendations = []

    with open(OUTPUT_DIR / "findings.json") as f:
        findings = json.load(f)
    with open(OUTPUT_DIR / "rankings.json") as f:
        rankings = json.load(f)

    # ── Cognee recall ─────────────────────────────────────────────────────────
    cognee_snippets: list[str] = []
    try:
        results = await cognee.search(
            "Harven Manufacturing data quality issues severity actions pipeline",
            SearchType.CHUNKS,
        )
        cognee_snippets = [str(r) for r in (results or [])][:8]
        print(f"[Agent 4 — Explain It] Cognee recall: {len(cognee_snippets)} chunks")
    except Exception as e:
        print(f"[Agent 4 — Explain It] Cognee recall warning: {e}")

    # ── Kaggle benchmark ──────────────────────────────────────────────────────
    from utils.benchmark import score_benchmark
    benchmark = score_benchmark()

    # ── Downloadable CSV as base64 data URI ───────────────────────────────────
    cleaned_csv_path = OUTPUT_DIR / "track01_cleaned.csv"
    csv_data_uri = ""
    if cleaned_csv_path.exists():
        csv_b64 = base64.b64encode(cleaned_csv_path.read_bytes()).decode()
        csv_data_uri = f"data:text/csv;base64,{csv_b64}"

    # ── Verdict ───────────────────────────────────────────────────────────────
    if action_summary["escalated"] > 0:
        verdict, verdict_color = "RED", "#ff5c5c"
        verdict_reason = (
            f"{action_summary['escalated']} records with unverified entity IDs. "
            "Complete Geodo research before the audit."
        )
    elif action_summary["flagged"] > 10:
        verdict, verdict_color = "YELLOW", "#ffb347"
        verdict_reason = f"{action_summary['flagged']} records need human review."
    else:
        verdict, verdict_color = "GREEN", "#00c87b"
        verdict_reason = "All critical issues resolved. Dataset is audit-ready."

    # ── Pre-compute HTML fragments ────────────────────────────────────────────
    total_rec_effort = sum(r["estimated_effort_hours"] for r in recommendations)

    priority_color = {"CRITICAL": "#ff5c5c", "HIGH": "#ffb347", "MEDIUM": "#4a90e2", "LOW": "#00c87b"}
    action_color   = {"AUTO_FIXED": "#00c87b", "FLAGGED": "#ffb347", "ESCALATED": "#ff5c5c"}

    # Issues tab rows
    issue_rows = "".join(
        f"""<tr onclick="showDetail(this)" data-detail="{rankings[i].get('ranking_reasoning','')[:600].replace('"','&quot;')}" style="cursor:pointer">
          <td><span class="badge" style="background:{priority_color.get(rankings[i]['priority'],'#888')}">{rankings[i]['priority']}</span>
          <span style="font-size:0.72em;color:#6b7280;margin-left:4px">{rankings[i].get('confidence','?')}</span></td>
          <td style="color:#e5e7eb">{rankings[i]['type']}/{rankings[i]['subtype']}</td>
          <td style="text-align:right;font-weight:600;color:#e5e7eb">{rankings[i]['count']:,}</td>
          <td style="text-align:right"><span style="color:{priority_color.get(rankings[i]['priority'],'#888')};font-weight:700">{rankings[i]['severity']['ensemble_score']:.2f}</span></td>
        </tr>"""
        for i in range(len(rankings))
    )

    # Decision log rows
    decision_rows = "".join(
        f"""<tr>
          <td><span class="badge" style="background:{action_color.get(a['action'],'#888')}">{a['action']}</span></td>
          <td style="color:#e5e7eb">{a['issue']}</td>
          <td style="text-align:right;color:#e5e7eb">{a['records_affected']:,}</td>
          <td style="color:#9ca3af;font-size:0.82em">{a['justification']}</td>
        </tr>"""
        for a in action_summary["audit_log"]
    )

    # Benchmark rows
    bench_rows = "".join(
        f"<tr><td style='color:#e5e7eb'>{b['category']}</td>"
        f"<td style='text-align:right;color:#00c87b;font-weight:600'>{b['our_count']:,}</td>"
        f"<td style='color:#9ca3af;font-size:0.82em'>{b['description'][:90]}</td></tr>"
        for b in benchmark.get("by_category", [])
    )

    # Cognee snippets list
    cognee_items = "".join(
        f"<li style='margin:6px 0;color:#9ca3af;font-size:0.83em;border-left:2px solid #2d3748;padding-left:10px'>{s[:200]}</li>"
        for s in cognee_snippets
    ) or "<li style='color:#6b7280'>Run pipeline to populate Cognee memory</li>"

    # Recommendation cards
    priority_order = {"DO_NOW": 0, "DO_TODAY": 1, "DO_THIS_WEEK": 2, "OPTIONAL": 3}
    rec_left_color = {"DO_NOW": "#ff5c5c", "DO_TODAY": "#ffb347", "DO_THIS_WEEK": "#4a90e2", "OPTIONAL": "#6b7280"}
    rec_badge_bg   = {"DO_NOW": "#ff5c5c", "DO_TODAY": "#ff9f43", "DO_THIS_WEEK": "#4a90e2", "OPTIONAL": "#6b7280"}
    rec_cards = ""
    for rec in sorted(recommendations, key=lambda r: (priority_order.get(r["priority"], 3), -r["priority_score"])):
        steps_html = "".join(f"<li style='margin:4px 0;font-size:0.85em;color:#9ca3af'>{s}</li>" for s in rec["recommended_steps"])
        lc = rec_left_color.get(rec["priority"], "#444")
        bb = rec_badge_bg.get(rec["priority"], "#444")
        rec_cards += f"""<div style="background:#1c1e2e;border-radius:10px;padding:16px 20px;margin:10px 0;border-left:4px solid {lc}">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span class="badge" style="background:{bb}">{rec['priority'].replace('_',' ')}</span>
    <strong style="color:#e5e7eb">{rec['issue']}</strong>
    <span style="color:#6b7280;font-size:0.83em">{rec['records_affected']:,} records · ~{rec['estimated_effort_hours']}h · {rec['owner']}</span>
  </div>
  <p style="color:#9ca3af;font-size:0.84em;margin:8px 0">{rec['reasoning']}</p>
  <ol style="margin:0;padding-left:18px">{steps_html}</ol>
</div>"""

    # Chart bar data (max 300px wide)
    max_count = max((i["count"] for i in findings), default=1)
    chart_bars = "".join(
        f"""<div style="display:flex;align-items:center;gap:10px;margin:7px 0">
          <div style="width:160px;font-size:0.78em;color:#9ca3af;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
               title="{i['type']}/{i['subtype']}">{i['subtype'].replace('_',' ')}</div>
          <div style="flex:1;background:#2d3748;border-radius:4px;height:22px;position:relative">
            <div style="background:{priority_color.get(rankings[j]['priority'] if j < len(rankings) else 'MEDIUM','#4a90e2')};
                        width:{min(i['count']/max_count*100,100):.0f}%;height:100%;border-radius:4px;
                        transition:width 1s ease"></div>
          </div>
          <div style="width:55px;text-align:right;font-size:0.82em;font-weight:600;color:#e5e7eb">{i['count']:,}</div>
        </div>"""
        for j, i in enumerate(
            sorted(findings, key=lambda x: x["count"], reverse=True)
        )
    )

    # Embedded JSON data for chatbot
    embedded_data = json.dumps({
        "findings": findings,
        "rankings": rankings,
        "audit_log": action_summary["audit_log"],
        "recommendations": recommendations,
        "cognee_snippets": cognee_snippets,
        "summary": {
            "total": action_summary["total_input_records"],
            "clean": action_summary["clean"],
            "flagged": action_summary["flagged"],
            "escalated": action_summary["escalated"],
            "verdict": verdict,
        },
    })

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataRescue — Audit Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body   {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117; color: #e5e7eb; min-height: 100vh; }}
  /* Top nav */
  .topbar {{ background: #111827; border-bottom: 1px solid #1f2937; padding: 0 24px;
             display: flex; align-items: center; justify-content: space-between; height: 56px; position:sticky; top:0; z-index:100; }}
  .logo  {{ font-size: 1.1em; font-weight: 700; color: #00c87b; letter-spacing:-0.3px; }}
  .logo span {{ color:#e5e7eb; }}
  .gen-time {{ font-size:0.76em; color:#6b7280; }}
  /* Nav tabs */
  .tabs  {{ display: flex; gap: 2px; background: #111827; padding: 0 24px; border-bottom: 1px solid #1f2937; }}
  .tab   {{ padding: 14px 20px; cursor: pointer; font-size: 0.88em; font-weight: 500;
            color: #6b7280; border-bottom: 2px solid transparent; transition: all .15s; }}
  .tab:hover {{ color: #e5e7eb; }}
  .tab.active {{ color: #00c87b; border-bottom-color: #00c87b; }}
  /* Pages */
  .page  {{ display: none; max-width: 1100px; margin: 0 auto; padding: 28px 24px; }}
  .page.active {{ display: block; }}
  /* Stat cards */
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 28px; }}
  .stat-card {{ background: #1c1e2e; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #2d3748; }}
  .stat-num  {{ font-size: 2.4em; font-weight: 700; line-height: 1; }}
  .stat-lbl  {{ font-size: 0.76em; color: #6b7280; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
  /* Verdict banner */
  .verdict-banner {{ border-radius: 12px; padding: 18px 24px; margin-bottom: 24px;
                     display: flex; align-items: center; gap: 16px; }}
  .verdict-dot {{ width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }}
  /* Tables */
  .card  {{ background: #1c1e2e; border-radius: 12px; border: 1px solid #2d3748; overflow: hidden; margin-bottom: 20px; }}
  .card-header {{ padding: 14px 20px; border-bottom: 1px solid #2d3748; display:flex; align-items:center; justify-content:space-between; }}
  .card-title {{ font-size: 0.9em; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.6px; }}
  table  {{ width: 100%; border-collapse: collapse; }}
  th     {{ background: #111827; padding: 10px 16px; text-align: left; font-size: 0.78em;
            font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
  td     {{ padding: 11px 16px; border-top: 1px solid #1f2937; vertical-align: top; font-size:0.88em; }}
  tr:hover td {{ background: #1f2937; }}
  /* Detail row */
  .detail-row td {{ background: #141622; color: #9ca3af; font-size: 0.82em; padding: 10px 16px; font-style: italic; }}
  /* Badge */
  .badge {{ display: inline-block; padding: 3px 9px; border-radius: 20px; font-size: 0.73em;
            font-weight: 700; color: #fff; letter-spacing: 0.3px; }}
  /* Download bar */
  .dl-bar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
  .dl-btn {{ display: inline-flex; align-items: center; gap: 6px; padding: 10px 18px;
             border-radius: 8px; font-weight: 600; font-size: 0.88em; text-decoration: none;
             color: #fff; cursor: pointer; border: none; transition: opacity .15s; }}
  .dl-btn:hover {{ opacity: 0.85; }}
  /* Chatbot */
  #chat-toggle {{ position:fixed; bottom:24px; right:24px; z-index:999;
                  background:#00c87b; color:#fff; border:none; border-radius:50px;
                  padding:12px 20px; font-size:0.9em; font-weight:700; cursor:pointer;
                  box-shadow:0 4px 20px rgba(0,200,123,.35); }}
  #chat-wrap  {{ position:fixed; bottom:76px; right:24px; z-index:998; width:370px;
                 background:#111827; border-radius:14px; border:1px solid #2d3748;
                 box-shadow:0 8px 32px rgba(0,0,0,.5); display:flex; flex-direction:column; }}
  #chat-wrap.hidden {{ display:none; }}
  .chat-hdr   {{ background:#0d1117; padding:14px 18px; border-radius:14px 14px 0 0;
                 border-bottom:1px solid #2d3748; }}
  .chat-hdr strong {{ color:#00c87b; font-size:0.92em; }}
  .chat-hdr p {{ color:#6b7280; font-size:0.76em; margin-top:2px; }}
  #chat-msgs  {{ height:320px; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:8px; }}
  .msg-bot {{ background:#1c1e2e; padding:10px 13px; border-radius:10px 10px 10px 2px;
              font-size:0.84em; color:#d1d5db; line-height:1.5; max-width:92%; }}
  .msg-user {{ background:#00c87b22; border:1px solid #00c87b44; padding:9px 13px;
               border-radius:10px 10px 2px 10px; font-size:0.84em; color:#d1d5db;
               max-width:88%; align-self:flex-end; }}
  #chat-input-row {{ display:flex; border-top:1px solid #2d3748; }}
  #chat-input {{ flex:1; background:transparent; border:none; padding:12px 14px;
                 color:#e5e7eb; font-size:0.88em; outline:none; }}
  #chat-input::placeholder {{ color:#4b5563; }}
  #chat-send {{ background:#00c87b; color:#fff; border:none; padding:12px 16px;
                cursor:pointer; font-weight:700; border-radius:0 0 14px 0; font-size:0.9em; }}
  /* Responsive */
  @media(max-width:600px) {{ .stat-num {{ font-size:1.8em; }} #chat-wrap {{ width:94vw; right:3vw; }} }}
</style>
</head>
<body>

<nav class="topbar">
  <div class="logo">DataRescue <span>/ Audit Dashboard</span></div>
  <div class="gen-time">Generated {generated_at} · {action_summary['total_input_records']:,} records</div>
</nav>

<div class="tabs">
  <div class="tab active" onclick="showTab('dashboard',this)">Dashboard</div>
  <div class="tab" onclick="showTab('issues',this)">Issues</div>
  <div class="tab" onclick="showTab('recommendations',this)">Action Plan</div>
  <div class="tab" onclick="showTab('decisions',this)">Decision Log</div>
  <div class="tab" onclick="showTab('benchmark',this)">Benchmark</div>
  <div class="tab" onclick="showTab('memory',this)">Cognee Memory</div>
</div>

<!-- ═══ DASHBOARD ═══════════════════════════════════════════════════════════ -->
<div class="page active" id="page-dashboard">

  <div class="verdict-banner" style="background:{verdict_color}18;border:1px solid {verdict_color}44">
    <div class="verdict-dot" style="background:{verdict_color}"></div>
    <div>
      <strong style="color:{verdict_color};font-size:1.1em">Audit Verdict: {verdict}</strong>
      <p style="color:#9ca3af;font-size:0.88em;margin-top:3px">{verdict_reason}</p>
    </div>
  </div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-num" style="color:#e5e7eb">{action_summary['total_input_records']:,}</div>
      <div class="stat-lbl">Total Records</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#00c87b">{action_summary['clean']:,}</div>
      <div class="stat-lbl">Clean / Audit-Ready</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#ffb347">{action_summary['flagged']}</div>
      <div class="stat-lbl">Flagged (Review)</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#ff5c5c">{action_summary['escalated']}</div>
      <div class="stat-lbl">Escalated (Block)</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#a78bfa">{action_summary['duplicates_removed']}</div>
      <div class="stat-lbl">Duplicates Removed</div>
    </div>
  </div>

  <div class="dl-bar">
    <a class="dl-btn" style="background:#1c1e2e;border:1px solid #2d3748"
       href="audit_report.html" download="audit_report.html">⬇ Download Report</a>
    {'<a class="dl-btn" style="background:#00c87b22;border:1px solid #00c87b44;color:#00c87b" href="' + csv_data_uri + '" download="track01_cleaned.csv">⬇ Clean Dataset (CSV)</a>' if csv_data_uri else ''}
    <a class="dl-btn" style="background:#4a90e222;border:1px solid #4a90e244;color:#4a90e2"
       href="audit_log.json" download="audit_log.json">⬇ Audit Log (JSON)</a>
  </div>

  <!-- Issue chart -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Issue Distribution</span>
      <span style="font-size:0.8em;color:#6b7280">{sum(i['count'] for i in findings):,} total affected records · {len(findings)} categories</span>
    </div>
    <div style="padding:20px 24px">
      {chart_bars}
    </div>
  </div>

  <!-- Quick stats row -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div class="card">
      <div class="card-header"><span class="card-title">Benchmark</span></div>
      <div style="padding:16px 20px">
        <div style="font-size:2em;font-weight:700;color:#00c87b">{benchmark.get('estimated_recall',0):.0%}</div>
        <div style="color:#6b7280;font-size:0.8em;margin-top:4px">Estimated Recall vs Kaggle seeded issues</div>
        <div style="margin-top:10px;color:#9ca3af;font-size:0.84em">
          {benchmark.get('categories_found',0)}/{benchmark.get('categories_expected',0)} categories detected
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">Action Plan</span></div>
      <div style="padding:16px 20px">
        <div style="font-size:2em;font-weight:700;color:#ffb347">{total_rec_effort:.0f}h</div>
        <div style="color:#6b7280;font-size:0.8em;margin-top:4px">Estimated work before audit</div>
        <div style="margin-top:10px;color:#9ca3af;font-size:0.84em">
          {len(recommendations)} action items · 4 days remaining
        </div>
      </div>
    </div>
  </div>

  <div style="margin-top:16px;padding:14px 18px;background:#1c1e2e;border-radius:10px;border:1px solid #2d3748;font-size:0.82em;color:#6b7280">
    <strong style="color:#9ca3af">Stack:</strong>
    Cognee (memory · add/search · all agents) ·
    PyMC v5 (Bayesian Beta posteriors) ·
    Decision-Lab (3-method convergence) ·
    Geodo (entity validation) ·
    Trupeer (demo video) ·
    <strong style="color:#00c87b">LLM API calls: 0</strong>
  </div>
</div>

<!-- ═══ ISSUES ═══════════════════════════════════════════════════════════════ -->
<div class="page" id="page-issues">
  <h2 style="color:#e5e7eb;font-size:1.1em;margin-bottom:16px">Issue Severity Rankings — 3-Method Convergence</h2>
  <p style="color:#6b7280;font-size:0.84em;margin-bottom:16px">
    Every score uses PyMC Bayesian + Frequency×Weight + Regulatory heuristic.
    Click any row to see full reasoning.
  </p>
  <div class="card">
    <table>
      <thead><tr>
        <th>Priority</th><th>Issue Type</th><th style="text-align:right">Records</th>
        <th style="text-align:right">Score</th>
      </tr></thead>
      <tbody id="issue-tbody">{issue_rows}</tbody>
    </table>
  </div>
  <div id="issue-detail" style="display:none;background:#141622;border-radius:10px;padding:16px 20px;
    margin-top:12px;font-size:0.84em;color:#9ca3af;line-height:1.6;border:1px solid #2d3748">
  </div>
</div>

<!-- ═══ RECOMMENDATIONS ═══════════════════════════════════════════════════════ -->
<div class="page" id="page-recommendations">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:10px">
    <h2 style="color:#e5e7eb;font-size:1.1em">Compliance Officer Action Plan</h2>
    <span style="color:#6b7280;font-size:0.84em">{len(recommendations)} items · ~{total_rec_effort:.0f}h total · 4 days to audit</span>
  </div>
  {rec_cards if rec_cards else '<div style="color:#6b7280;padding:20px">No pending actions — dataset is audit-ready.</div>'}
</div>

<!-- ═══ DECISION LOG ═══════════════════════════════════════════════════════════ -->
<div class="page" id="page-decisions">
  <h2 style="color:#e5e7eb;font-size:1.1em;margin-bottom:16px">Full Decision Log (R07)</h2>
  <div class="card">
    <table>
      <thead><tr><th>Action</th><th>Issue</th><th style="text-align:right">Records</th><th>Justification</th></tr></thead>
      <tbody>{decision_rows}</tbody>
    </table>
  </div>
</div>

<!-- ═══ BENCHMARK ═════════════════════════════════════════════════════════════ -->
<div class="page" id="page-benchmark">
  <h2 style="color:#e5e7eb;font-size:1.1em;margin-bottom:8px">Kaggle Benchmark Verification</h2>
  <p style="color:#6b7280;font-size:0.84em;margin-bottom:16px">{benchmark.get('note','')}</p>
  <div class="stat-grid" style="max-width:600px">
    <div class="stat-card">
      <div class="stat-num" style="color:#00c87b">{benchmark.get('estimated_recall',0):.0%}</div>
      <div class="stat-lbl">Est. Recall</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#4a90e2">{benchmark.get('estimated_precision',0):.0%}</div>
      <div class="stat-lbl">Est. Precision</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" style="color:#a78bfa">{benchmark.get('categories_found',0)}/{benchmark.get('categories_expected',0)}</div>
      <div class="stat-lbl">Categories Found</div>
    </div>
  </div>
  <div class="card" style="margin-top:16px">
    <table>
      <thead><tr><th>Category</th><th style="text-align:right">Our Count</th><th>Description</th></tr></thead>
      <tbody>{bench_rows}</tbody>
    </table>
  </div>
</div>

<!-- ═══ COGNEE MEMORY ══════════════════════════════════════════════════════════ -->
<div class="page" id="page-memory">
  <h2 style="color:#e5e7eb;font-size:1.1em;margin-bottom:8px">Cognee Memory — Inter-Agent Context</h2>
  <p style="color:#6b7280;font-size:0.84em;margin-bottom:16px">
    {len(cognee_snippets)} chunks recalled via vector search. Each agent reads prior agents' writes.
  </p>
  <ul style="list-style:none;padding:0">{cognee_items}</ul>
</div>

<!-- ═══ CHATBOT ════════════════════════════════════════════════════════════════ -->
<script id="pipeline-data" type="application/json">{embedded_data}</script>

<button id="chat-toggle" onclick="toggleChat()">Ask Compliance Assistant</button>
<div id="chat-wrap">
  <div class="chat-hdr">
    <strong>Compliance Assistant</strong>
    <p>Ask about any flagged record, issue, or what to do next</p>
  </div>
  <div id="chat-msgs">
    <div class="msg-bot">Hi! I can explain any flagged record or issue.
    Try: <em>"Why were records escalated?"</em>, <em>"What is weight inflation?"</em>,
    <em>"What should I do first?"</em>, or <em>"Show summary"</em>.</div>
  </div>
  <div id="chat-input-row">
    <input id="chat-input" type="text" placeholder="Ask a question…"
           onkeydown="if(event.key==='Enter')sendMsg()"/>
    <button id="chat-send" onclick="sendMsg()">↑</button>
  </div>
</div>

<script>
(function() {{
  const D = JSON.parse(document.getElementById('pipeline-data').textContent);
  const issueExplain = {{}};
  D.findings.forEach(f => {{
    issueExplain[f.subtype.toLowerCase()] = f.description;
    issueExplain[f.type.toLowerCase()] = f.description;
  }});

  function reply(text) {{
    const msgs = document.getElementById('chat-msgs');
    const div = document.createElement('div');
    div.className = 'msg-bot'; div.innerHTML = text;
    msgs.appendChild(div); msgs.scrollTop = msgs.scrollHeight;
  }}
  function addUser(text) {{
    const msgs = document.getElementById('chat-msgs');
    const div = document.createElement('div');
    div.className = 'msg-user'; div.textContent = text;
    msgs.appendChild(div); msgs.scrollTop = msgs.scrollHeight;
  }}

  function answer(q) {{
    const lq = q.toLowerCase();
    if (/summary|overview|verdict|total|stat/.test(lq)) {{
      const s = D.summary;
      return `<strong>Summary:</strong> ${{s.total.toLocaleString()}} records &nbsp;·&nbsp;
        <span style="color:#00c87b">✓ ${{s.clean.toLocaleString()}} clean</span> &nbsp;·&nbsp;
        <span style="color:#ffb347">⚠ ${{s.flagged}} flagged</span> &nbsp;·&nbsp;
        <span style="color:#ff5c5c">✕ ${{s.escalated}} escalated</span><br>
        Verdict: <strong style="color:${{s.verdict==='GREEN'?'#00c87b':s.verdict==='RED'?'#ff5c5c':'#ffb347'}}">${{s.verdict}}</strong>`;
    }}
    if (/escalat/.test(lq)) {{
      const esc = D.audit_log.filter(a=>a.action==='ESCALATED');
      if (!esc.length) return 'No escalated records — great!';
      const rec = D.recommendations.find(r=>r.action_type==='ESCALATED');
      const steps = rec ? '<br><strong>Steps:</strong> ' + rec.recommended_steps.slice(0,2).join(' → ') : '';
      return esc.map(a=>`<strong style="color:#ff5c5c">ESCALATED: ${{a.issue}}</strong><br>${{a.justification}}`).join('<br><br>') + steps;
    }}
    if (/flag/.test(lq)) {{
      const fl = D.audit_log.filter(a=>a.action==='FLAGGED');
      if (!fl.length) return 'No flagged records.';
      return '<strong>Flagged:</strong><br>' + fl.map(a=>`• ${{a.issue}} (${{a.records_affected}} records) — ${{a.justification}}`).join('<br>');
    }}
    if (/recommend|what.*(do|should|next|fix)|step|action|plan/.test(lq)) {{
      const matched = D.recommendations.filter(r=>
        lq.includes(r.issue.split('/').pop().toLowerCase().replace(/_/g,' ')) || r.action_type==='ESCALATED'
      );
      const show = matched.length ? matched.slice(0,2) : D.recommendations.slice(0,2);
      if (!show.length) return 'No pending recommendations.';
      return show.map(r=>
        `<strong style="color:${{r.priority==='DO_NOW'?'#ff5c5c':'#ffb347'}}">${{r.priority.replace(/_/g,' ')}}</strong> — ${{r.issue}}<br>
         1. ${{r.recommended_steps[0]}}<br><span style="color:#6b7280">Owner: ${{r.owner}} · ~${{r.estimated_effort_hours}}h</span>`
      ).join('<br><br>');
    }}
    for (const k of Object.keys(issueExplain)) {{
      if (lq.includes(k.replace(/_/g,' '))||lq.includes(k)) {{
        const rank = D.rankings.find(r=>r.subtype===k||r.type===k);
        const ri = rank ? `<br><strong>Severity:</strong> ${{rank.priority}} (${{rank.severity.ensemble_score.toFixed(2)}})
          <br><span style="font-size:0.85em;color:#9ca3af">${{rank.ranking_reasoning.substring(0,250)}}…</span>` : '';
        return `<strong>${{k.replace(/_/g,' ')}}</strong><br>${{issueExplain[k]}}${{ri}}`;
      }}
    }}
    if (/rank|severit|priorit|critical|high/.test(lq)) {{
      return '<strong>Top issues:</strong><br>' + D.rankings.slice(0,4).map(r=>
        `<span style="color:${{r.priority==='CRITICAL'?'#ff5c5c':r.priority==='HIGH'?'#ffb347':'#4a90e2'}}">${{r.priority}}</span>
         ${{r.type}}/${{r.subtype}} (n=${{r.count}}, score=${{r.severity.ensemble_score.toFixed(2)}})`
      ).join('<br>');
    }}
    if (/auto.?fix|fixed|clean/.test(lq)) {{
      const f = D.audit_log.filter(a=>a.action==='AUTO_FIXED');
      return '<strong>Auto-fixed:</strong><br>' + f.map(a=>`• ${{a.issue}} (${{a.records_affected}} records) — ${{a.justification}}`).join('<br>');
    }}
    if (/cognee|memory|agent/.test(lq)) {{
      return '<strong>Cognee memory (inter-agent):</strong><br>' +
        D.cognee_snippets.slice(0,3).map(s=>`• ${{s.substring(0,180)}}`).join('<br>');
    }}
    return `No match for "<em>${{q}}</em>".<br>Try: <em>summary</em> · <em>escalated</em> · <em>flagged</em> ·
      <em>what is weight inflation</em> · <em>what should I do first</em> · <em>rankings</em>`;
  }}

  window.sendMsg = function() {{
    const inp = document.getElementById('chat-input');
    const q = inp.value.trim(); if (!q) return;
    addUser(q); inp.value = '';
    setTimeout(()=>reply(answer(q)), 160);
  }};
  window.toggleChat = function() {{
    document.getElementById('chat-wrap').classList.toggle('hidden');
  }};

  // Tab navigation
  window.showTab = function(name, el) {{
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.getElementById('page-'+name).classList.add('active');
    el.classList.add('active');
  }};

  // Issue detail row expand
  window.showDetail = function(row) {{
    const detail = document.getElementById('issue-detail');
    const text = row.dataset.detail;
    if (detail.style.display==='block' && detail.dataset.src===row.rowIndex?.toString()) {{
      detail.style.display='none'; return;
    }}
    detail.innerHTML = '<strong style="color:#a78bfa">Full reasoning:</strong><br>' + text.replace(/\|/g,'<br>');
    detail.style.display='block';
    detail.dataset.src = row.rowIndex?.toString();
  }};
}})();
</script>
</body>
</html>"""

    report_path = OUTPUT_DIR / "audit_report.html"
    with open(report_path, "w") as f:
        f.write(html)

    try:
        await cognee.add(
            f"Agent 4 (Explain It): Robinhood-style dashboard generated. Verdict={verdict}. "
            f"Benchmark recall={benchmark.get('estimated_recall',0):.0%}. "
            f"Chatbot embedded. Download buttons: HTML + CSV + JSON.",
            dataset_name="data_rescue",
        )
    except Exception:
        pass

    print(f"[Agent 4 — Explain It] Dashboard written → {report_path}  (verdict: {verdict})")
    print(f"[Agent 4 — Explain It] Benchmark: ~{benchmark.get('estimated_recall',0):.0%} recall, {benchmark.get('categories_found',0)} categories")
    return str(report_path)
