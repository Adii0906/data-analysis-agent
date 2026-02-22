"""tools/report_tool.py — fixed absolute OUTPUTS_DIR"""

import base64, json, logging
from datetime import datetime, timezone
from pathlib import Path
from langchain.tools import tool
from exceptions import ReportGenerationError

logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent


def _b64(p):
    try: return base64.b64encode(Path(p).read_bytes()).decode()
    except: return ""


@tool
def generate_html_report(stats_json: str, analysis_summary: str,
                          chart_paths_json: str, dataset_name: str = "Dataset") -> str:
    """Generates self-contained HTML report with embedded charts. Returns JSON with report_html."""
    try:
        stats       = json.loads(stats_json) if stats_json else {}
        chart_paths = json.loads(chart_paths_json) if chart_paths_json else []
        ts          = datetime.now(timezone.utc).strftime("%B %d, %Y — %H:%M UTC")

        charts_html = ""
        for cp in chart_paths:
            if cp and Path(cp).exists():
                b64  = _b64(cp)
                name = Path(cp).stem.replace("_", " ").title()
                charts_html += f'<div class="chart-card"><h3>{name}</h3><img src="data:image/png;base64,{b64}" /></div>\n'

        desc       = stats.get("descriptive_stats", {})
        stat_cards = ""
        for col, s in desc.items():
            def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v) if v is not None else "N/A"
            stat_cards += f"""<div class="scard"><div class="scol">{col}</div>
            <div class="sgrid">
              <div><span class="sl">Mean</span><span class="sv">{fmt(s.get('mean'))}</span></div>
              <div><span class="sl">Std</span><span class="sv">{fmt(s.get('std'))}</span></div>
              <div><span class="sl">Min</span><span class="sv">{fmt(s.get('min'))}</span></div>
              <div><span class="sl">Max</span><span class="sv">{fmt(s.get('max'))}</span></div>
            </div></div>"""

        outliers    = stats.get("outlier_counts_iqr", {})
        outlier_rows = "".join(f"<tr><td>{c}</td><td>{v}</td></tr>" for c,v in outliers.items())

        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>Report — {dataset_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet"/>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080D18;color:#F0F4FF;font-family:'Plus Jakarta Sans',sans-serif;padding: 20px;}}
.hero{{background:linear-gradient(135deg,#0a1628,#080D18);border:1px solid #1E2E4A;border-radius:12px;padding:48px 56px 40px}}
.hero h1{{font-size:2.5rem;font-weight:800;margin-bottom:8px}}.hero h1 span{{color:#3B82F6;font-style:italic}}
.hero p{{color:#8899BB;font-size:.85rem;margin-top:10px}}
.pills{{display:flex;gap:8px;margin-top:16px}}
.pill{{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;letter-spacing:1px;
       padding:3px 12px;border-radius:20px;background:#1e3a6e22;border:1px solid #3B82F644;color:#60A5FA}}
.container{{max-width:1200px;margin:0 auto;padding:48px 0}}
.sh{{font-size:.75rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
    color:#8899BB;margin:40px 0 16px;display:flex;align-items:center;gap:10px}}
.sh::after{{content:'';flex:1;height:1px;background:#1E2E4A}}
.summary{{background:#0F1729;border:1px solid #1E2E4A;border-left:3px solid #3B82F6;
          border-radius:0 8px 8px 0;padding:24px 28px;font-size:.9rem;line-height:1.8;
          color:#C8D8F0;white-space:pre-wrap}}
.scards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
.scard{{background:#0F1729;border:1px solid #1E2E4A;border-radius:8px;padding:16px}}
.scol{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:#3B82F6;
       text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}}
.sgrid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.sgrid div{{display:flex;flex-direction:column;gap:2px}}
.sl{{font-size:10px;color:#8899BB;text-transform:uppercase;letter-spacing:.8px}}
.sv{{font-size:1rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:#0F1729;color:#3B82F6;padding:10px 14px;text-align:left;font-size:10px;
    letter-spacing:1.5px;text-transform:uppercase;border-bottom:1px solid #1E2E4A}}
td{{padding:9px 14px;border-bottom:1px solid #1E2E4A}}
tr:hover td{{background:#0F1729}}
.charts{{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:20px}}
.chart-card{{background:#0F1729;border:1px solid #1E2E4A;border-radius:8px;padding:18px}}
.chart-card h3{{font-size:11px;font-weight:700;color:#8899BB;text-transform:uppercase;
               letter-spacing:1.5px;margin-bottom:14px}}
.chart-card img{{width:100%;border-radius:6px}}
footer{{border-top:1px solid #1E2E4A;padding:28px 40px;text-align:center;font-size:11px;color:#8899BB;margin-top:60px}}
</style></head><body>
<div class="hero">
  <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#3B82F6;margin-bottom:10px;">Autonomous Data Analysis Report</div>
  <h1>Insight Report — <span>{dataset_name}</span></h1>
  <div class="pills"><span class="pill">Groq LLM</span><span class="pill">LangGraph</span><span class="pill">LangSmith</span><span class="pill">Auto-Cleaned</span></div>
  <p>Generated {ts}</p>
</div>
<div class="container">
  <div class="sh">Executive Summary</div>
  <div class="summary">{analysis_summary}</div>
  <div class="sh">Statistical Overview</div>
  <div class="scards">{stat_cards}</div>
  {"<div class='sh'>Outlier Analysis</div><table><thead><tr><th>Column</th><th>Outlier Count (IQR)</th></tr></thead><tbody>" + outlier_rows + "</tbody></table>" if outlier_rows else ""}
  <div class="sh">Visualizations</div>
  <div class="charts">{charts_html}</div>
</div>
<footer>Autonomous Data Analysis Agent — Groq · LangGraph · LangChain · LangSmith</footer>
</body></html>"""

        return json.dumps({"status": "success", "report_html": html})
    except Exception as e:
        raise ReportGenerationError(f"Report failed: {e}", {"dataset": dataset_name}, e)