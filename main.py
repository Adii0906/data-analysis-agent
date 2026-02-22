"""
main.py — Autonomous Data Analysis Agent
- Charts rendered with ONLY st.image() — no HTML wrappers blocking them
- Sidebar: card-style with pipeline status tracker
- Download report shown LAST after all results
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from exceptions import AgentBaseException, SlackErrorReporter
from graph import AgentState, build_graph
from utils.session import init_session, push_message, reset_session

logging.basicConfig(level=logging.INFO)
logger    = logging.getLogger(__name__)
slack_rep = SlackErrorReporter()

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#060E1C; --surf:#0B1628; --s2:#0F1F38; --bdr:#1A2F4A;
  --blue:#3B82F6; --blue2:#1D4ED8; --grn:#22C55E; --red:#EF4444;
  --amber:#F59E0B; --txt:#EDF2FF; --sub:#6B85A8; --muted:#3A5070;
}

html,body,[class*="css"]{background:var(--bg)!important;color:var(--txt)!important;font-family:'Inter',sans-serif!important;}
.stApp{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2.2rem 3rem!important;max-width:1140px!important;}

/* ── sidebar ── */
section[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bdr)!important;}
section[data-testid="stSidebar"] .block-container{padding:0!important;max-width:100%!important;}

/* ── buttons ── */
div.stButton>button{
  background:var(--blue)!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-weight:700!important;font-size:.84rem!important;
  padding:9px 22px!important;transition:background .15s!important;letter-spacing:.1px!important;
}
div.stButton>button:hover{background:var(--blue2)!important;}
div.stButton>button:disabled{background:var(--muted)!important;opacity:.6!important;}

[data-testid="stDownloadButton"]>button{
  background:var(--grn)!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-weight:700!important;font-size:.84rem!important;padding:9px 22px!important;
}

/* ── metrics ── */
[data-testid="stMetric"]{background:var(--surf)!important;border:1px solid var(--bdr)!important;border-radius:10px!important;padding:18px 16px!important;}
[data-testid="stMetricLabel"]{color:var(--sub)!important;font-size:.68rem!important;letter-spacing:1px!important;text-transform:uppercase!important;font-family:'JetBrains Mono',monospace!important;}
[data-testid="stMetricValue"]{color:var(--txt)!important;font-size:1.65rem!important;font-weight:800!important;}

/* ── tabs ── */
button[data-baseweb="tab"]{font-weight:600!important;font-size:.84rem!important;color:var(--sub)!important;background:transparent!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--blue)!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:var(--blue)!important;}

/* ── expander ── */
details{background:var(--surf)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;}
details summary{font-family:'JetBrains Mono',monospace!important;font-size:.76rem!important;color:var(--sub)!important;padding:10px 14px!important;}

/* ── inputs ── */
[data-testid="stFileUploader"]{background:var(--s2)!important;border:1.5px dashed var(--bdr)!important;border-radius:10px!important;}
[data-testid="stTextInput"] input{background:var(--s2)!important;border:1px solid var(--bdr)!important;border-radius:7px!important;color:var(--txt)!important;font-family:'JetBrains Mono',monospace!important;font-size:.79rem!important;}
label{font-size:.72rem!important;color:var(--sub)!important;font-weight:600!important;letter-spacing:.3px!important;}

hr{border-color:var(--bdr)!important;margin:18px 0!important;}

/* ── Glassmorphism & Cards ── */
[data-testid="stExpander"], .stMetric, div[style*="background:#0F1F38"] {
  background: rgba(15, 31, 56, 0.7) !important;
  backdrop-filter: blur(10px) !important;
  border: 1px solid rgba(59, 130, 246, 0.2) !important;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
  border-radius: 12px !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1628 0%, #060E1C 100%) !important;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bdr); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
STEPS = [
    ("create_working_copy", "Working Copy"),
    ("inspect_schema",      "Schema Inspection"),
    ("detect_missing",      "Missing Values"),
    ("handle_missing",      "Data Cleaning"),
    ("generate_plan",       "Analysis Plan"),
    ("execute_analysis",    "Execution"),
    ("summarize",           "Narrative"),
    ("generate_report",     "Report"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure Streamlit helpers — ZERO HTML wrappers around images
# ─────────────────────────────────────────────────────────────────────────────



def render_chat():
    msgs = st.session_state.get("messages", [])
    if not msgs:
        st.caption("No messages yet.")
        return
    for m in msgs:
        role    = m.get("role", "assistant")
        content = m.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def run_graph(state: dict) -> dict:
    try:
        if "graph" not in st.session_state or st.session_state.graph is None:
            from graph import build_graph
            st.session_state.graph = build_graph()
        
        # Initialize missing state fields
        state.setdefault("current_step", "")
        state.setdefault("step_status", "pending")
        state.setdefault("messages", [])
        state.setdefault("retry_count", 0)
        state.setdefault("error_log", [])
        state.setdefault("awaiting_user_input", False)
        state.setdefault("chart_paths", [])
        
        for chunk in st.session_state.graph.stream(state, stream_mode="values"):
            state = chunk
            st.session_state.agent_state = state
            st.session_state.messages = state.get("messages", [])
            
            err = state.get("last_error")
            if err and isinstance(err, AgentBaseException):
                slack_rep.report(err, {"dataset": state.get("dataset_name", "")})
            
            if state.get("awaiting_user_input"):
                break
    except Exception as e:
        logger.exception("Graph error")
        state["step_status"] = "error"
        state.setdefault("messages", []).append(
            {"role": "assistant", "content": f"❌ Pipeline error: {e}"})
        st.session_state.messages = state["messages"]
        st.session_state.agent_state = state
    return state


def send_to_slack(state: dict) -> bool:
    wh = os.getenv("SLACK_WEBHOOK_URL", "")
    if not wh:
        return False
    import requests as rq
    summary = state.get("analysis_summary", "No summary.")[:600]
    shape   = state.get("schema_info", {}).get("shape", {})
    rows    = shape.get("rows", "?")
    payload = {
        "blocks": [
            {"type": "header",  "text": {"type": "plain_text", "text": f"📊 Report: {state.get('dataset_name','Dataset')}"}},
            {"type": "divider"},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Rows:* {rows:,}" if isinstance(rows, int) else f"*Rows:* {rows}"},
                {"type": "mrkdwn", "text": f"*Columns:* {shape.get('columns','?')}"},
                {"type": "mrkdwn", "text": f"*Strategy:* `{state.get('missing_strategy','—')}`"},
                {"type": "mrkdwn", "text": f"*Charts:* {len(state.get('chart_paths',[]))}"},
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Summary:*\n> {summary}"}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "_Autonomous Data Analysis Agent 🔬_"}]},
        ]
    }
    try:
        r = rq.post(wh, json=payload, timeout=8)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Slack error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — card style with pipeline status
# ─────────────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        # ── Branding card ──────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#0F1F38;border:1px solid #1A2F4A;border-radius:10px;
                    padding:16px 18px;margin:16px 12px 8px;">
          <div style="font-size:1rem;font-weight:800;color:#EDF2FF;margin-bottom:4px;">
            🔬 Data Analysis Agent
          </div>
          <div style="font-size:.72rem;color:#6B85A8;line-height:1.5;">
            Autonomous · LangGraph · Groq LLM
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;">
            <span style="font-size:.6rem;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:#60A5FA;padding:2px 8px;border-radius:20px;font-weight:600;">LANGCHAIN</span>
            <span style="font-size:.6rem;background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:#60A5FA;padding:2px 8px;border-radius:20px;font-weight:600;">LANGGRAPH</span>
            <span style="font-size:.6rem;background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);color:#4ADE80;padding:2px 8px;border-radius:20px;font-weight:600;">LANGSMITH</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # ── Pipeline status card ────────────────────────────────────────────
        state  = st.session_state.get("agent_state", {})
        cur    = state.get("current_step", "")
        status = state.get("step_status", "")

        # Find index of current step
        cur_idx = -1
        for i, (sid, _) in enumerate(STEPS):
            if sid == cur:
                cur_idx = i
                break

        # Build pipeline progress display
        progress_html = '<style>@keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.2); opacity: 0.7; } 100% { transform: scale(1); opacity: 1; } }</style>'
        progress_html += '<div style="background:#0F1F38;border:1px solid #1A2F4A;border-radius:12px;padding:16px 18px;margin:4px 12px 12px;box-shadow: 0 4px 15px rgba(0,0,0,0.3);">'
        progress_html += '<div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#6B85A8;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;"><span>PIPELINE PROGRESS</span><span style="background:rgba(59,130,246,0.1);padding:2px 6px;border-radius:4px;color:#3B82F6;font-size:0.6rem;">' + str(cur_idx+1 if cur_idx >=0 else 0) + '/' + str(len(STEPS)) + '</span></div>'

        for i, (sid, slabel) in enumerate(STEPS):
            if i < cur_idx:
                # Completed
                dot = 'style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#22C55E;box-shadow:0 0 4px #22C55E;"'
                txt = 'style="flex-grow:1;color:#4ADE80;opacity:0.9;font-size:.78rem;font-family:\'Inter\',sans-serif;"'
                icon = "✅"
                bg = 'style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin-bottom:6px;background:rgba(34,197,94,0.05);border-radius:8px;transition:all 0.3s ease;"'
            elif i == cur_idx:
                # Active
                if status == "error":
                    dot = 'style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#EF4444;box-shadow:0 0 8px #EF4444;"'
                    txt = 'style="flex-grow:1;color:#F87171;font-weight:700;font-size:.78rem;font-family:\'Inter\',sans-serif;"'
                    icon = "❌"
                    bg = 'style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin-bottom:6px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);border-radius:8px;transition:all 0.3s ease;"'
                else:  # pending or running
                    dot = 'style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#F59E0B;box-shadow:0 0 10px #F59E0B;animation:pulse 2s infinite;"'
                    txt = 'style="flex-grow:1;color:#FBBF24;font-weight:700;font-size:.78rem;font-family:\'Inter\',sans-serif;"'
                    icon = "⏳"
                    bg = 'style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin-bottom:6px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);border-radius:8px;transition:all 0.3s ease;"'
            else:
                # Future
                dot = 'style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#1A2F4A;"'
                txt = 'style="flex-grow:1;color:#3A5070;font-size:.78rem;font-family:\'Inter\',sans-serif;"'
                icon = "⚪"
                bg = 'style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin-bottom:6px;background:transparent;border-radius:8px;transition:all 0.3s ease;"'

            progress_html += f'<div {bg}><div {dot}></div><div {txt}>{slabel}</div><div style="font-size:0.8rem;">{icon}</div></div>'

        progress_html += '</div>'
        st.markdown(progress_html, unsafe_allow_html=True)

        # ── Config card ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#0F1F38;border:1px solid #1A2F4A;border-radius:10px;
                    padding:14px 18px;margin:4px 12px 8px;">
          <div style="font-size:.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                      color:#6B85A8;margin-bottom:12px;">CONFIGURATION</div>
        """, unsafe_allow_html=True)

        gk = st.text_input("Groq API Key",  type="password", value=os.getenv("GROQ_API_KEY", ""),      placeholder="gsk_…",                  key="sb_groq")
        lk = st.text_input("LangSmith Key", type="password", value=os.getenv("LANGCHAIN_API_KEY", ""), placeholder="ls__…",                  key="sb_ls")
        sw = st.text_input("Slack Webhook", type="password", value=os.getenv("SLACK_WEBHOOK_URL", ""), placeholder="https://hooks.slack.com/…", key="sb_slack")

        if gk: os.environ["GROQ_API_KEY"]         = gk
        if lk: os.environ["LANGCHAIN_API_KEY"]     = lk; os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if sw: os.environ["SLACK_WEBHOOK_URL"]     = sw

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Errors card ──────────────────────────────────────────────────────
        errors = state.get("error_log", [])
        if errors:
            err_rows = ""
            for e in errors[-3:]:
                err_rows += f"""
                <div style="background:#1a0808;border:1px solid #7f1d1d;border-radius:6px;
                            padding:8px 10px;margin-bottom:6px;">
                  <div style="color:#F87171;font-size:.7rem;font-weight:700;">{e['exception_type']}</div>
                  <div style="color:#9CA3AF;font-size:.67rem;margin-top:2px;">{e['message'][:55]}…</div>
                </div>"""
            st.markdown(f"""
            <div style="background:#0F1F38;border:1px solid #7f1d1d;border-radius:10px;
                        padding:14px 16px;margin:4px 12px 8px;">
              <div style="font-size:.65rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                          color:#F87171;margin-bottom:10px;">ERRORS ({len(errors)})</div>
              {err_rows}
            </div>
            """, unsafe_allow_html=True)

        # ── Reset button ─────────────────────────────────────────────────────
        st.markdown("<div style='padding:4px 12px 16px;'>", unsafe_allow_html=True)
        if st.button("🔄  New Analysis", use_container_width=True, key="reset_btn"):
            reset_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_session()
    sidebar()

    # Top bar
    st.markdown("""
    <div style="background:#0B1628;border-bottom:1px solid #1A2F4A;
                padding:14px 0 12px;margin-bottom:28px;">
      <div style="max-width:1140px;margin:0 auto;padding:0 2.2rem;
                  display:flex;align-items:center;justify-content:space-between;">
        <div>
          <span style="font-size:1.1rem;font-weight:800;color:#EDF2FF;">🔬 Autonomous</span>
          <span style="font-size:1.1rem;font-weight:800;color:#3B82F6;font-style:italic;"> Data Analysis</span>
          <span style="font-size:1.1rem;font-weight:800;color:#EDF2FF;"> Agent</span>
        </div>
        <div style="font-size:.72rem;color:#6B85A8;font-family:'JetBrains Mono',monospace;">
          Upload → Clean → Analyze → Visualize → Share
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    stage = st.session_state.stage

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE: UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    if stage == "upload":
        st.markdown("## Upload Your Dataset")
        st.caption("Supported formats: CSV · Excel (.xlsx/.xls) · JSON · Parquet")
        st.markdown("")

        uploaded = st.file_uploader(
            "Drop your file here or click to browse",
            type=["csv", "xlsx", "xls", "json", "parquet"],
        )

        if uploaded:
            # dedupe guard
            if st.session_state.get("_fname") != uploaded.name:
                st.session_state["_fname"] = uploaded.name
                suf = Path(uploaded.name).suffix
                
                # Use a dedicated tmp directory instead of cluttering the root
                tmp_dir = BASE_DIR / "tmp"
                tmp_dir.mkdir(exist_ok=True)
                
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=suf, dir=str(tmp_dir))
                tmp.write(uploaded.getvalue())
                tmp.close()
                st.session_state.original_path = tmp.name
                st.session_state.dataset_name  = Path(uploaded.name).stem

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.success(f"✅ **{uploaded.name}** — {round(uploaded.size/1024, 1)} KB ready")
            with col_btn:
                if st.button("▶  Run Analysis", use_container_width=True):
                    st.session_state.stage = "init"
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE: INIT
    # ══════════════════════════════════════════════════════════════════════════
    elif stage == "init":
        if st.session_state.graph is None:
            st.session_state.graph = build_graph()

        with st.spinner("🔄 Initializing pipeline — creating working copy & inspecting schema…"):
            state = run_graph({
                "original_path":       st.session_state.original_path,
                "dataset_name":        st.session_state.dataset_name,
                "messages":            [],
                "retry_count":         0,
                "error_log":           [],
                "user_confirmed":      False,
                "awaiting_user_input": False,
                "current_step":        "",
                "step_status":         "pending",
                "chart_paths":         [],
            })

        st.session_state.agent_state = state
        st.session_state.stage = (
            "missing_strategy" if state.get("awaiting_user_input") else "running"
        )
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE: MISSING STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    elif stage == "missing_strategy":
        st.markdown("## 🧹 Handle Missing Values")

        with st.expander("📋 Agent findings", expanded=True):
            render_chat()

        st.markdown("---")
        st.markdown("**Choose how to handle missing values:**")

        c1, c2, c3, _ = st.columns([1, 1, 1, 2])
        strategy = None
        with c1:
            if st.button("📊  Mean",   use_container_width=True): strategy = "mean"
        with c2:
            if st.button("📐  Median", use_container_width=True): strategy = "median"
        with c3:
            if st.button("🗑️  Drop",   use_container_width=True): strategy = "drop"

        if strategy:
            push_message("user", f"Handle missing values using **{strategy}** strategy.")
            state = st.session_state.agent_state
            state.update({
                "missing_strategy":      strategy,
                "awaiting_user_input":   False,
                "pending_user_question": None,
                "messages":              st.session_state.messages,
            })
            with st.spinner(f"Applying '{strategy}' strategy…"):
                state = run_graph(state)
            st.session_state.agent_state = state
            st.session_state.stage = "running"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE: RUNNING
    # ══════════════════════════════════════════════════════════════════════════
    elif stage == "running":
        state = st.session_state.agent_state
        if not state.get("report_html"):
            with st.spinner("🤖 Agent running — computing stats, generating charts, writing narrative…"):
                state["awaiting_user_input"] = False
                state["messages"]            = st.session_state.messages
                state = run_graph(state)
            st.session_state.agent_state = state

            if state.get("report_html"):
                st.session_state.stage = "done"
                st.rerun()
            else:
                st.error("Pipeline finished but no report was generated.")
                with st.expander("Agent Log"):
                    render_chat()
                return

        # If somehow landed here with report already done
        st.session_state.stage = "done"
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE: DONE  ← FULL RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    elif stage == "done":
        state = st.session_state.agent_state

        # ── Banner ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(90deg,#052e16,#060E1C);
                    border:1px solid rgba(34,197,94,.4);border-radius:12px;
                    padding:18px 24px;display:flex;align-items:center;gap:14px;margin-bottom:24px;">
          <span style="font-size:2rem;">✅</span>
          <div>
            <div style="font-size:1.1rem;font-weight:800;color:#4ADE80;">Analysis Complete</div>
            <div style="font-size:.78rem;color:#6B85A8;margin-top:3px;">
              The interactive report is ready below.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab_report, tab_log = st.tabs(["📄  Interactive Report", "💬  Agent Log"])

        with tab_report:
            report_html = state.get("report_html", "")
            if report_html:
                # Inject a small script to auto-resize the iframe if possible, or just give it a large fixed height
                components.html(report_html, height=1200, scrolling=True)
            else:
                st.error("No report HTML found in state.")

        with tab_log:
            render_chat()

        # ── Non-fatal errors ─────────────────────────────────────────────────
        errs = state.get("error_log", [])
        if errs:
            with st.expander(f"⚠️ {len(errs)} non-fatal error(s) during run"):
                for e in errs:
                    st.json(e)

        st.markdown("---")

        # ── DOWNLOAD & SHARE ───────────────────────────────────────────────
        st.markdown("---")
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("### 📄 Download Report")
            st.caption("Save the full interactive HTML report to your computer.")
            if report_html:
                st.download_button(
                    label="💾  Download HTML Report",
                    data=report_html,
                    file_name=f"Analysis_Report_{state.get('dataset_name', 'Dataset')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_btn"
                )

        with c2:
            st.markdown("### 📨 Share Findings")
            st.caption("Send a summary to your Slack team channel.")
            has_slack = bool(os.getenv("SLACK_WEBHOOK_URL"))
            if st.button("📨  Send to Slack", disabled=not has_slack, use_container_width=True, key="slack_btn"):
                with st.spinner("Sending report summary to Slack…"):
                    ok = send_to_slack(state)
                if ok:
                    st.success("✅ Sent to Slack!")
                else:
                    st.error("❌ Failed — verify your Slack webhook URL in the sidebar.")

        st.markdown("")
        st.info("💡 Tip: Click **🔄 New Analysis** in the sidebar to start over with a new dataset")


if __name__ == "__main__":
    main()