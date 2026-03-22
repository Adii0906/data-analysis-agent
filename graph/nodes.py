"""
graph/nodes.py
Contains the logic for each node in the LangGraph workflow.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from exceptions import (
    AnalysisPlanError,
    CleaningError,
    LLMError,
    ReportGenerationError,
    SchemaInspectionError,
    SelfCorrectionError,
    ToolExecutionError,
    VisualizationError,
)
from tools import (
    compute_statistics,
    detect_missing_values,
    generate_html_report,
    handle_missing_values,
    inspect_schema,
    plot_boxplots,
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_distributions,
)

logger    = logging.getLogger(__name__)
MAX_RETRY = 3


# ── safe wrapper ──────────────────────────────────────────────────────────────

def _call(tool_fn, payload, state: dict, err_cls=ToolExecutionError):
    """
    Invoke tool_fn with payload (str or dict).
    Retries up to MAX_RETRY times internally.
    """
    attempts = 0
    while attempts < MAX_RETRY:
        try:
            result = tool_fn.invoke(payload)
            state["retry_count"] = 0
            return result, state
        except Exception as e:
            attempts += 1
            state["retry_count"] = attempts
            err = err_cls(str(e), {"tool": tool_fn.name}, e)
            state.setdefault("error_log", []).append(err.to_dict())
            state["last_error"] = err
            if attempts >= MAX_RETRY:
                raise SelfCorrectionError(
                    f"'{tool_fn.name}' failed after {MAX_RETRY} attempts.",
                    {"last_error": str(e)}, e,
                )
            logger.warning(f"{tool_fn.name} failed (attempt {attempts}): {e}")
    return None, state


# ── Node 1: create working copy ───────────────────────────────────────────────

def create_working_copy_node(state: dict) -> dict:
    state["current_step"] = "create_working_copy"
    state["step_status"]  = "running"

    # Per user request: "dont create a duplicate csv for now"
    # We'll just use the original path as the working copy path.
    state["working_copy_path"] = state["original_path"]
    state["step_status"] = "done"
    state.setdefault("messages", []).append({
        "role":    "assistant",
        "content": "ℹ️ Using original dataset file (no duplicate created as requested).",
    })
    return state


# ── Node 2: inspect schema ────────────────────────────────────────────────────

def inspect_schema_node(state: dict) -> dict:
    if state.get("schema_info"):
        state["step_status"] = "done"
        return state

    state["current_step"] = "inspect_schema"
    state["step_status"]  = "running"

    wcp = state.get("working_copy_path")
    if not wcp:
        state["working_copy_path"] = state.get("original_path")
        wcp = state["working_copy_path"]

    if not wcp:
        state["step_status"] = "error"
        state.setdefault("messages", []).append({
            "role": "assistant", "content": "❌ No dataset path found in state."
        })
        return state

    result_str, state = _call(inspect_schema, wcp, state, SchemaInspectionError)
    if result_str:
        schema = json.loads(result_str)
        state["schema_info"]  = schema
        state["step_status"]  = "done"
        shape = schema["shape"]
        cols  = ", ".join(f"`{c['name']}` ({c['dtype']})" for c in schema["columns"][:6])
        extra = f" +{len(schema['columns'])-6} more" if len(schema["columns"]) > 6 else ""
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": (
                f"📊 **Schema** — {shape['rows']:,} rows × {shape['columns']} cols\n"
                f"Columns: {cols}{extra}\n"
                f"Memory: {schema['memory_usage_kb']} KB"
            ),
        })
    return state


# ── Node 3: detect missing ────────────────────────────────────────────────────

def detect_missing_node(state: dict) -> dict:
    if state.get("missing_strategy"):
        state["step_status"] = "done"
        state["awaiting_user_input"] = False
        return state

    state["current_step"] = "detect_missing"
    state["step_status"]  = "running"

    wcp = state.get("working_copy_path") or state.get("original_path")
    if not wcp:
        state["step_status"] = "error"
        state.setdefault("messages", []).append({
            "role": "assistant", "content": "❌ No dataset path found for missing value detection."
        })
        return state

    result_str, state = _call(detect_missing_values, wcp, state)
    if result_str:
        report   = json.loads(result_str)
        affected = report["columns_with_missing"]
        total    = report["total_missing_cells"]
        state["missing_value_report"] = report
        state["step_status"]          = "done"

        if total == 0:
            state["missing_strategy"] = "none"
            state["user_confirmed"]   = True
            state.setdefault("messages", []).append({
                "role": "assistant", "content": "✅ No missing values — dataset is clean."
            })
        else:
            lines = [f"⚠️ **{total:,} missing cells** across {len(affected)} columns:\n"]
            for c in affected[:8]:
                lines.append(f"  • `{c['column']}` — {c['missing_count']:,} ({c['missing_pct']}%)")
            if len(affected) > 8:
                lines.append(f"  … +{len(affected)-8} more")
            lines.append("\n**Choose how to handle missing values below ↓**")
            state.setdefault("messages", []).append({
                "role": "assistant", "content": "\n".join(lines)
            })
            state["awaiting_user_input"]  = True
            state["pending_user_question"] = "missing_strategy"
    return state


# ── Node 4: handle missing ────────────────────────────────────────────────────

def handle_missing_node(state: dict) -> dict:
    # If strategy is already applied (indicated by step_status='done' or similar from previous stream), skip
    if state.get("current_step") == "handle_missing" and state.get("step_status") == "done":
        return state

    state["current_step"] = "handle_missing"
    state["step_status"]  = "running"

    strategy = state.get("missing_strategy", "median")
    if strategy == "none":
        state["step_status"] = "done"
        return state

    # Tool expects two keyword args — pass as dict
    wcp = state.get("working_copy_path") or state.get("original_path")
    result_str, state = _call(
        handle_missing_values,
        {"file_path": wcp, "strategy": strategy},
        state, CleaningError,
    )
    if result_str:
        r = json.loads(result_str)
        state["step_status"] = "done"
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": (
                f"🧹 **Cleaned** (strategy: `{strategy}`)\n"
                f"  Missing before: {r['missing_before']:,}  →  after: {r['missing_after']:,}\n"
                f"  Rows remaining: {r['rows_remaining']:,}"
            ),
        })
    return state


# ── Node 5: generate plan ─────────────────────────────────────────────────────

def generate_analysis_plan_node(state: dict, llm) -> dict:
    if state.get("analysis_plan"):
        state["step_status"] = "done"
        return state

    state["current_step"] = "generate_plan"
    state["step_status"]  = "running"

    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "You are an expert data scientist. Given a dataset schema, "
                "produce a concise numbered analysis plan (5–8 steps). "
                "Each step is one sentence. Return ONLY the numbered list."
            )),
            HumanMessage(content=(
                f"Dataset: {state.get('dataset_name','unknown')}\n"
                f"Schema: {json.dumps(state.get('schema_info',{}), default=str)[:1000]}\n\n"
                "Produce the analysis plan."
            )),
        ])
        plan = [l.strip() for l in resp.content.split("\n") if l.strip()]
        state["analysis_plan"] = plan
        state["step_status"]   = "done"
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": "📋 **Analysis Plan**\n\n" + "\n".join(plan),
        })
    except Exception as e:
        err = AnalysisPlanError(str(e), {}, e)
        state["last_error"] = err
        state.setdefault("error_log", []).append(err.to_dict())
        state["step_status"]   = "error"
        state["analysis_plan"] = [
            "1. Compute descriptive statistics",
            "2. Analyze distributions",
            "3. Check correlations",
            "4. Identify outliers via IQR",
            "5. Analyze categorical variables",
            "6. Generate visualizations",
            "7. Summarize findings",
        ]
    return state


# ── Node 6: execute analysis ──────────────────────────────────────────────────

def execute_analysis_node(state: dict) -> dict:
    if state.get("chart_paths") and state.get("stats_json"):
        state["step_status"] = "done"
        return state

    state["current_step"] = "execute_analysis"
    state["step_status"]  = "running"

    path        = state.get("working_copy_path") or state.get("original_path")
    chart_paths = []
    
    query_text = state.get('user_query', '').lower()
    is_auto_eda = not query_text or any(k in query_text for k in [
        "auto eda", "analyze dataset", "full eda", "complete overview",
        "generate report", "final report", "export report"
    ])

    msg = "⚙️ Running full Auto EDA and generating charts…" if is_auto_eda else "⚙️ Computing statistics for specific query…"
    state.setdefault("messages", []).append({
        "role": "assistant", "content": msg
    })

    stats_str, state = _call(compute_statistics, path, state)
    if stats_str:
        state["stats_json"] = stats_str

    if is_auto_eda:
        for fn in [plot_distributions, plot_correlation_heatmap, plot_boxplots, plot_categorical_bars]:
            res, state = _call(fn, path, state, VisualizationError)
            if res:
                try:
                    r = json.loads(res)
                    if r.get("status") == "success":
                        chart_paths.append(r["path"])
                except Exception:
                    pass

    state["chart_paths"] = chart_paths
    state["step_status"] = "done"
    
    if is_auto_eda:
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"📈 Done — {len(chart_paths)} charts generated.",
        })
    else:
         state.setdefault("messages", []).append({
            "role": "assistant",
            "content": "📈 Done computing statistics.",
        })       
    return state


# ── Node 7: summarize ─────────────────────────────────────────────────────────

def summarize_findings_node(state: dict, llm) -> dict:
    if state.get("analysis_summary"):
        state["step_status"] = "done"
        return state

    state["current_step"] = "summarize"
    state["step_status"]  = "running"

    try:
        query_text = state.get('user_query', '').strip()
        system_prompt = (
            "You are a Data Analysis Agent.\n"
            "A smart assistant that helps users explore datasets step by step, answer questions in natural language, and generate clear insights with visualizations and final reports.\n\n"
            "## How It Works\n"
            "* Users ask questions in natural language about the dataset\n"
            "* You analyze ONLY what is asked (no full EDA by default)\n"
            "* You guide the user step by step like a real data analyst\n"
            "* You maintain context for follow-up questions\n"
            "* You generate a final structured report when requested\n\n"
            "## Instructions\n"
            "1. Understand the user query\n"
            "   * Identify intent (trend, comparison, distribution, summary, anomaly)\n"
            "   * Identify relevant dataset columns\n"
            "2. Step-by-step analysis\n"
            "   * Perform ONLY the requested analysis\n"
            "   * Do NOT run full dataset analysis unless explicitly asked\n"
            "   * Build on previous context for follow-up queries\n"
            "3. User interaction\n"
            "   * Briefly explain what you are doing (1-2 lines, simple language)\n"
            "   * Keep responses conversational and clear\n"
            "4. Visualization\n"
            "   * Suggest the most appropriate chart (line, bar, histogram, scatter)\n"
            "   * Explain what the chart represents\n"
            "5. Insights\n"
            "   * Provide clear, simple insights\n"
            "   * Highlight key trends, comparisons, or anomalies\n\n"
            "## Auto EDA (ONLY WHEN EXPLICITLY REQUESTED)\n"
            "Run full dataset analysis ONLY if user asks (e.g., \"analyze dataset\", \"full EDA\", \"complete overview\"):\n"
            "* Dataset shape\n"
            "* Column types\n"
            "* Missing values\n"
            "* Summary statistics\n"
            "* Key patterns\n"
            "* 3-5 relevant visualizations\n\n"
            "## Final Report Mode\n"
            "If the user asks for a report (e.g., \"generate report\", \"final report\", \"export report\"):\n"
            "Generate a structured report using previous analyses following the structure:\n"
            "### 📌 Overview\n"
            "* Brief summary of dataset and purpose\n"
            "### 🔍 Key Analyses Performed\n"
            "* Step-by-step summary of what was analyzed\n"
            "### 📊 Visualizations\n"
            "* Charts created and what they show\n"
            "### 💡 Key Insights\n"
            "* Most important findings\n"
            "### 📈 Recommendations (Optional)\n"
            "* Suggested next steps or actions\n\n"
            "## Output Format (Normal Queries)\n"
            "### 💬 Plan\n"
            "(short explanation of current step)\n"
            "### 📊 Visualization\n"
            "(chart type + what it shows)\n"
            "### 💡 Insights\n"
            "* Key finding 1\n"
            "* Key finding 2\n"
            "* Key finding 3\n\n"
            "## Rules\n"
            "* Do NOT run full EDA automatically\n"
            "* Do NOT hallucinate column names\n"
            "* Keep responses simple and non-technical\n"
            "* Maintain context across queries\n"
            "* Do NOT generate code\n"
            "* Be concise, clear, and product-like\n"
            "You behave like a mini SaaS data analysis tool, not just a chatbot."
        )
        human_prompt = (
            f"Dataset: {state.get('dataset_name','Dataset')}\n"
            f"User Query: {query_text if query_text else 'Analyze dataset (Auto EDA)'}\n"
            f"Schema: {json.dumps(state.get('schema_info',{}), default=str)[:1000]}\n"
            f"Historical Context / Chat Messages: {json.dumps(state.get('messages', [])[-3:])}\n"
            f"Statistics Sample: {state.get('stats_json','{}')[:2000]}\n\n"
            "Perform the analysis and generate the required markdown response."
        )

        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        state["analysis_summary"] = resp.content
        state["step_status"]      = "done"
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"📝 **Analysis Narrative**\n\n{resp.content}"
        })
    except Exception as e:
        err = LLMError(str(e), {}, e)
        state["last_error"] = err
        state.setdefault("error_log", []).append(err.to_dict())
        state["analysis_summary"] = "Summary unavailable — review charts directly."
        state["step_status"]      = "error"
    return state


# ── Node 8: generate report ───────────────────────────────────────────────────

def generate_report_node(state: dict) -> dict:
    if state.get("report_html") and state.get("report_html") != "skip":
        state["step_status"] = "done"
        return state

    state["current_step"] = "generate_report"
    state["step_status"]  = "running"
    
    query_text = state.get('user_query', '').lower()
    is_report = not query_text or any(k in query_text for k in [
        "generate report", "final report", "export report", 
        "auto eda", "analyze dataset", "full eda", "complete overview"
    ])
    
    if not is_report:
        state["report_html"] = "skip"
        state["step_status"] = "done"
        return state

    result_str, state = _call(
        generate_html_report,
        {
            "stats_json":       state.get("stats_json", "{}"),
            "analysis_summary": state.get("analysis_summary", ""),
            "chart_paths_json": json.dumps(state.get("chart_paths", [])),
            "dataset_name":     state.get("dataset_name", "Dataset"),
        },
        state, ReportGenerationError,
    )
    if result_str:
        r = json.loads(result_str)
        state["report_html"] = r.get("report_html", "")
        state["step_status"] = "done"
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": "📄 **Report generated** — full HTML layout is ready.",
        })
    return state