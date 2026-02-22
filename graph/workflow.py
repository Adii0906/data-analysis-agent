"""
graph/workflow.py
Builds and compiles the LangGraph StateGraph for the autonomous data analysis agent.
"""

import functools
import logging
import os
from typing import Literal

from langgraph.graph import END, StateGraph

from .nodes import (
    create_working_copy_node,
    detect_missing_node,
    execute_analysis_node,
    generate_analysis_plan_node,
    generate_report_node,
    handle_missing_node,
    inspect_schema_node,
    summarize_findings_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm():
    """Build Groq LLM via LangChain with LangSmith tracing enabled."""
    from langchain_groq import ChatGroq

    # LangSmith tracing — set these in .env
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "data-analysis-agent")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        logger.warning("GROQ_API_KEY not set — LLM calls will fail.")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=groq_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_detect_missing(state: AgentState) -> Literal["await_user", "handle_missing"]:
    if state.get("awaiting_user_input") and state.get("pending_user_question") == "missing_strategy":
        return "await_user"
    return "handle_missing"


def _route_after_missing_handled(state: AgentState) -> Literal["generate_plan", "error_end"]:
    if state.get("step_status") == "error":
        return "error_end"
    return "generate_plan"


def _route_self_correction(
    state: AgentState,
    next_node: str,
) -> Literal["retry", "error_end", "continue"]:
    if state.get("step_status") == "error":
        retries = state.get("retry_count", 0)
        if retries < 3:
            return "retry"
        return "error_end"
    return "continue"


# ─────────────────────────────────────────────────────────────────────────────
# Passthrough "await user" node
# ─────────────────────────────────────────────────────────────────────────────

def await_user_input_node(state: AgentState) -> AgentState:
    """Placeholder node — graph pauses here until Streamlit injects user input."""
    logger.info("Graph paused: awaiting user input for '%s'", state.get("pending_user_question"))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Error terminal node
# ─────────────────────────────────────────────────────────────────────────────

def error_end_node(state: AgentState) -> AgentState:
    last = state.get("last_error")
    msg = str(last) if last else "An unrecoverable error occurred."
    state.setdefault("messages", []).append({
        "role": "assistant",
        "content": f"❌ **Agent stopped due to an error:** {msg}\n\nPlease check the error log.",
    })
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    llm = _build_llm()

    # Partial-apply llm into nodes that need it
    plan_node = functools.partial(generate_analysis_plan_node, llm=llm)
    summarize_node = functools.partial(summarize_findings_node, llm=llm)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("create_working_copy", create_working_copy_node)
    graph.add_node("inspect_schema", inspect_schema_node)
    graph.add_node("detect_missing", detect_missing_node)
    graph.add_node("await_user_input", await_user_input_node)
    graph.add_node("handle_missing", handle_missing_node)
    graph.add_node("generate_plan", plan_node)
    graph.add_node("execute_analysis", execute_analysis_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("error_end", error_end_node)

    # Entry point
    graph.set_entry_point("create_working_copy")

    # Linear happy path edges
    graph.add_edge("create_working_copy", "inspect_schema")
    graph.add_edge("inspect_schema", "detect_missing")

    # Branch: wait for user OR proceed
    graph.add_conditional_edges(
        "detect_missing",
        _route_after_detect_missing,
        {
            "await_user": "await_user_input",
            "handle_missing": "handle_missing",
        },
    )

    # After user responds (Streamlit resumes graph from handle_missing)
    graph.add_edge("await_user_input", "handle_missing")

    graph.add_conditional_edges(
        "handle_missing",
        _route_after_missing_handled,
        {"generate_plan": "generate_plan", "error_end": "error_end"},
    )

    graph.add_edge("generate_plan", "execute_analysis")
    graph.add_edge("execute_analysis", "summarize")
    graph.add_edge("summarize", "generate_report")
    graph.add_edge("generate_report", END)
    graph.add_edge("error_end", END)

    return graph.compile()