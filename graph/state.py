"""
graph/state.py
TypedDict state shared across all LangGraph nodes.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Dataset paths ────────────────────────────────────────────────────────
    original_path: str
    working_copy_path: str
    dataset_name: str

    # ── Schema & missing values ──────────────────────────────────────────────
    schema_info: Dict[str, Any]
    missing_value_report: Dict[str, Any]
    missing_strategy: str            # 'mean' | 'median' | 'drop'

    # ── Analysis plan ────────────────────────────────────────────────────────
    analysis_plan: List[str]

    # ── Results ──────────────────────────────────────────────────────────────
    stats_json: str
    chart_paths: List[str]
    analysis_summary: str
    report_path: str
    report_html: str

    # ── Conversation / messages ──────────────────────────────────────────────
    messages: List[Dict[str, str]]
    user_confirmed: bool

    # ── Error handling ───────────────────────────────────────────────────────
    last_error: Optional[Any]
    retry_count: int
    error_log: List[Dict[str, Any]]

    # ── Control flow ─────────────────────────────────────────────────────────
    current_step: str
    step_status: str                 # 'pending' | 'running' | 'done' | 'error'
    awaiting_user_input: bool
    pending_user_question: str