"""
utils/session.py
Manages Streamlit session-state for agent lifecycle.
"""

import streamlit as st
from pathlib import Path


def init_session():
    """Initialize all session-state keys on first run."""
    defaults = {
        "agent_state": {},
        "graph": None,
        "messages": [],
        "stage": "upload",        # upload | missing_strategy | running | done | error
        "working_copy_path": None,
        "original_path": None,
        "dataset_name": "Dataset",
        "missing_strategy": None,
        "report_path": None,
        "chart_paths": [],
        "error_log": [],
        "run_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def push_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session()