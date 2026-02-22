"""
exceptions/__init__.py
Centralized exception management + Slack error reporter.
"""

from .base import (
    AgentBaseException,
    DataLoadError,
    SchemaInspectionError,
    MissingValueError,
    CleaningError,
    AnalysisPlanError,
    ToolExecutionError,
    VisualizationError,
    ReportGenerationError,
    SelfCorrectionError,
    LLMError,
    FileOperationError,
    SlackNotificationError,
)
from .slack_reporter import SlackErrorReporter

__all__ = [
    "AgentBaseException",
    "DataLoadError",
    "SchemaInspectionError",
    "MissingValueError",
    "CleaningError",
    "AnalysisPlanError",
    "ToolExecutionError",
    "VisualizationError",
    "ReportGenerationError",
    "SelfCorrectionError",
    "LLMError",
    "FileOperationError",
    "SlackNotificationError",
    "SlackErrorReporter",
]