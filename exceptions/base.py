"""
Base exception classes for the Data Analysis Agent.
All custom exceptions inherit from AgentBaseException for centralized handling.
"""

import traceback
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentBaseException(Exception):
    """Root exception for all agent errors. Captures context & timestamp."""

    def __init__(self, message: str, context: dict = None, original: Exception = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original = original
        self.timestamp = datetime.utcnow().isoformat()
        self.traceback_str = traceback.format_exc() if original else ""

        logger.error(
            f"[{self.__class__.__name__}] {message} | context={context} | ts={self.timestamp}"
        )

    def to_dict(self) -> dict:
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "traceback": self.traceback_str,
        }


# ── Data lifecycle ──────────────────────────────────────────────────────────

class DataLoadError(AgentBaseException):
    """Raised when a dataset cannot be loaded or parsed."""


class FileOperationError(AgentBaseException):
    """Raised when file copy / read / write operations fail."""


class SchemaInspectionError(AgentBaseException):
    """Raised when schema introspection fails."""


# ── Cleaning ────────────────────────────────────────────────────────────────

class MissingValueError(AgentBaseException):
    """Raised when missing-value detection or reporting fails."""


class CleaningError(AgentBaseException):
    """Raised when a cleaning operation cannot be completed."""


# ── Agent orchestration ─────────────────────────────────────────────────────

class AnalysisPlanError(AgentBaseException):
    """Raised when the LLM fails to produce a valid analysis plan."""


class ToolExecutionError(AgentBaseException):
    """Raised when a LangChain tool execution fails."""


class SelfCorrectionError(AgentBaseException):
    """Raised when self-correction retries are exhausted."""


# ── Output ──────────────────────────────────────────────────────────────────

class VisualizationError(AgentBaseException):
    """Raised when chart generation fails."""


class ReportGenerationError(AgentBaseException):
    """Raised when HTML report generation fails."""


# ── External services ────────────────────────────────────────────────────────

class LLMError(AgentBaseException):
    """Raised when the Groq / LangChain LLM call fails."""


class SlackNotificationError(AgentBaseException):
    """Raised when sending an error alert to Slack fails."""