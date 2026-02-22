"""
exceptions/slack_reporter.py
Sends structured error payloads to a Slack channel via Incoming Webhook.
Set SLACK_WEBHOOK_URL in your .env file.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from .base import AgentBaseException, SlackNotificationError

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_CHANNEL = os.getenv("SLACK_ERROR_CHANNEL", "#data-agent-errors")


class SlackErrorReporter:
    """
    Posts exception details to a Slack channel using Block Kit.
    Safe to instantiate even when no webhook is configured —
    it will log a warning and skip posting.
    """

    def __init__(self, webhook_url: str = SLACK_WEBHOOK_URL, channel: str = SLACK_CHANNEL):
        self.webhook_url = webhook_url
        self.channel = channel

    # ------------------------------------------------------------------
    def report(self, exc: AgentBaseException, extra_context: Optional[dict] = None) -> bool:
        """
        Post the exception to Slack.
        Returns True on success, False on failure (never raises).
        """
        if not self.webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification.")
            return False

        payload = self._build_payload(exc, extra_context or {})
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Slack error report sent for {exc.__class__.__name__}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    # ------------------------------------------------------------------
    def report_success(self, dataset_name: str, report_url: Optional[str] = None) -> bool:
        """Post a success notification to Slack."""
        if not self.webhook_url:
            return False
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": "#00C9A7",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "✅ Analysis Complete!"}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Dataset:* {dataset_name}\nStatus: *Completed Successfully*"}
                    }
                ]
            }]
        }
        if report_url:
            payload["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{report_url}|View Performance Report>"}
            })

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack success report: {e}")
            return False

    # ------------------------------------------------------------------
    def report_from_state(self, state: dict) -> bool:
        """Convenience helper — post the last error stored in agent state."""
        last_error = state.get("last_error")
        if last_error is None:
            return False
        if isinstance(last_error, AgentBaseException):
            return self.report(last_error)
        # Plain exception fallback
        generic = AgentBaseException(str(last_error))
        return self.report(generic)

    # ------------------------------------------------------------------
    def _build_payload(self, exc: AgentBaseException, extra: dict) -> dict:
        color = self._severity_color(exc)
        ts = exc.timestamp
        ctx_text = (
            "\n".join(f"• *{k}*: `{v}`" for k, v in exc.context.items())
            if exc.context
            else "_none_"
        )
        extra_text = (
            "\n".join(f"• *{k}*: `{v}`" for k, v in extra.items()) if extra else "_none_"
        )
        tb = exc.traceback_str[:800] + "…" if len(exc.traceback_str) > 800 else exc.traceback_str

        return {
            "channel": self.channel,
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"🚨 Data Agent Error — {exc.__class__.__name__}",
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Type:*\n`{exc.__class__.__name__}`"},
                                {"type": "mrkdwn", "text": f"*Timestamp:*\n`{ts}`"},
                            ],
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Message:*\n> {exc.message}"},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Context:*\n{ctx_text}"},
                        },
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Extra:*\n{extra_text}"},
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Traceback (truncated):*\n```{tb}```" if tb else "_no traceback_",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Autonomous Data Analysis Agent | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                                }
                            ],
                        },
                    ],
                }
            ],
        }

    @staticmethod
    def _severity_color(exc: AgentBaseException) -> str:
        critical = ("LLMError", "SelfCorrectionError", "ReportGenerationError")
        warning = ("MissingValueError", "CleaningError", "VisualizationError")
        if exc.__class__.__name__ in critical:
            return "#FF4444"
        if exc.__class__.__name__ in warning:
            return "#FFA500"
        return "#36A64F"