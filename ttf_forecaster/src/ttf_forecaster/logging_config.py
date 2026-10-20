"""Structured (JSON-lines) logging without extra dependencies.

Usage::

    from ttf_forecaster.logging_config import get_logger
    log = get_logger(__name__)
    log.info("fold_done", extra={"fields": {"fold": 3, "rmse": 0.012}})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_REDACT_KEYS = ("secret", "password", "token", "api_key", "client_id", "credential")


class JsonFormatter(logging.Formatter):
    """One JSON object per line; keys containing secret-like names are redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                redact = any(s in key.lower() for s in _REDACT_KEYS)
                payload[key] = "***" if redact else value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(level: int = logging.INFO) -> None:
    """Install the JSON handler on the package root logger (idempotent)."""
    root = logging.getLogger("ttf_forecaster")
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a package logger; configure() is called lazily on first use."""
    configure()
    return logging.getLogger(name if name.startswith("ttf_forecaster") else f"ttf_forecaster.{name}")
