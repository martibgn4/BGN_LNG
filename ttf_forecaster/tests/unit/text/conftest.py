"""SYNTHETIC event fixtures. Every document, asset and number here is invented for tests only."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from ttf_forecaster.text.schema import DirectionalEffect, EventRecord, EventType, make_event_id

SYN_DOC = (
    "SYNTHETIC test message. Unplanned outage at SYNTHETIC-FIELD-A. "
    "Available capacity reduced by 12.5 mcm/d. Expected back on 2026-01-12 06:00 UTC."
)
SYN_HASH = hashlib.sha256(SYN_DOC.encode()).hexdigest()


def t(day: int, hour: int = 0) -> datetime:
    """SYNTHETIC UTC timestamp in January 2026."""
    return datetime(2026, 1, day, hour, tzinfo=UTC)


@pytest.fixture
def make_event() -> Callable[..., EventRecord]:
    def _make(index: int = 0, **overrides: Any) -> EventRecord:
        base: dict[str, Any] = {
            "event_type": EventType.OUTAGE,
            "directional_effect": DirectionalEffect.BULLISH,
            "confidence": 0.8,
            "publication_time": t(10, 8),
            "event_time": t(10, 7),
            "asset": "SYNTHETIC-FIELD-A",
            "region": "NWE",
            "estimated_volume_impact_mcmd": None,
            "expected_start": t(10, 7),
            "expected_end": t(12, 6),
            "timing_quote": "Expected back on 2026-01-12 06:00 UTC.",
            "event_id": make_event_id(SYN_HASH, index),
            "document_sha256": SYN_HASH,
            "source_publisher": "SYNTHETIC",
            "source_reference": f"synthetic/doc/{index}",
            "source_entitlement": "synthetic",
            "extracted_at": t(10, 9),
            "extraction_model_id": "synthetic-model",
            "extraction_model_version": "0",
            "prompt_version": "event_extraction_v0",
        }
        base.update(overrides)
        return EventRecord(**base)

    return _make
