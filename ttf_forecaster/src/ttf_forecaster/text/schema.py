"""Event record for LLM-assisted extraction from authorised text (``docs/llm_event_schema.md``).

The LLM is **not** a price forecaster. It proposes structured, reviewable event records from text
the desk is authorised to process. Every record carries full provenance (which document, which
model, which prompt, which reviewer) and a review status. Numeric impacts are ``None`` unless the
source states them verbatim (``impact_quote``) or an independently maintained asset-capacity table
supports them (``capacity_mapping_ref``).

Records are immutable: review actions create new versions (``review.py``), never edit in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "0.1"


class EventType(StrEnum):
    OUTAGE = "outage"
    MAINTENANCE = "maintenance"
    POLICY = "policy"
    WEATHER = "weather"
    LNG_DIVERSION = "lng_diversion"
    DEMAND = "demand"


class DirectionalEffect(StrEnum):
    """Direction for TTF as stated or clearly implied by the text. Not a forecast."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    UNCERTAIN = "uncertain"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


PASSING_STATUSES: frozenset[ReviewStatus] = frozenset({ReviewStatus.APPROVED, ReviewStatus.EDITED})
"""Only these statuses may feed features (``review.py``, ``to_features.py``)."""

TIME_FIELDS: tuple[str, ...] = (
    "event_time",
    "publication_time",
    "expected_start",
    "expected_end",
    "extracted_at",
    "review_time",
)

PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "event_id",
        "document_sha256",
        "source_publisher",
        "source_reference",
        "source_entitlement",
        "publication_time",
        "extracted_at",
        "extraction_model_id",
        "extraction_model_version",
        "prompt_version",
    }
)
"""Fields fixed at extraction. A reviewer edit may not change them (``review.py``)."""

REVIEW_FIELDS: frozenset[str] = frozenset(
    {"version", "parent_version", "reviewer", "review_status", "review_time", "review_note"}
)
"""Fields set only by the review workflow."""


@dataclass(frozen=True)
class EventRecord:
    """One extracted market event, one version.

    Content (what the text says):
        event_time: when the occurrence happened/was stated to happen; ``None`` if not stated.
        publication_time: when the *source* published the document (tz-aware). Not our ingest time.
        event_type: one of :class:`EventType`.
        asset: asset name as written in the source (e.g. a field, terminal, pipeline); ``None`` if
            not stated. Mapping to an internal asset id is the capacity table's job, not the LLM's.
        region: region/hub code as written or unambiguously implied; ``None`` if not stated.
        estimated_volume_impact_mcmd: offline/affected volume in mcm/d. ``None`` unless supported
            by ``impact_quote`` (literal number + unit) or ``capacity_mapping_ref``.
        expected_start / expected_end: stated timing; ``None`` if not stated.
        directional_effect: one of :class:`DirectionalEffect`; ``uncertain`` when not clear.
        confidence: extractor's confidence the record faithfully reflects the text, in [0, 1]. It
            is *not* a probability of a price move.

    Provenance:
        event_id: stable id (``make_event_id``) shared by all versions of the event.
        document_sha256: SHA-256 of the exact document bytes the model saw.
        source_publisher: publisher of the document (e.g. TSO/operator name); from the source list.
        source_reference: archive key / message id of the document in *our* store.
        source_entitlement: ``data.base.Entitlement`` value of the source.
        extracted_at: when extraction ran (tz-aware).
        extraction_model_id / extraction_model_version: model identity; ``"human"`` for manual
            entry.
        prompt_version: id of the prompt template (e.g. ``event_extraction_v0``).
        impact_quote: verbatim span supporting any non-null number.
        capacity_mapping_ref: key into an independently maintained, versioned asset-capacity table.
        timing_quote: verbatim span supporting expected_start/expected_end, if any.
        human_review_required: always ``True`` in v1.

    Review:
        version: 0 for the original extraction; each review action appends ``version + 1``.
        parent_version: version this one was derived from (``None`` for version 0).
        reviewer, review_status, review_time, review_note.
    """

    # --- content ---
    event_type: EventType
    directional_effect: DirectionalEffect
    confidence: float
    publication_time: datetime
    event_time: datetime | None = None
    asset: str | None = None
    region: str | None = None
    estimated_volume_impact_mcmd: float | None = None
    expected_start: datetime | None = None
    expected_end: datetime | None = None
    # --- provenance ---
    event_id: str = ""
    document_sha256: str = ""
    source_publisher: str = ""
    source_reference: str = ""
    source_entitlement: str = ""
    extracted_at: datetime | None = None
    extraction_model_id: str = ""
    extraction_model_version: str = ""
    prompt_version: str = ""
    impact_quote: str | None = None
    capacity_mapping_ref: str | None = None
    timing_quote: str | None = None
    human_review_required: bool = True
    # --- review ---
    version: int = 0
    parent_version: int | None = None
    reviewer: str | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_time: datetime | None = None
    review_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: enums as values, datetimes as ISO-8601 strings with offset."""
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            elif isinstance(v, StrEnum):
                out[k] = v.value
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRecord:
        """Inverse of :meth:`to_dict`. Unknown keys are an error (no silent schema drift)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"EventRecord: unknown fields {sorted(unknown)}")
        kw: dict[str, Any] = dict(data)
        for name in TIME_FIELDS:
            v = kw.get(name)
            if isinstance(v, str):
                kw[name] = datetime.fromisoformat(v)
        if "event_type" in kw:
            kw["event_type"] = EventType(kw["event_type"])
        if "directional_effect" in kw:
            kw["directional_effect"] = DirectionalEffect(kw["directional_effect"])
        if "review_status" in kw:
            kw["review_status"] = ReviewStatus(kw["review_status"])
        return cls(**kw)


def make_event_id(document_sha256: str, index: int) -> str:
    """Deterministic event id: ``<first 16 hex of document hash>-<index within document>``."""
    if len(document_sha256) < 16:
        raise ValueError("document_sha256 must be a full hex digest")
    if index < 0:
        raise ValueError("index must be >= 0")
    return f"{document_sha256[:16]}-{index:03d}"


# ---------------------------------------------------------------------------------------------
# JSON Schema (draft 2020-12) for the model's output and for stored records
# ---------------------------------------------------------------------------------------------

_TS = {"type": "string", "format": "date-time"}
_TS_NULL = {"type": ["string", "null"], "format": "date-time"}
_STR_NULL = {"type": ["string", "null"]}


def event_json_schema() -> dict[str, Any]:
    """JSON Schema of a stored :class:`EventRecord` (all keys required; nullables explicit).

    Structural only. Cross-field rules (quote must contain the number, end >= start, tz-aware
    times, publication <= extraction) live in ``validation.py``; JSON Schema cannot express them.
    """
    props: dict[str, Any] = {
        "event_type": {"enum": [e.value for e in EventType]},
        "directional_effect": {"enum": [e.value for e in DirectionalEffect]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "publication_time": _TS,
        "event_time": _TS_NULL,
        "asset": _STR_NULL,
        "region": _STR_NULL,
        "estimated_volume_impact_mcmd": {"type": ["number", "null"], "minimum": 0.0},
        "expected_start": _TS_NULL,
        "expected_end": _TS_NULL,
        "event_id": {"type": "string", "minLength": 1},
        "document_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source_publisher": {"type": "string", "minLength": 1},
        "source_reference": {"type": "string", "minLength": 1},
        "source_entitlement": {"type": "string", "minLength": 1},
        "extracted_at": _TS,
        "extraction_model_id": {"type": "string", "minLength": 1},
        "extraction_model_version": {"type": "string", "minLength": 1},
        "prompt_version": {"type": "string", "minLength": 1},
        "impact_quote": _STR_NULL,
        "capacity_mapping_ref": _STR_NULL,
        "timing_quote": _STR_NULL,
        "human_review_required": {"const": True},
        "version": {"type": "integer", "minimum": 0},
        "parent_version": {"type": ["integer", "null"], "minimum": 0},
        "reviewer": _STR_NULL,
        "review_status": {"enum": [s.value for s in ReviewStatus]},
        "review_time": _TS_NULL,
        "review_note": _STR_NULL,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"ttf_forecaster/text/event_record/{SCHEMA_VERSION}",
        "title": "EventRecord",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(props),
        "properties": props,
    }


EXTRACTION_OUTPUT_FIELDS: tuple[str, ...] = (
    "event_time",
    "event_type",
    "asset",
    "region",
    "estimated_volume_impact_mcmd",
    "impact_quote",
    "expected_start",
    "expected_end",
    "timing_quote",
    "directional_effect",
    "confidence",
)
"""Fields the *model* fills. Provenance and review fields are set by our code, never the model."""


def extraction_output_json_schema() -> dict[str, Any]:
    """JSON Schema for the model's raw output: ``{"events": [...]}`` with content fields only."""
    full = event_json_schema()["properties"]
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(EXTRACTION_OUTPUT_FIELDS),
        "properties": {k: full[k] for k in EXTRACTION_OUTPUT_FIELDS},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "EventExtractionOutput",
        "type": "object",
        "additionalProperties": False,
        "required": ["events"],
        "properties": {"events": {"type": "array", "items": item}},
    }
