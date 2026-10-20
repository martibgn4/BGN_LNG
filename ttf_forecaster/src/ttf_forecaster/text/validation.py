"""Validation gate for :class:`~ttf_forecaster.text.schema.EventRecord`.

An event that fails here never reaches review, let alone features.
Rules (``docs/llm_event_schema.md``):

1. Enumerations are valid; ``confidence`` is a finite number in [0, 1].
2. Every timestamp is tz-aware; ``publication_time <= extracted_at <= review_time``.
3. ``expected_end >= expected_start`` when both are stated.
4. A non-null ``estimated_volume_impact_mcmd`` needs **either** an ``impact_quote`` that literally
   contains the number followed by an accepted unit, **or** a ``capacity_mapping_ref`` (checked
   against the registry of known capacity-table refs when one is supplied).
5. ``human_review_required`` cannot be ``False`` in v1.
6. Provenance is complete; the publisher is on the authorised list when one is supplied; quotes
   are verbatim substrings of the document when the document text is supplied.
7. Review fields are consistent with ``review_status``; a model may not be its own reviewer.

Errors block the record; warnings are surfaced to the reviewer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ttf_forecaster.data.base import DataError, Entitlement
from ttf_forecaster.text.schema import (
    TIME_FIELDS,
    DirectionalEffect,
    EventRecord,
    EventType,
    ReviewStatus,
)

DEFAULT_MCMD_UNITS: tuple[str, ...] = (
    "million cubic metres per day",
    "million cubic meters per day",
    "mcm per day",
    "mcm/day",
    "mcm/d",
    "mcmd",
    "mcm",
)
"""Unit spellings accepted after the number in ``impact_quote``. Overridable from
``configs/text.yaml`` (``validation.accepted_mcmd_units``). Bare ``mcm`` only counts when a per-day
marker follows it (``"10 mcm per day"``); ``"10 mcm"`` is a volume, not a rate. ``MSm3/d``
(Gassco, standard conditions) and energy units (GWh/d, kWh/h in REMIT UMMs) are deliberately not
accepted: they need a documented conversion, which is not literal support."""

_PER_DAY = re.compile(r"^\s*(?:/\s*d(?:ay)?\b|per\s+day\b|a\s+day\b|daily\b)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:[,.  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EventValidationError(DataError):
    """An event record breaks a validation rule. Message lists every violation."""


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_event`. ``ok`` is True iff there are no errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _number_readings(token: str) -> set[float]:
    """All plausible numeric readings of a token (decimal comma vs thousands separators)."""
    t = token.replace(" ", " ").replace(" ", "")
    out: set[float] = set()
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        thou = "." if dec == "," else ","
        out.add(float(t.replace(thou, "").replace(dec, ".")))
    elif "," in t or "." in t:
        sep = "," if "," in t else "."
        out.add(float(t.replace(sep, "")))  # thousands separator
        if t.count(sep) == 1:
            out.add(float(t.replace(sep, ".")))  # decimal separator
    else:
        out.add(float(t))
    return out


def quote_supports_number(
    quote: str, value: float, units: Sequence[str] = DEFAULT_MCMD_UNITS
) -> bool:
    """True iff ``quote`` literally contains ``value`` immediately followed by an accepted unit.

    ``"Capacity reduced by 12.5 mcm/d"`` supports 12.5; ``"reduced by 12.5"`` (no unit),
    ``"12.5 GWh/d"`` (other unit: conversion is not literal support) and ``"12 mcm/d"``
    (different number) do not.
    """
    units_sorted = sorted({u.lower() for u in units}, key=len, reverse=True)
    for m in _NUMBER.finditer(quote):
        readings = _number_readings(m.group(1))
        if not any(math.isclose(r, value, rel_tol=1e-9, abs_tol=1e-9) for r in readings):
            continue
        rest = quote[m.end():].lstrip().lower()
        for u in units_sorted:
            if rest.startswith(u) and (len(rest) == len(u) or not rest[len(u)].isalnum()):
                if u == "mcm" and not _PER_DAY.match(rest[len(u):]):
                    break  # bare volume, not a rate
                return True
    return False


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def validate_event(
    event: EventRecord,
    *,
    authorised_publishers: Collection[str] | None = None,
    capacity_refs: Collection[str] | None = None,
    document_text: str | None = None,
    accepted_units: Sequence[str] = DEFAULT_MCMD_UNITS,
) -> ValidationResult:
    """Check every rule; return all errors and warnings (never stops at the first).

    Args:
        event: record to check.
        authorised_publishers: if given, ``source_publisher`` must be in it.
        capacity_refs: if given, a ``capacity_mapping_ref`` must be one of these registered refs.
        document_text: if given, ``impact_quote``/``timing_quote`` must be verbatim substrings
            (whitespace-normalised).
        accepted_units: unit spellings accepted after the number in ``impact_quote``.
    """
    res = ValidationResult()
    err, warn = res.errors.append, res.warnings.append

    # 1. enumerations and confidence
    for name, enum in (
        ("event_type", EventType),
        ("directional_effect", DirectionalEffect),
        ("review_status", ReviewStatus),
    ):
        v = getattr(event, name)
        if not isinstance(v, enum):
            try:
                enum(v)
            except ValueError:
                err(f"{name}: {v!r} not in {[e.value for e in enum]}")
    c = event.confidence
    if isinstance(c, bool) or not isinstance(c, int | float) or not math.isfinite(c):
        err(f"confidence: must be a finite number, got {c!r}")
    elif not 0.0 <= c <= 1.0:
        err(f"confidence: {c} outside [0, 1]")
    if event.source_entitlement not in {e.value for e in Entitlement}:
        err(f"source_entitlement: {event.source_entitlement!r} not a data.base.Entitlement value")

    # 2. times
    aware: dict[str, datetime] = {}
    for name in TIME_FIELDS:
        v = getattr(event, name)
        if v is None:
            continue
        if not isinstance(v, datetime):
            err(f"{name}: expected datetime, got {type(v).__name__}")
        elif v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            err(f"{name}: timezone-naive; store tz-aware times")
        else:
            aware[name] = v
    if event.publication_time is None:
        err("publication_time: required")
    if event.extracted_at is None:
        err("extracted_at: required")
    pub, ext, rev = (aware.get(k) for k in ("publication_time", "extracted_at", "review_time"))
    if pub and ext and pub > ext:
        err(f"publication_time {pub.isoformat()} after extracted_at {ext.isoformat()}")
    if ext and rev and rev < ext:
        err(f"review_time {rev.isoformat()} before extracted_at {ext.isoformat()}")

    # 3. expected window
    st, en = aware.get("expected_start"), aware.get("expected_end")
    if st and en and en < st:
        err(f"expected_end {en.isoformat()} before expected_start {st.isoformat()}")
    if (st or en) and not event.timing_quote:
        warn("expected_start/expected_end set without timing_quote; reviewer must check timing")

    # 4. numeric impact
    vol = event.estimated_volume_impact_mcmd
    if vol is not None:
        if isinstance(vol, bool) or not isinstance(vol, int | float) or not math.isfinite(vol):
            err(f"estimated_volume_impact_mcmd: must be finite or null, got {vol!r}")
        elif vol < 0:
            err(f"estimated_volume_impact_mcmd: {vol} < 0 (state offline volume as positive)")
        else:
            quoted = bool(event.impact_quote) and quote_supports_number(
                event.impact_quote or "", float(vol), accepted_units
            )
            mapped = bool(event.capacity_mapping_ref)
            ref = event.capacity_mapping_ref
            if mapped and capacity_refs is not None and ref not in capacity_refs:
                err(f"capacity_mapping_ref {ref!r} not in the capacity registry")
                mapped = False
            if not (quoted or mapped):
                err(
                    f"estimated_volume_impact_mcmd={vol} unsupported: impact_quote does not "
                    "literally contain the number with an accepted unit and there is no "
                    "capacity_mapping_ref; set it to null"
                )
    elif event.impact_quote:
        warn("impact_quote present but estimated_volume_impact_mcmd is null")

    # 5. human review
    if event.human_review_required is not True:
        err("human_review_required must be True in schema v1")

    # 6. provenance
    for name in (
        "event_id",
        "source_publisher",
        "source_reference",
        "extraction_model_id",
        "extraction_model_version",
        "prompt_version",
    ):
        if not getattr(event, name):
            err(f"{name}: required provenance field is empty")
    if not _HEX64.match(event.document_sha256 or ""):
        err("document_sha256: expected 64 lowercase hex characters")
    if authorised_publishers is not None and event.source_publisher not in authorised_publishers:
        err(f"source_publisher {event.source_publisher!r} is not an authorised source")
    if document_text is not None:
        doc = _norm_ws(document_text)
        for name in ("impact_quote", "timing_quote"):
            q = getattr(event, name)
            if q and _norm_ws(q) not in doc:
                err(f"{name} is not a verbatim span of the source document")

    # 7. review consistency
    status = event.review_status
    if event.version < 0:
        err("version must be >= 0")
    if event.version == 0 and event.parent_version is not None:
        err("version 0 cannot have a parent_version")
    parent = event.parent_version
    if event.version > 0 and (parent is None or parent >= event.version):
        err("version > 0 needs parent_version < version")
    if status == ReviewStatus.PENDING:
        if event.reviewer or event.review_time:
            err("pending event cannot carry reviewer/review_time")
    else:
        if not event.reviewer:
            err(f"review_status {status} requires a reviewer")
        if event.review_time is None:
            err(f"review_status {status} requires review_time")
        if event.reviewer and event.reviewer == event.extraction_model_id:
            err("the extraction model cannot review its own output")
    return res


def assert_valid(event: EventRecord, **kwargs: object) -> EventRecord:
    """Return ``event`` unchanged or raise :class:`EventValidationError` listing every error."""
    res = validate_event(event, **kwargs)  # type: ignore[arg-type]
    if not res.ok:
        label = f"event {event.event_id or '?'} v{event.version}: "
        raise EventValidationError(label + "; ".join(res.errors))
    return event


def validate_json_schema(record: dict[str, object], schema: dict[str, object]) -> list[str]:
    """Structural JSON-Schema check (optional ``jsonschema`` dependency). Returns error messages.

    Raises ``ImportError`` with an explanation if ``jsonschema`` is not installed; it is not a
    declared project dependency (see report).
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("validate_json_schema needs the optional 'jsonschema' package") from exc
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
        for e in validator.iter_errors(record)
    ]
