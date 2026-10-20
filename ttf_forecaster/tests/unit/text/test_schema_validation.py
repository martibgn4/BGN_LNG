"""Schema and validation rules (SYNTHETIC events only)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ttf_forecaster.text.schema import (
    EXTRACTION_OUTPUT_FIELDS,
    EventRecord,
    ReviewStatus,
    event_json_schema,
    extraction_output_json_schema,
    make_event_id,
)
from ttf_forecaster.text.validation import (
    EventValidationError,
    assert_valid,
    quote_supports_number,
    validate_event,
    validate_json_schema,
)

from .conftest import SYN_DOC, t


def errors(ev: EventRecord, **kw: object) -> list[str]:
    return validate_event(ev, **kw).errors  # type: ignore[arg-type]


def test_baseline_event_is_valid(make_event):
    res = validate_event(make_event(), document_text=SYN_DOC, authorised_publishers={"SYNTHETIC"})
    assert res.ok, res.errors


def test_roundtrip_dict(make_event):
    ev = make_event(estimated_volume_impact_mcmd=12.5, impact_quote="reduced by 12.5 mcm/d")
    assert EventRecord.from_dict(ev.to_dict()) == ev
    with pytest.raises(ValueError, match="unknown fields"):
        EventRecord.from_dict({**ev.to_dict(), "price_forecast": 50.0})


def test_make_event_id_deterministic():
    h = "a" * 64
    assert make_event_id(h, 3) == make_event_id(h, 3) == "aaaaaaaaaaaaaaaa-003"
    with pytest.raises(ValueError):
        make_event_id("abc", 0)


@pytest.mark.parametrize("bad", [-0.1, 1.01, float("nan"), True])
def test_confidence_bounds(make_event, bad):
    assert any("confidence" in e for e in errors(make_event(confidence=bad)))


def test_bad_enum(make_event):
    assert any("event_type" in e for e in errors(make_event(event_type="price_spike")))


def test_naive_time_rejected(make_event):
    assert any("naive" in e for e in errors(make_event(expected_start=datetime(2026, 1, 10))))


def test_non_utc_aware_time_is_fine(make_event):
    cet = timezone(timedelta(hours=1))
    assert errors(make_event(publication_time=datetime(2026, 1, 10, 9, tzinfo=cet))) == []


def test_end_before_start(make_event):
    assert any("expected_end" in e for e in errors(make_event(expected_end=t(9))))


def test_publication_after_extraction(make_event):
    assert any("after extracted_at" in e for e in errors(make_event(publication_time=t(11))))


def test_human_review_cannot_be_disabled(make_event):
    errs = errors(make_event(human_review_required=False))
    assert any("human_review_required" in e for e in errs)


def test_unsupported_number_rejected(make_event):
    errs = errors(make_event(estimated_volume_impact_mcmd=12.5))
    assert any("unsupported" in e for e in errs)


def test_number_supported_by_quote(make_event):
    ev = make_event(
        estimated_volume_impact_mcmd=12.5, impact_quote="capacity reduced by 12.5 mcm/d"
    )
    assert errors(ev, document_text=SYN_DOC) == []


def test_quote_must_be_verbatim_span(make_event):
    ev = make_event(estimated_volume_impact_mcmd=12.5, impact_quote="cut by 12.5 mcm/d")
    assert any("verbatim" in e for e in errors(ev, document_text=SYN_DOC))


def test_number_supported_by_capacity_ref(make_event):
    ev = make_event(estimated_volume_impact_mcmd=30.0, capacity_mapping_ref="SYNTHETIC-CAP@v1:row7")
    assert errors(ev) == []
    assert errors(ev, capacity_refs={"SYNTHETIC-CAP@v1:row7"}) == []
    assert any("capacity registry" in e for e in errors(ev, capacity_refs={"other"}))


@pytest.mark.parametrize(
    ("quote", "value", "ok"),
    [
        ("reduced by 12.5 mcm/d", 12.5, True),
        ("reduced by 12,5 mcm/d", 12.5, True),
        ("reduced by 12.5mcm/d", 12.5, True),
        ("a 1,200 mcm/day cut", 1200.0, True),
        ("reduced by 12.5 mcm per day", 12.5, True),
        ("reduced by 12 mcm/d", 12.5, False),  # different number
        ("reduced by 12.5", 12.5, False),  # no unit
        ("reduced by 12.5 GWh/d", 12.5, False),  # other unit: conversion is not literal
        ("12.5 mcm in total", 12.5, False),  # a volume, not a rate
        ("10 mcm per day", 10.0, True),
        ("112.5 mcm/d", 12.5, False),  # substring of a bigger number
    ],
)
def test_quote_supports_number(quote, value, ok):
    assert quote_supports_number(quote, value) is ok


def test_review_fields_consistency(make_event):
    assert any("reviewer" in e for e in errors(make_event(review_status=ReviewStatus.APPROVED)))
    assert any(
        "pending" in e for e in errors(make_event(reviewer="alice", review_time=t(10, 10)))
    )
    self_review = make_event(
        version=1, parent_version=0, review_status=ReviewStatus.APPROVED,
        reviewer="synthetic-model", review_time=t(10, 10),
    )
    assert any("own output" in e for e in errors(self_review))


def test_unauthorised_publisher(make_event):
    assert any("authorised" in e for e in errors(make_event(), authorised_publishers={"X"}))


def test_assert_valid_raises(make_event):
    with pytest.raises(EventValidationError, match="confidence"):
        assert_valid(make_event(confidence=2.0))


def test_json_schema_accepts_record_and_rejects_false_review(make_event):
    pytest.importorskip("jsonschema")
    schema = event_json_schema()
    ev = make_event()
    assert validate_json_schema(ev.to_dict(), schema) == []
    bad = replace(ev, human_review_required=False).to_dict()
    assert validate_json_schema(bad, schema)


def test_extraction_output_schema_has_no_provenance():
    pytest.importorskip("jsonschema")
    s = extraction_output_json_schema()
    item = s["properties"]["events"]["items"]
    assert set(item["required"]) == set(EXTRACTION_OUTPUT_FIELDS)
    assert "reviewer" not in item["properties"] and "document_sha256" not in item["properties"]
    sample = {"events": [{
        "event_time": None, "event_type": "outage", "asset": None, "region": None,
        "estimated_volume_impact_mcmd": None, "impact_quote": None, "expected_start": None,
        "expected_end": None, "timing_quote": None, "directional_effect": "uncertain",
        "confidence": 0.3,
    }]}
    assert validate_json_schema(sample, s) == []
