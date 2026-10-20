"""Append-only review workflow (SYNTHETIC events only)."""

from __future__ import annotations

import pytest

from ttf_forecaster.text.review import ReviewError, ReviewLedger
from ttf_forecaster.text.schema import ReviewStatus
from ttf_forecaster.text.validation import EventValidationError

from .conftest import t


def test_only_approved_or_edited_pass(make_event):
    led = ReviewLedger()
    for i in range(3):
        led.add_extraction(make_event(i))
    ids = list(led)
    led.approve(ids[0], "alice", t(10, 10))
    led.reject(ids[1], "alice", t(10, 10))
    assert [e.event_id for e in led.passing()] == [ids[0]]
    led.edit(ids[2], "bob", t(10, 11), {"region": "UK"})
    assert {e.event_id for e in led.passing()} == {ids[0], ids[2]}


def test_edit_never_overwrites_original(make_event):
    led = ReviewLedger()
    ev = led.add_extraction(make_event())
    new = led.edit(
        ev.event_id, "bob", t(10, 11),
        {"estimated_volume_impact_mcmd": 12.5, "impact_quote": "reduced by 12.5 mcm/d"},
        note="number stated in message",
    )
    hist = led.history(ev.event_id)
    assert len(hist) == 2 and hist[0] == ev
    assert led.original(ev.event_id).estimated_volume_impact_mcmd is None
    assert (new.version, new.parent_version, new.reviewer) == (1, 0, "bob")
    assert new.review_status == ReviewStatus.EDITED and new.review_time == t(10, 11)


def test_edit_cannot_invent_numbers(make_event):
    led = ReviewLedger()
    ev = led.add_extraction(make_event())
    with pytest.raises(EventValidationError, match="unsupported"):
        led.edit(ev.event_id, "bob", t(10, 11), {"estimated_volume_impact_mcmd": 20.0})
    assert len(led.history(ev.event_id)) == 1


def test_edit_cannot_touch_provenance(make_event):
    led = ReviewLedger()
    ev = led.add_extraction(make_event())
    with pytest.raises(ReviewError, match="fixed"):
        led.edit(ev.event_id, "bob", t(10, 11), {"publication_time": t(9)})
    with pytest.raises(ReviewError, match="fixed"):
        led.edit(ev.event_id, "bob", t(10, 11), {"review_status": "approved"})


def test_review_guards(make_event):
    led = ReviewLedger()
    ev = led.add_extraction(make_event())
    with pytest.raises(ReviewError):
        led.add_extraction(ev)
    with pytest.raises(ReviewError, match="named human"):
        led.approve(ev.event_id, " ", t(10, 10))
    with pytest.raises(EventValidationError, match="own output"):
        led.approve(ev.event_id, "synthetic-model", t(10, 10))
    with pytest.raises(EventValidationError, match="before extracted_at"):
        led.approve(ev.event_id, "alice", t(10, 8))
    led.approve(ev.event_id, "alice", t(10, 12))
    with pytest.raises(ReviewError, match="precedes"):
        led.reject(ev.event_id, "alice", t(10, 11))
    with pytest.raises(ReviewError, match="unknown"):
        led.latest("nope")


def test_new_extraction_must_be_pending_v0(make_event):
    led = ReviewLedger()
    with pytest.raises(ReviewError):
        led.add_extraction(make_event(review_status=ReviewStatus.APPROVED, reviewer="a",
                                      review_time=t(10, 10)))


def test_ledger_enforces_authorised_sources(make_event):
    led = ReviewLedger(authorised_publishers={"SOMEONE_ELSE"})
    with pytest.raises(EventValidationError, match="authorised"):
        led.add_extraction(make_event())
