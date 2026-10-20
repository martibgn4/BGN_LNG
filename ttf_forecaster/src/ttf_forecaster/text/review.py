"""Human-review workflow for extracted events. Append-only: nothing is ever overwritten.

Every action (approve, reject, edit) appends a new :class:`EventRecord` version carrying the
reviewer, the review time and a pointer to its parent version. The original extraction (version 0)
is kept forever, so the diff between what the model said and what a human accepted is auditable
and can itself be evaluated (extraction precision/recall).

Only events whose **latest** version is ``approved`` or ``edited`` pass to features.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

from ttf_forecaster.text.schema import (
    PASSING_STATUSES,
    PROVENANCE_FIELDS,
    REVIEW_FIELDS,
    EventRecord,
    ReviewStatus,
)
from ttf_forecaster.text.validation import EventValidationError, assert_valid


class ReviewError(EventValidationError):
    """An illegal review action (unknown event, bad reviewer, time going backwards...)."""


class ReviewLedger:
    """Append-only store of event versions keyed by ``event_id``.

    Args:
        validation_kwargs: forwarded to :func:`validation.validate_event` for every version
            (``authorised_publishers``, ``capacity_refs``...).
    """

    def __init__(self, **validation_kwargs: Any) -> None:
        self._versions: dict[str, list[EventRecord]] = {}
        self._vkw = validation_kwargs

    # -- reading ---------------------------------------------------------------------------
    def __contains__(self, event_id: object) -> bool:
        return event_id in self._versions

    def __len__(self) -> int:
        return len(self._versions)

    def __iter__(self) -> Iterator[str]:
        return iter(self._versions)

    def history(self, event_id: str) -> tuple[EventRecord, ...]:
        """All versions of an event, oldest first."""
        try:
            return tuple(self._versions[event_id])
        except KeyError as exc:
            raise ReviewError(f"unknown event_id {event_id!r}") from exc

    def latest(self, event_id: str) -> EventRecord:
        return self.history(event_id)[-1]

    def original(self, event_id: str) -> EventRecord:
        """The unedited extraction (version 0)."""
        return self.history(event_id)[0]

    def passing(self) -> list[EventRecord]:
        """Latest version of every event whose latest status is approved or edited."""
        return [v[-1] for v in self._versions.values() if v[-1].review_status in PASSING_STATUSES]

    def all_versions(self) -> list[EventRecord]:
        return [r for v in self._versions.values() for r in v]

    # -- writing ---------------------------------------------------------------------------
    def add_extraction(self, event: EventRecord) -> EventRecord:
        """Register a fresh extraction. Must be version 0, pending, unreviewed, and valid."""
        if event.event_id in self._versions:
            raise ReviewError(f"event {event.event_id!r} already registered; use edit()")
        if event.version != 0 or event.review_status != ReviewStatus.PENDING:
            raise ReviewError("a new extraction must be version 0 with review_status pending")
        assert_valid(event, **self._vkw)
        self._versions[event.event_id] = [event]
        return event

    def _append(
        self,
        event_id: str,
        reviewer: str,
        at: datetime,
        status: ReviewStatus,
        note: str | None,
        changes: Mapping[str, Any] | None = None,
    ) -> EventRecord:
        prev = self.latest(event_id)
        if not reviewer or not reviewer.strip():
            raise ReviewError("reviewer must be a named human")
        if at.tzinfo is None:
            raise ReviewError("review time must be tz-aware")
        if prev.review_time is not None and at < prev.review_time:
            raise ReviewError(
                f"review time {at.isoformat()} precedes previous review "
                f"{prev.review_time.isoformat()}"
            )
        new = replace(
            prev,
            **dict(changes or {}),
            version=prev.version + 1,
            parent_version=prev.version,
            reviewer=reviewer,
            review_status=status,
            review_time=at,
            review_note=note,
        )
        assert_valid(new, **self._vkw)
        self._versions[event_id].append(new)
        return new

    def approve(
        self, event_id: str, reviewer: str, at: datetime, note: str | None = None
    ) -> EventRecord:
        """Accept the latest content unchanged."""
        return self._append(event_id, reviewer, at, ReviewStatus.APPROVED, note)

    def reject(
        self, event_id: str, reviewer: str, at: datetime, note: str | None = None
    ) -> EventRecord:
        """Reject: the event stops passing from ``at`` onwards; its history is kept."""
        return self._append(event_id, reviewer, at, ReviewStatus.REJECTED, note)

    def edit(
        self,
        event_id: str,
        reviewer: str,
        at: datetime,
        changes: Mapping[str, Any],
        note: str | None = None,
    ) -> EventRecord:
        """Store a corrected version. Provenance and review fields cannot be edited; the edited
        version must pass validation (an edited number still needs a quote or a capacity ref)."""
        if not changes:
            raise ReviewError("edit() needs at least one changed field; use approve()")
        forbidden = set(changes) & (PROVENANCE_FIELDS | REVIEW_FIELDS)
        if forbidden:
            raise ReviewError(f"fields {sorted(forbidden)} are fixed at extraction / set by review")
        unknown = set(changes) - set(EventRecord.__dataclass_fields__)
        if unknown:
            raise ReviewError(f"unknown fields {sorted(unknown)}")
        return self._append(event_id, reviewer, at, ReviewStatus.EDITED, note, changes)
