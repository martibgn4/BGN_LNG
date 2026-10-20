"""The Excel driver (data/bql_excel.py) against a fake Excel: one reused scratch book, no disturbance of the user's
windows, idempotent close, and only stale scratch books closed (see the module docstring)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ttf_forecaster.data import bql_excel


class _Api:
    """The COM Workbook object: Saved, Calculate and Close(SaveChanges=...)."""

    def __init__(self, book: "_Book", saved: bool) -> None:
        self._book = book
        self.Saved = saved
        self.calculated_while_screen_updating: list[bool] = []
        self.closed_with: bool | None = None

    def Calculate(self) -> None:
        self.calculated_while_screen_updating.append(self._book.app.screen_updating)

    def Close(self, SaveChanges: bool = True) -> None:  # noqa: N803 (COM keyword)
        self.closed_with = SaveChanges
        self._book.closed = True
        self._book.app.books.remove(self._book)


class _SheetApi:
    """The COM Worksheet object: Calculate recalculates that sheet only."""

    def __init__(self, sheet: "_Sheet") -> None:
        self._sheet = sheet

    def Calculate(self) -> None:  # noqa: N802 (COM name)
        self._sheet.book.api.calculated_while_screen_updating.append(self._sheet.book.app.screen_updating)


class _Sheet:
    def __init__(self, book: "_Book", name: str) -> None:
        self.book, self.name = book, name
        self.api = _SheetApi(self)
        self.formula: str | None = None
        self._block: list | None = None
        self._expand = False

    def range(self, _addr: str) -> "_Sheet":
        return self

    def expand(self) -> "_Sheet":
        self._expand = True
        return self

    @property
    def value(self) -> Any:
        if self.formula is not None and self._block is None:
            self._block = next((v for k, v in self.book.app.results.items() if k in self.formula), [["ID"]])
        if self._expand:
            return self._block
        if self.book.app.polls > 0:  # the add-in is still requesting data
            self.book.app.polls -= 1
            return "#N/A Requesting data..."
        return self._block[0][0] if self._block else None

    def delete(self) -> None:
        self.book.sheets.remove(self)
        self.book.app.deleted.append(self.name)


class _Sheets(list):
    def __init__(self, book: "_Book") -> None:
        super().__init__()
        self._book = book

    def add(self, name: str) -> _Sheet:
        sh = _Sheet(self._book, name)
        self.append(sh)
        return sh


class _Book:
    def __init__(self, app: "_App", name: str, sheet_names=("Sheet1",), saved: bool = False) -> None:
        self.app, self._name, self.closed = app, name, False
        self.api = _Api(self, saved)
        self.sheets = _Sheets(self)
        for n in sheet_names:
            self.sheets.add(n)

    @property
    def name(self) -> str:
        if self.closed:
            raise RuntimeError("workbook is closed")
        return self._name

    def activate(self) -> None:
        self.app.books.active = self


class _Books(list):
    def __init__(self, app: "_App") -> None:
        super().__init__()
        self.app = app
        self.active: _Book | None = None
        self._next = 1

    def add(self) -> _Book:
        book = _Book(self.app, f"Book{self._next}")
        self._next += 1
        self.append(book)
        self.active = book  # Excel activates a new workbook
        return book


class _App:
    def __init__(self, results: dict[str, list], polls: int = 0) -> None:
        self.results, self.polls = results, polls
        self.calculation, self.display_alerts, self.screen_updating = "automatic", True, True
        self.deleted: list[str] = []
        self.books = _Books(self)

    def user_book(self, name: str, sheet_names=("Sheet1",), saved: bool = True) -> _Book:
        book = _Book(self, name, sheet_names, saved)
        self.books.append(book)
        self.books.active = book
        return book


@pytest.fixture
def fake_excel(monkeypatch: pytest.MonkeyPatch):
    def make(results: dict[str, list], polls: int = 0) -> _App:
        app = _App(results, polls)
        app.user_book("2167499.xlsx")  # a user workbook that must never be touched
        monkeypatch.setattr(bql_excel, "xw", type("xw", (), {"apps": type("apps", (), {"active": app})}))
        monkeypatch.setattr(bql_excel, "_book", None)
        monkeypatch.setattr(bql_excel, "time", SimpleNamespace(sleep=lambda s: None, time=_clock()))
        return app

    return make


def _clock():
    t = [0.0]

    def now() -> float:
        t[0] += 1.0
        return t[0]

    return now


TEXT = "let(#w=weather()) get(#w().HDD_18C) for('DE Country')"
BLOCK = [["ID", "#w().HDD_18C"], ["DE Country", 1.0]]


def test_driver_refuses_more_than_20_queries() -> None:
    with pytest.raises(ValueError, match="at most 20"):
        bql_excel.run_queries({f"q{i}": "x" for i in range(21)})


def test_driver_turns_a_one_column_block_into_a_frame() -> None:
    assert isinstance(bql_excel._to_frame(["DATES", "2026-10-01", "2026-10-02"]), pd.DataFrame)
    df = bql_excel._to_frame([["ID", "#w().HDD_18C"], ["DE Country", 1.5]])
    assert list(df.columns) == ["ID", "#w().HDD_18C"] and df.iloc[0, 1] == 1.5


def test_one_scratch_book_is_reused_and_its_query_sheets_cleared(fake_excel) -> None:
    app = fake_excel({TEXT: BLOCK, "other": [["ID", "v"], ["NL", 2.0]]})
    first = bql_excel.run_queries({"a": TEXT}, timeout_s=10)
    second = bql_excel.run_queries({"b": "other"}, timeout_s=10)
    assert isinstance(first["a"], pd.DataFrame) and isinstance(second["b"], pd.DataFrame)
    scratch = [b for b in app.books if b.name.startswith("Book")]
    assert len(scratch) == 1  # one scratch workbook for the whole run
    assert "q0" in app.deleted  # the first batch's sheet was cleared before the second batch
    assert [s.name for s in scratch[0].sheets] == [bql_excel.HOLD, "q0"]


def test_close_scratch_closes_only_the_scratch_book_unsaved(fake_excel) -> None:
    app = fake_excel({TEXT: BLOCK})
    bql_excel.run_queries({"a": TEXT}, timeout_s=10)
    scratch = next(b for b in app.books if b.name.startswith("Book"))
    bql_excel.close_scratch()
    assert scratch.closed and scratch.api.closed_with is False
    bql_excel.close_scratch()  # a second call is a no-op
    assert [b.name for b in app.books] == ["2167499.xlsx"]


def test_user_workbook_is_not_calculated_closed_or_left_active(fake_excel) -> None:
    app = fake_excel({TEXT: BLOCK})
    user = app.books[0]
    bql_excel.run_queries({"a": TEXT}, timeout_s=10)
    assert user.api.calculated_while_screen_updating == []  # never calculated
    assert user.closed is False
    assert app.books.active is user  # the user's window keeps focus


def test_screen_updating_is_off_during_a_batch_and_restored_after(fake_excel) -> None:
    app = fake_excel({TEXT: BLOCK})
    bql_excel.run_queries({"a": TEXT}, timeout_s=10)
    scratch = next(b for b in app.books if b.name.startswith("Book"))
    assert scratch.api.calculated_while_screen_updating == [False]
    assert app.screen_updating is True and app.display_alerts is True and app.calculation == "automatic"


def test_stale_scratch_book_is_closed_and_other_books_are_not(fake_excel) -> None:
    app = fake_excel({TEXT: BLOCK})
    stale = _Book(app, "Book52", ("q0", "q1", "q2", "q3", "q4"), saved=False)  # left by a failed run
    saved_q = _Book(app, "Book9", ("q0",), saved=True)  # saved: not ours to close
    own_sheet = _Book(app, "Book10", ("Sheet1", "q0"), saved=False)  # the user has a sheet of their own
    app.books.extend([stale, saved_q, own_sheet])
    bql_excel.run_queries({"a": TEXT}, timeout_s=10)
    assert stale.closed and stale.api.closed_with is False
    assert not saved_q.closed and not own_sheet.closed
    assert not app.books[0].closed  # the user's workbook


def test_driver_retries_a_com_refusal_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class com_error(Exception):  # named like pywintypes.com_error
        pass

    calls: list[int] = []

    def flaky(queries, timeout_s, opts):
        calls.append(1)
        if len(calls) == 1:
            raise com_error("(-2147352567, 'OLE error 0x800ac472')")
        return {"a": "ok"}

    monkeypatch.setattr(bql_excel, "_run_once", flaky)
    monkeypatch.setattr(bql_excel, "time", SimpleNamespace(sleep=lambda s: None))
    assert bql_excel.run_queries({"a": "x"}) == {"a": "ok"}
    assert len(calls) == 2


def test_driver_does_not_retry_an_unrelated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def broken(queries, timeout_s, opts):
        calls.append(1)
        raise TypeError("'bool' object is not callable")

    monkeypatch.setattr(bql_excel, "_run_once", broken)
    monkeypatch.setattr(bql_excel, "time", SimpleNamespace(sleep=lambda s: None))
    with pytest.raises(TypeError):
        bql_excel.run_queries({"a": "x"})
    assert len(calls) == 1


def test_driver_reports_timeout(fake_excel) -> None:
    fake_excel({"x": [["ID", "v"]]}, polls=10**9)
    out = bql_excel.run_queries({"a": "x"}, timeout_s=50)
    assert out["a"] == "TIMEOUT"
