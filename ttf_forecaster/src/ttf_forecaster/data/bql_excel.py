"""Run BQL queries through the user's running Excel (Bloomberg add-in) via xlwings.

User-authorised route (2026-10-19): the Python API login is not entitled to BQL (//blp/bqlsvc returns
"User not authorized to use BQL"), while the user's Excel BQL add-in works.

All queries of one process go on ONE scratch workbook (``_book``), opened once and reused. Each batch clears its
query sheets, writes the queries as ``=BQL.QUERY(...)`` on ``q<i>`` sheets, waits for the add-in, reads the block
and leaves the scratch book in place. The scratch book is closed unsaved by ``close_scratch`` (also registered
with atexit, so an error still closes it). The user's other workbooks are never closed, activated or recalculated:
only the scratch book is calculated, screen updating is off during a batch, and the window that was active before
a batch is re-activated afterwards.

A scratch book left open by an earlier failed run (named Book<n>, unsaved, containing only ``hold`` and ``q<i>``
sheets) is closed before a batch. Nothing else is closed.

Result shapes (measured 2026-10-19, see docs/bloomberg_discovery.md section 7):
  * one field, one location: a single column, returned as a one-column DataFrame;
  * several fields, several locations (with ``showids=t``): one row per (ID, lead), with an ``ID``
    column and one column per field; each location block is in lead order;
  * a date field and a value field in the same ``get`` do NOT align (values come back empty), so the
    valid date is never requested together with a value (see weather_bql.lead_dates).
"""
from __future__ import annotations

import atexit
import re
import time
from collections.abc import Mapping

import pandas as pd
import xlwings as xw

PENDING = ("#N/A Requesting", "Requesting Data", "#GETTING", "Loading")
HOLD = "hold"  # the scratch book's permanent sheet (a workbook always has one sheet); queries go on q<i> sheets
RETRIES = 3  # Excel can refuse a COM call while it is busy (2026-10-19: "OLE error 0x800ac472" on sheets.add)
_SCRATCH_BOOK = re.compile(r"^Book\d+$")
_SCRATCH_SHEET = re.compile(r"^q\d+$")
_book = None  # the one scratch workbook of this process


def _is_pending(v) -> bool:
    return isinstance(v, str) and any(p.lower() in v.lower() for p in PENDING)


def _to_frame(block) -> pd.DataFrame | str:
    """xlwings returns a list of rows for 2-D blocks and a flat list for a one-column block."""
    if block is None or block == []:
        return "EMPTY"
    if isinstance(block, list) and block and not isinstance(block[0], list):
        # one column: first element is the header
        header, rows = block[0], block[1:]
        return pd.DataFrame({str(header): rows})
    if isinstance(block, list) and block and isinstance(block[0], list):
        return pd.DataFrame(block[1:], columns=[str(c) for c in block[0]])
    return f"SCALAR: {block!r}"


def run_queries(queries: Mapping[str, str], timeout_s: int = 240,
                opts: str = "showallcols=t, showids=t") -> dict:
    """``{name: bql}`` -> ``{name: DataFrame | error string}``. At most 20 queries per call. A COM refusal is
    retried up to RETRIES times. The scratch book stays open for the next batch (see ``close_scratch``)."""
    if len(queries) > 20:
        raise ValueError("send at most 20 queries per Excel book (user instruction 2026-10-19)")
    for attempt in range(1, RETRIES + 1):
        try:
            return _run_once(queries, timeout_s, opts)
        except Exception as exc:  # COM refusals come through as pywintypes.com_error ("OLE error ...")
            retryable = "com_error" in type(exc).__name__ or "OLE error" in str(exc)
            if attempt == RETRIES or not retryable:
                raise
            time.sleep(10 * attempt)
    raise RuntimeError("unreachable")


def close_scratch() -> None:
    """Close this process's scratch workbook without saving. Safe to call twice; runs at exit too."""
    global _book
    book, _book = _book, None
    if book is not None:
        _close_unsaved(book)


atexit.register(close_scratch)


def _is_open(book) -> bool:
    try:
        book.name
    except Exception:  # the COM object is gone: the user or Excel closed it
        return False
    return True


def _close_unsaved(book) -> None:
    if _is_open(book):
        book.api.Close(SaveChanges=False)


def _is_stale_scratch(book) -> bool:
    """A Book<n> this tool left behind: unsaved, and holding only the hold sheet and q<i> sheets."""
    try:
        if not _SCRATCH_BOOK.match(book.name) or bool(book.api.Saved):
            return False
        return all(s.name == HOLD or _SCRATCH_SHEET.match(s.name) for s in book.sheets)
    except Exception:
        return False


def _close_stale(app) -> None:
    for book in list(app.books):
        if _book is not None and book.name == _book.name:
            continue
        if _is_stale_scratch(book):
            _close_unsaved(book)


def _scratch_book(app):
    """The process's one scratch workbook: opened on first use, reused after. Opening it activates it, so the
    caller re-activates the previous window."""
    global _book
    _close_stale(app)
    if _book is not None and not _is_open(_book):
        _book = None
    if _book is None:
        _book = app.books.add()
        _book.sheets[0].name = HOLD
    return _book


def _run_once(queries: Mapping[str, str], timeout_s: int, opts: str) -> dict:
    app = xw.apps.active
    if app is None:
        raise RuntimeError("No running Excel instance found")
    prev_calc, prev_alerts, prev_screen = app.calculation, app.display_alerts, app.screen_updating
    prev_active = app.books.active
    app.display_alerts = False  # the scratch book is closed without a save prompt
    app.screen_updating = False  # no repaint while the user's window is in front
    out: dict = {}
    try:
        book = _scratch_book(app)
        for s in list(book.sheets):  # clear the previous batch's query sheets
            if s.name != HOLD:
                s.delete()
        sheets = {}
        for i, (name, q) in enumerate(queries.items()):
            sh = book.sheets.add(name=f"q{i}")
            esc = q.replace('"', '""')
            sh.range("A1").formula = f'=BQL.QUERY("{esc}","{opts}")'
            sheets[name] = sh
        for sh in sheets.values():  # only the scratch book's query sheets; the user's books are not recalculated
            sh.api.Calculate()
        t0 = time.time()
        pending = set(sheets)
        while pending and time.time() - t0 < timeout_s:
            time.sleep(3)
            for name in list(pending):
                if not _is_pending(sheets[name].range("A1").value):
                    pending.discard(name)
        for name, sh in sheets.items():
            if name in pending:
                out[name] = "TIMEOUT"
                continue
            block = sh.range("A1").expand().value
            if isinstance(block, str) or (isinstance(block, list) and block and isinstance(block[0], str)
                                          and not isinstance(block[0], list) and len(block) == 1):
                out[name] = f"SCALAR: {block!r}"
                continue
            out[name] = _to_frame(block)
    finally:
        app.calculation, app.display_alerts, app.screen_updating = prev_calc, prev_alerts, prev_screen
        if prev_active is not None and _is_open(prev_active):
            prev_active.activate()  # the user's window keeps focus
    return out
