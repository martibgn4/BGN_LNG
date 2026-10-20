"""The daily snapshot: today's BSRCH outage snapshot and today's BQL weather forecasts (D-043).

Idempotent: each part skips what the store already holds for today's request date, so a second run on the same day
adds nothing. The BQL part needs a running Excel with the Bloomberg add-in; without one it is skipped with a message.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from ttf_forecaster.data import outages as out
from ttf_forecaster.data.storage import Store
from ttf_forecaster.pipelines import bql_weather
from ttf_forecaster.pipelines.outages import pull_snapshot


TERMINAL_NOT_CONNECTED = (
    "Bloomberg Terminal not connected: log in to the Terminal and re-run. "
    "Nothing was pulled today (outages, BQL weather and LNG fleet all need it)."
)


class TerminalNotConnected(RuntimeError):
    """The desktop API (localhost:8194) did not accept a session or open //blp/refdata."""


def check_terminal(session_factory: Callable[[], Any] | None = None) -> None:
    """Open a desktop-API session and the refdata service, then close it. Requests no data.

    Raises ``TerminalNotConnected`` with ``TERMINAL_NOT_CONNECTED`` when the Terminal is not running or not
    logged in. ``session_factory`` (default: a blpapi session to localhost:8194) is injectable for tests.
    """
    if session_factory is None:
        def session_factory() -> Any:
            import blpapi  # noqa: PLC0415 - only on a live run

            opts = blpapi.SessionOptions()
            opts.setServerHost("localhost")
            opts.setServerPort(8194)
            opts.setNumStartAttempts(1)
            opts.setConnectTimeout(5000)
            return blpapi.Session(opts)

    try:
        session = session_factory()
        started = session.start()
    except Exception as exc:  # blpapi missing or refusing the connection
        raise TerminalNotConnected(f"{TERMINAL_NOT_CONNECTED} ({type(exc).__name__}: {exc})") from exc
    try:
        if not started or not session.openService("//blp/refdata"):
            raise TerminalNotConnected(TERMINAL_NOT_CONNECTED)
    finally:
        if started:
            session.stop()


def excel_available() -> bool:
    """True when a running Excel instance is reachable through xlwings."""
    try:
        import xlwings as xw  # noqa: PLC0415 - only needed for this check

        return xw.apps.active is not None
    except Exception:  # no xlwings, no Excel, or Excel refuses the COM call
        return False


def run_daily(today: date, store: Store, outage_adapter: out.LiveOutageAdapter,
              *, request_time: pd.Timestamp | None = None,
              bql_runner: Callable[..., Any] | None = None) -> dict[str, str]:
    """Outages for ``today`` then BQL weather for ``today``. ``bql_runner=None`` skips the BQL part."""
    report: dict[str, str] = {}
    report["outages"] = str(pull_snapshot(today, store, outage_adapter, request_time=request_time))
    if bql_runner is None:
        report["bql_weather"] = "skipped: Excel with the Bloomberg add-in is not running"
        return report
    summary = bql_weather.pull_forecasts(today, today, store, bql_runner)
    report["bql_weather"] = f"rows={summary.rows} failures={len(summary.failures)}"
    return report
