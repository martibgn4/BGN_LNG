"""
============================================================
bbg/session.py  --  Hardened Bloomberg Desktop API client
============================================================
A thin, dependency-light wrapper over ``blpapi`` built for bulk
equity research pulls. It exists because ``xbbg`` hides the
request/response plumbing we need to control here: chunking,
field exceptions, partial responses and dead-ticker handling.

Design notes
------------
* One long-lived session, reused across every request. Opening a
  session per request is the single biggest source of slowness.
* Every request is chunked. Bloomberg silently degrades on large
  security lists, so we cap securities-per-request conservatively.
* Field exceptions are *collected*, never raised. In a 500-name
  cross-section it is normal for a handful of names to lack a
  field; killing the whole pull for that would be useless.
* Dead/delisted tickers (e.g. ``1436513D UN Equity``) are first
  class citizens -- they are what makes the backtest survivorship
  bias free -- so a ``securityError`` is recorded, not raised.

Requires Bloomberg Desktop (bbcomm) running on localhost:8194.
============================================================
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

try:
    import blpapi
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "blpapi is required. Install with:\n"
        "  pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
    ) from exc

log = logging.getLogger(__name__)

_RESPONSE_TYPES = (blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE)

# Bloomberg degrades / truncates on oversized requests. These are
# deliberately conservative; raising them buys little and risks
# silent truncation.
MAX_SECURITIES_REF = 100
MAX_SECURITIES_HIST = 25
MAX_FIELDS = 25


@dataclass
class PullStats:
    """Tracks request volume so we can reason about Desktop API fair-use limits."""

    requests: int = 0
    securities: int = 0
    field_hits: int = 0
    errors: list[str] = field(default_factory=list)
    started: dt.datetime = field(default_factory=dt.datetime.now)

    def summary(self) -> str:
        elapsed = (dt.datetime.now() - self.started).total_seconds()
        return (
            f"{self.requests} requests | {self.securities} security-pulls | "
            f"{self.field_hits} field-hits | {len(self.errors)} errors | {elapsed:.1f}s"
        )


def _chunks(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class BloombergSession:
    """Reusable Bloomberg Desktop API session.

    Usage::

        with BloombergSession() as bbg:
            px = bbg.bdh(["AAPL US Equity"], ["PX_LAST"], "20200101", "20241231")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        timeout_ms: int = 30_000,
        throttle_s: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.throttle_s = throttle_s
        self.max_retries = max_retries
        self._session: blpapi.Session | None = None
        self._refdata = None
        self.stats = PullStats()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self) -> "BloombergSession":
        if self._session is not None:
            return self
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        # Suppress the chatty admin messages we do not consume.
        opts.setDefaultSubscriptionService("//blp/refdata")
        session = blpapi.Session(opts)
        if not session.start():
            raise ConnectionError(
                f"Could not start Bloomberg session on {self.host}:{self.port}. "
                "Is the Bloomberg Terminal running and logged in?"
            )
        if not session.openService("//blp/refdata"):
            raise ConnectionError("Could not open //blp/refdata service.")
        self._session = session
        self._refdata = session.getService("//blp/refdata")
        log.info("Bloomberg session opened on %s:%s", self.host, self.port)
        return self

    def close(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
            self._refdata = None

    def __enter__(self) -> "BloombergSession":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def service(self):
        if self._refdata is None:
            self.open()
        return self._refdata

    # ------------------------------------------------------------------
    # low level request driver
    # ------------------------------------------------------------------
    def _send(self, request) -> list:
        """Send a request and collect every message across partial responses."""
        assert self._session is not None
        messages: list = []
        for attempt in range(1, self.max_retries + 1):
            try:
                self._session.sendRequest(request)
                messages.clear()
                while True:
                    ev = self._session.nextEvent(self.timeout_ms)
                    if ev.eventType() in _RESPONSE_TYPES:
                        for msg in ev:
                            messages.append(msg)
                    if ev.eventType() == blpapi.Event.RESPONSE:
                        break
                    if ev.eventType() == blpapi.Event.TIMEOUT:
                        raise TimeoutError("Bloomberg request timed out")
                self.stats.requests += 1
                if self.throttle_s:
                    time.sleep(self.throttle_s)
                return messages
            except Exception as exc:  # noqa: BLE001 - retry any transport failure
                if attempt == self.max_retries:
                    self.stats.errors.append(f"request failed after {attempt} attempts: {exc}")
                    log.error("Bloomberg request failed permanently: %s", exc)
                    return []
                wait = 2.0 * attempt
                log.warning("Bloomberg request failed (attempt %s), retrying in %.0fs: %s", attempt, wait, exc)
                time.sleep(wait)
        return messages

    @staticmethod
    def _apply_overrides(request, overrides: dict[str, str] | None) -> None:
        if not overrides:
            return
        element = request.getElement("overrides")
        for key, value in overrides.items():
            ov = element.appendElement()
            ov.setElement("fieldId", key)
            ov.setElement("value", str(value))

    @staticmethod
    def _element_value(el):
        """Convert a blpapi element to a native Python value."""
        try:
            dtype = el.datatype()
        except Exception:  # noqa: BLE001
            return el.getValueAsString()
        if dtype == blpapi.DataType.FLOAT64 or dtype == blpapi.DataType.FLOAT32:
            return el.getValueAsFloat()
        if dtype in (blpapi.DataType.INT32, blpapi.DataType.INT64):
            return el.getValueAsInteger()
        if dtype == blpapi.DataType.DATE:
            d = el.getValueAsDatetime()
            return dt.date(d.year, d.month, d.day)
        if dtype == blpapi.DataType.DATETIME:
            return el.getValueAsDatetime()
        if dtype == blpapi.DataType.BOOL:
            return el.getValueAsBool()
        return el.getValueAsString()

    # ------------------------------------------------------------------
    # reference data (BDP)
    # ------------------------------------------------------------------
    def bdp(
        self,
        securities: Sequence[str],
        fields: Sequence[str],
        overrides: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Current reference data. Returns a DataFrame indexed by security.

        Missing values come back as NaN rather than raising -- field
        exceptions are logged into ``self.stats.errors``.
        """
        securities = list(securities)
        fields = list(fields)
        if not securities or not fields:
            return pd.DataFrame()

        rows: dict[str, dict] = {}
        for sec_chunk in _chunks(securities, MAX_SECURITIES_REF):
            for fld_chunk in _chunks(fields, MAX_FIELDS):
                request = self.service.createRequest("ReferenceDataRequest")
                for s in sec_chunk:
                    request.append("securities", s)
                for f in fld_chunk:
                    request.append("fields", f)
                self._apply_overrides(request, overrides)

                for msg in self._send(request):
                    if msg.hasElement("responseError"):
                        self.stats.errors.append(
                            f"BDP responseError: {msg.getElement('responseError').getElementAsString('message')}"
                        )
                        continue
                    sec_data = msg.getElement("securityData")
                    for i in range(sec_data.numValues()):
                        d = sec_data.getValueAsElement(i)
                        sec = d.getElementAsString("security")
                        row = rows.setdefault(sec, {})
                        if d.hasElement("securityError"):
                            self.stats.errors.append(
                                f"BDP securityError {sec}: "
                                f"{d.getElement('securityError').getElementAsString('message')}"
                            )
                            continue
                        fe = d.getElement("fieldExceptions")
                        for j in range(fe.numValues()):
                            fx = fe.getValueAsElement(j)
                            self.stats.errors.append(
                                f"BDP fieldException {sec}.{fx.getElement('fieldId').getValueAsString()}: "
                                f"{fx.getElement('errorInfo').getElementAsString('message')}"
                            )
                        fd = d.getElement("fieldData")
                        for f in fld_chunk:
                            if fd.hasElement(f):
                                el = fd.getElement(f)
                                if not el.isArray():
                                    row[f] = self._element_value(el)
                                    self.stats.field_hits += 1
                self.stats.securities += len(sec_chunk)

        df = pd.DataFrame.from_dict(rows, orient="index")
        return df.reindex(index=securities, columns=fields)

    # ------------------------------------------------------------------
    # bulk reference data (e.g. index members)
    # ------------------------------------------------------------------
    def bds(
        self,
        security: str,
        field: str,
        overrides: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Bulk reference data -- returns the array field as a DataFrame."""
        request = self.service.createRequest("ReferenceDataRequest")
        request.append("securities", security)
        request.append("fields", field)
        self._apply_overrides(request, overrides)

        records: list[dict] = []
        for msg in self._send(request):
            if msg.hasElement("responseError"):
                self.stats.errors.append(
                    f"BDS responseError: {msg.getElement('responseError').getElementAsString('message')}"
                )
                continue
            sec_data = msg.getElement("securityData")
            for i in range(sec_data.numValues()):
                d = sec_data.getValueAsElement(i)
                if d.hasElement("securityError"):
                    self.stats.errors.append(
                        f"BDS securityError {security}: "
                        f"{d.getElement('securityError').getElementAsString('message')}"
                    )
                    continue
                fd = d.getElement("fieldData")
                if not fd.hasElement(field):
                    continue
                arr = fd.getElement(field)
                for k in range(arr.numValues()):
                    item = arr.getValueAsElement(k)
                    rec = {}
                    for m in range(item.numElements()):
                        sub = item.getElement(m)
                        rec[str(sub.name())] = self._element_value(sub)
                    records.append(rec)
        self.stats.securities += 1
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # historical data (BDH)
    # ------------------------------------------------------------------
    def bdh(
        self,
        securities: Sequence[str],
        fields: Sequence[str],
        start: str | dt.date,
        end: str | dt.date,
        periodicity: str = "DAILY",
        adjust_split: bool = True,
        adjust_abnormal: bool = True,
        currency: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Historical time series.

        Returns a long DataFrame with columns ``[date, security, <fields>]``.
        Long format keeps memory sane for 500 x 15y x many-fields pulls and
        maps cleanly onto a parquet store partitioned by security.

        Adjustments default to *split and abnormal adjusted* so that price
        series are continuous through corporate actions -- essential for a
        momentum model, which would otherwise read a 4-for-1 split as a -75%
        return.
        """
        securities = list(securities)
        fields = list(fields)
        if not securities or not fields:
            return pd.DataFrame(columns=["date", "security", *fields])

        start_s = _fmt_date(start)
        end_s = _fmt_date(end)
        frames: list[pd.DataFrame] = []

        for sec_chunk in _chunks(securities, MAX_SECURITIES_HIST):
            for fld_chunk in _chunks(fields, MAX_FIELDS):
                request = self.service.createRequest("HistoricalDataRequest")
                for s in sec_chunk:
                    request.append("securities", s)
                for f in fld_chunk:
                    request.append("fields", f)
                request.set("startDate", start_s)
                request.set("endDate", end_s)
                request.set("periodicitySelection", periodicity)
                request.set("adjustmentNormal", True)
                request.set("adjustmentAbnormal", adjust_abnormal)
                request.set("adjustmentSplit", adjust_split)
                request.set("adjustmentFollowDPDF", False)
                if currency:
                    request.set("currency", currency)
                self._apply_overrides(request, overrides)

                for msg in self._send(request):
                    if msg.hasElement("responseError"):
                        self.stats.errors.append(
                            f"BDH responseError: {msg.getElement('responseError').getElementAsString('message')}"
                        )
                        continue
                    sec_data = msg.getElement("securityData")
                    sec = sec_data.getElementAsString("security")
                    if sec_data.hasElement("securityError"):
                        self.stats.errors.append(
                            f"BDH securityError {sec}: "
                            f"{sec_data.getElement('securityError').getElementAsString('message')}"
                        )
                        continue
                    fd = sec_data.getElement("fieldData")
                    recs = []
                    for i in range(fd.numValues()):
                        point = fd.getValueAsElement(i)
                        rec = {"security": sec}
                        for m in range(point.numElements()):
                            sub = point.getElement(m)
                            rec[str(sub.name())] = self._element_value(sub)
                        recs.append(rec)
                    if recs:
                        frames.append(pd.DataFrame(recs))
                        self.stats.field_hits += len(recs) * len(fld_chunk)
                self.stats.securities += len(sec_chunk)

        if not frames:
            return pd.DataFrame(columns=["date", "security", *fields])

        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        # Chunking over fields can split one security across frames; collapse.
        out = out.groupby(["date", "security"], as_index=False).first()
        cols = ["date", "security"] + [f for f in fields if f in out.columns]
        return out[cols].sort_values(["security", "date"]).reset_index(drop=True)


def _fmt_date(d: str | dt.date) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")
