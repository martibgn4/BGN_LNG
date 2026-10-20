"""Gas futures contracts: delivery windows, DST-aware delivery hours, expiry and vendor tickers.

Conventions (``docs/target_definitions.md`` §0-§1):

* Internal contract ids come from :func:`ttf_forecaster.data.base.contract_id`:
  ``TTF.M.2026-11``, ``TTF.Q.2027-Q1``, ``TTF.S.2027-SUM`` (Apr-Sep 2027),
  ``TTF.S.2026-WIN`` (Oct 2026 - Mar 2027), ``TTF.Y.2027`` (Jan-Dec 2027).
* ``delivery_start`` / ``delivery_end`` are naive **local dates**; ``delivery_end`` is exclusive.
  Physical delivery runs gas day to gas day (06:00-06:00 local for TTF), which is what
  :meth:`FuturesContract.delivery_hours` counts. DST is handled by the zone database, so March is
  743 h and October 745 h in Europe/Amsterdam.
* Expiry: last trading day = ``N`` sessions before the first calendar day of delivery, ``N`` from
  ``configs/instruments.yaml`` (default 2, **[CONFIRM]**). A computed expiry that disagrees with
  vendor metadata raises :class:`ExpiryMismatchError` - never a warning.
* Vendor tickers are built from roots in ``configs/sources/bloomberg.yaml`` by the rule documented
  there (provenance: ``Data/lng_report/bloomberg.py``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.calendars import DateLike, TradingCalendar, to_local_day
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    MARKET_TZ,
    DataError,
    IdentifierStatus,
    IdentifierUnknown,
    contract_id,
    validate_frame,
)
from ttf_forecaster.settings import ConfigError, get, load_config

FUTURES_TENORS: tuple[str, ...] = ("M", "Q", "S", "Y")
SEASONS: tuple[str, ...] = ("SUM", "WIN")


class ExpiryMismatchError(DataError):
    """A rule-computed last trading day disagrees with vendor contract metadata."""


class ContractParseError(ValueError):
    """A contract id or vendor ticker cannot be parsed."""


# ---------------------------------------------------------------------------------------------
# Contract objects
# ---------------------------------------------------------------------------------------------


def _month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1)


@dataclass(frozen=True, order=True)
class FuturesContract:
    """One listed futures contract. Orders by delivery start, then tenor.

    Attributes:
        delivery_start: first delivery day (naive local date, first of a month).
        tenor_type: ``M``, ``Q``, ``S`` or ``Y``.
        hub: hub code, e.g. ``TTF``.
    """

    delivery_start: pd.Timestamp
    tenor_type: str
    hub: str = field(default="TTF")

    def __post_init__(self) -> None:
        if self.tenor_type not in FUTURES_TENORS:
            raise ContractParseError(f"tenor_type {self.tenor_type!r} not in {FUTURES_TENORS}")
        ds = pd.Timestamp(self.delivery_start)
        if ds.tzinfo is not None or ds != ds.normalize() or ds.day != 1:
            raise ContractParseError(f"delivery_start must be a naive first-of-month, got {ds}")
        m = ds.month
        if self.tenor_type == "Q" and m not in (1, 4, 7, 10):
            raise ContractParseError(f"quarter must start in Jan/Apr/Jul/Oct, got month {m}")
        if self.tenor_type == "S" and m not in (4, 10):
            raise ContractParseError(f"season must start in Apr or Oct, got month {m}")
        if self.tenor_type == "Y" and m != 1:
            raise ContractParseError(f"calendar year must start in Jan, got month {m}")
        object.__setattr__(self, "delivery_start", ds)

    # constructors ------------------------------------------------------------------------
    @classmethod
    def month(cls, hub: str, year: int, month: int) -> FuturesContract:
        return cls(_month_start(year, month), "M", hub)

    @classmethod
    def quarter(cls, hub: str, year: int, q: int) -> FuturesContract:
        if q not in (1, 2, 3, 4):
            raise ContractParseError(f"quarter {q} not in 1..4")
        return cls(_month_start(year, 3 * (q - 1) + 1), "Q", hub)

    @classmethod
    def season(cls, hub: str, year: int, season: str) -> FuturesContract:
        """``season`` SUM (Apr-Sep ``year``) or WIN (Oct ``year`` - Mar ``year+1``)."""
        if season not in SEASONS:
            raise ContractParseError(f"season {season!r} not in {SEASONS}")
        return cls(_month_start(year, 4 if season == "SUM" else 10), "S", hub)

    @classmethod
    def calendar_year(cls, hub: str, year: int) -> FuturesContract:
        return cls(_month_start(year, 1), "Y", hub)

    # properties --------------------------------------------------------------------------
    @property
    def n_months(self) -> int:
        return {"M": 1, "Q": 3, "S": 6, "Y": 12}[self.tenor_type]

    @property
    def delivery_end(self) -> pd.Timestamp:
        """Exclusive end of delivery (naive local date)."""
        return self.delivery_start + pd.DateOffset(months=self.n_months)

    @property
    def label(self) -> str:
        ds = self.delivery_start
        if self.tenor_type == "M":
            return ds.strftime("%Y-%m")
        if self.tenor_type == "Q":
            return f"{ds.year}-Q{(ds.month - 1) // 3 + 1}"
        if self.tenor_type == "S":
            return f"{ds.year}-{'SUM' if ds.month == 4 else 'WIN'}"
        return f"{ds.year}"

    @property
    def contract_id(self) -> str:
        return contract_id(self.hub, self.tenor_type, self.label)

    @property
    def season_code(self) -> str | None:
        if self.tenor_type != "S":
            return None
        return "SUM" if self.delivery_start.month == 4 else "WIN"

    def constituent_months(self) -> tuple[FuturesContract, ...]:
        """The monthly contracts covering this contract's delivery period."""
        return tuple(
            FuturesContract(self.delivery_start + pd.DateOffset(months=i), "M", self.hub)
            for i in range(self.n_months)
        )

    def next(self) -> FuturesContract:
        """The following contract of the same tenor (next month / quarter / season / year)."""
        step = {"M": 1, "Q": 3, "S": 6, "Y": 12}[self.tenor_type]
        return FuturesContract(self.delivery_start + pd.DateOffset(months=step), self.tenor_type,
                               self.hub)

    def delivery_window(
        self, tz: str = MARKET_TZ, gas_day_start_hour: int = 6
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Physical delivery window as tz-aware **UTC** timestamps [start, end)."""
        hour = pd.Timedelta(hours=gas_day_start_hour)
        start = (self.delivery_start + hour).tz_localize(tz).tz_convert("UTC")
        end = (self.delivery_end + hour).tz_localize(tz).tz_convert("UTC")
        return start, end

    def delivery_hours(self, tz: str = MARKET_TZ, gas_day_start_hour: int = 6) -> int:
        """Hours of delivery, DST-aware (Amsterdam: March 743, October 745, other 31-day 744)."""
        start, end = self.delivery_window(tz, gas_day_start_hour)
        hours = (end - start).total_seconds() / 3600.0
        if not float(hours).is_integer():
            raise DataError(f"{self.contract_id}: non-integer delivery hours {hours}")
        return int(hours)

    def __str__(self) -> str:
        return self.contract_id


_ID_RE = re.compile(
    r"^(?P<hub>[A-Za-z0-9_]+)\.(?P<tenor>[MQSY])\.(?P<label>\d{4}(?:-(?:\d{2}|Q[1-4]|SUM|WIN))?)$"
)


def parse_contract_id(cid: str) -> FuturesContract:
    """Inverse of :attr:`FuturesContract.contract_id`; raises :class:`ContractParseError`."""
    m = _ID_RE.match(cid)
    if not m:
        raise ContractParseError(f"Cannot parse contract id {cid!r}")
    hub, tenor, label = m["hub"], m["tenor"], m["label"]
    year = int(label[:4])
    rest = label[5:]
    try:
        if tenor == "M":
            return FuturesContract.month(hub, year, int(rest))
        if tenor == "Q":
            return FuturesContract.quarter(hub, year, int(rest[1]))
        if tenor == "S":
            return FuturesContract.season(hub, year, rest)
        if rest:
            raise ContractParseError(f"Calendar contract id {cid!r} must have no suffix")
        return FuturesContract.calendar_year(hub, year)
    except (ValueError, IndexError) as exc:
        raise ContractParseError(f"Cannot parse contract id {cid!r}: {exc}") from exc


def monthly_contracts(hub: str, first: DateLike, last: DateLike) -> list[FuturesContract]:
    """Monthly contracts whose delivery month lies in [month of ``first``, month of ``last``]."""
    start = to_local_day(first).replace(day=1)
    end = to_local_day(last).replace(day=1)
    return [FuturesContract(d, "M", hub) for d in pd.date_range(start, end, freq="MS")]


def contracts_between(hub: str, tenor_type: str, first: DateLike, last: DateLike
                      ) -> list[FuturesContract]:
    """All ``tenor_type`` contracts whose delivery starts in [``first``, ``last``]."""
    starts = {"M": range(1, 13), "Q": (1, 4, 7, 10), "S": (4, 10), "Y": (1,)}[tenor_type]
    a, b = to_local_day(first), to_local_day(last)
    out = [
        FuturesContract(d, tenor_type, hub)
        for d in pd.date_range(a.replace(day=1), b, freq="MS")
        if d.month in starts and a <= d <= b
    ]
    return out


# ---------------------------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpiryRule:
    """Last trading day = ``business_days_before_delivery[tenor]`` sessions before delivery start.

    ``status`` is carried from config so reports can flag unconfirmed rules.
    """

    business_days_before_delivery: Mapping[str, int | None]
    status: str = "requires_confirmation"

    @classmethod
    def default(cls) -> ExpiryRule:
        return cls({t: 2 for t in FUTURES_TENORS})

    @classmethod
    def from_config(cls, hub: str = "TTF", config_dir: str | None = None) -> ExpiryRule:
        cfg = get(load_config("instruments", config_dir), "hubs", hub, "expiry")
        bdays = cfg.get("business_days_before_delivery")
        if not isinstance(bdays, Mapping):
            raise ConfigError(f"instruments.yaml hubs.{hub}.expiry.business_days_before_delivery "
                              "must map tenor -> int")
        return cls(dict(bdays), str(cfg.get("status", "requires_confirmation")))

    def days_for(self, tenor_type: str) -> int:
        n = self.business_days_before_delivery.get(tenor_type)
        if n is None:
            raise ConfigError(f"No expiry rule configured for tenor {tenor_type!r} "
                              f"(status {self.status}); the rule must be supplied, not assumed")
        return int(n)


def last_trade_date(contract: FuturesContract, calendar: TradingCalendar, rule: ExpiryRule
                    ) -> pd.Timestamp:
    """Rule-computed last trading day (naive local date)."""
    return calendar.add_sessions(contract.delivery_start, -rule.days_for(contract.tenor_type))


def cross_check_expiry(cid: str, computed: DateLike, vendor: DateLike | None) -> None:
    """Raise :class:`ExpiryMismatchError` when vendor metadata disagrees with the rule.

    ``vendor=None`` / NaT means no vendor metadata: nothing to check.
    """
    if vendor is None or pd.isna(vendor):
        return
    c, v = to_local_day(computed), to_local_day(vendor)
    if c != v:
        raise ExpiryMismatchError(
            f"{cid}: rule-computed last trade {c.date()} != vendor {v.date()}. Fix the expiry rule "
            "or the calendar in configs/instruments.yaml; targets are not built on a guessed expiry."
        )


def _local_midnight_utc(days: Sequence[pd.Timestamp], tz: str) -> pd.Series:
    idx = pd.DatetimeIndex(list(days)).tz_localize(tz).tz_convert("UTC")
    return pd.Series(idx)


def contract_metadata(
    contracts: Iterable[FuturesContract],
    calendar: TradingCalendar,
    rule: ExpiryRule,
    *,
    vendor_last_trade: Mapping[str, DateLike] | None = None,
    currency: str = "EUR",
    unit: str = "EUR/MWh",
    tz: str = MARKET_TZ,
    vendor_authoritative: bool = False,
) -> pd.DataFrame:
    """CONTRACT_METADATA_SCHEMA rows; every expiry is cross-checked against ``vendor_last_trade``.

    ``last_trade_date_source`` is ``"vendor"`` when vendor metadata exists (and agreed with the
    rule), else ``"rule"``. Dates are local midnights stored in UTC, like ``trade_date``.
    """
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    vendor_last_trade = vendor_last_trade or {}
    for c in contracts:
        ltd = last_trade_date(c, calendar, rule)
        vendor = vendor_last_trade.get(c.contract_id)
        has_vendor = vendor is not None and not pd.isna(vendor)
        if vendor_authoritative and has_vendor:
            # D-035: the exchange date wins; disagreements are reported in attrs, never silent.
            if to_local_day(ltd) != to_local_day(vendor):
                # ISO strings: attrs are serialised into Parquet metadata as JSON
                mismatches.append({"contract_id": c.contract_id,
                                   "rule": str(to_local_day(ltd).date()),
                                   "vendor": str(to_local_day(vendor).date())})
            ltd = to_local_day(vendor)
        else:
            cross_check_expiry(c.contract_id, ltd, vendor)
        rows.append(
            {
                "contract_id": c.contract_id,
                "hub": c.hub,
                "tenor_type": c.tenor_type,
                "delivery_start": c.delivery_start,
                "delivery_end": c.delivery_end,
                "last_trade_date": ltd,
                "last_trade_date_source": "vendor" if has_vendor else "rule",
                "currency": currency,
                "unit": unit,
            }
        )
    df = pd.DataFrame(rows, columns=CONTRACT_METADATA_SCHEMA.names)
    for col in ("delivery_start", "delivery_end", "last_trade_date"):
        df[col] = (
            _local_midnight_utc(df[col].tolist(), tz) if len(df)
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
    out = validate_frame(df, CONTRACT_METADATA_SCHEMA)
    out.attrs["expiry_mismatches"] = mismatches
    return out


def last_trade_map(metadata: pd.DataFrame, tz: str = MARKET_TZ) -> pd.Series:
    """``contract_id -> last trade (naive local date)`` from a CONTRACT_METADATA_SCHEMA frame."""
    ltd = metadata["last_trade_date"]
    if isinstance(ltd.dtype, pd.DatetimeTZDtype):
        ltd = ltd.dt.tz_convert(tz).dt.tz_localize(None)
    return pd.Series(ltd.dt.normalize().to_numpy(), index=metadata["contract_id"].to_numpy(),
                     name="last_trade_date")


# ---------------------------------------------------------------------------------------------
# Bloomberg tickers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BloombergCurveIds:
    """Bloomberg roots and ticker rule for one hub (from ``configs/sources/bloomberg.yaml``)."""

    hub: str
    roots: Mapping[str, str]
    period_suffix: bool
    yellow_key: str
    status: str
    month_codes: tuple[str, ...]
    season_codes: Mapping[str, str]
    year_code: str
    year_digits: int
    suffix_letters: Mapping[str, str]
    provenance: str = ""
    # "tzt": one ticker per contract, root + month + 2-digit year (+ period letter if period_suffix).
    # "pnx": monthly only, the PNX/EEX form. The listed contract is root+month+1-digit year+"M";
    #        an expired contract is root+month+1-digit year+"M"+2-digit year (verified 2026-10-19).
    form: str = "tzt"

    @classmethod
    def from_config(cls, hub: str, config_dir: str | None = None) -> BloombergCurveIds:
        cfg = load_config("sources/bloomberg", config_dir)
        rule = get(cfg, "ticker_rule")
        curve = get(cfg, "curves", hub)
        return cls.from_mappings(hub, curve, rule)

    @classmethod
    def from_mappings(cls, hub: str, curve: Mapping[str, Any], rule: Mapping[str, Any]
                      ) -> BloombergCurveIds:
        codes = tuple(rule["month_codes"])
        if len(codes) != 12:
            raise ConfigError("ticker_rule.month_codes must list 12 codes")
        return cls(
            hub=hub,
            roots=dict(curve.get("roots") or {}),
            period_suffix=bool(curve.get("period_suffix", False)),
            yellow_key=str(curve.get("yellow_key", "Comdty")),
            status=str(curve.get("status", IdentifierStatus.REQUIRES_USER_INPUT.value)),
            month_codes=codes,
            season_codes=dict(rule["season_codes"]),
            year_code=str(rule["year_code"]),
            year_digits=int(rule.get("year_digits", 2)),
            suffix_letters=dict(rule["suffix_letters"]),
            provenance=str(curve.get("provenance", "")),
            form=str(curve.get("ticker_form", "tzt")),
        )


def bloomberg_ticker(contract: FuturesContract, ids: BloombergCurveIds) -> str:
    """Bloomberg ticker for ``contract`` under the repo rule (``configs/sources/bloomberg.yaml``).

    Raises:
        IdentifierUnknown: no root configured for the tenor, or root status is not ``known``.
    """
    if ids.form != "tzt":
        raise IdentifierUnknown(f"{ids.hub} uses ticker form {ids.form!r}; use "
                                "bloomberg_ticker_candidates")
    if ids.status != IdentifierStatus.KNOWN.value:
        raise IdentifierUnknown(f"Bloomberg roots for {ids.hub} have status {ids.status!r}")
    root = ids.roots.get(contract.tenor_type)
    if not root:
        raise IdentifierUnknown(f"No Bloomberg root for {ids.hub} tenor {contract.tenor_type}")
    ds = contract.delivery_start
    if contract.tenor_type in ("M", "Q"):
        code = ids.month_codes[ds.month - 1]
    elif contract.tenor_type == "S":
        code = ids.season_codes[contract.season_code or ""]
    else:
        code = ids.year_code
    yy = f"{ds.year % (10 ** ids.year_digits):0{ids.year_digits}d}"
    suffix = ids.suffix_letters[contract.tenor_type] if ids.period_suffix else ""
    return f"{root}{code}{yy}{suffix} {ids.yellow_key}"


def bloomberg_ticker_candidates(contract: FuturesContract, ids: BloombergCurveIds
                                ) -> tuple[str, ...]:
    """Every Bloomberg ticker that may name ``contract`` under the hub's form.

    ``tzt`` hubs have one ticker. ``pnx`` hubs have two: ``(listed, expired)``. The caller requests
    both and keeps the one that returns data (see ``LiveBloombergAdapter.get_curve_snapshot``). Nothing
    is guessed: a contract with data under both forms is an error.
    """
    if ids.form == "tzt":
        return (bloomberg_ticker(contract, ids),)
    if ids.form != "pnx":
        raise IdentifierUnknown(f"unknown ticker form {ids.form!r} for {ids.hub}")
    if contract.tenor_type != "M":
        raise IdentifierUnknown(f"{ids.hub} pnx form is verified for monthly contracts only")
    root = ids.roots.get("M")
    if not root or ids.status != IdentifierStatus.KNOWN.value:
        raise IdentifierUnknown(f"Bloomberg roots for {ids.hub} are not known")
    ds = contract.delivery_start
    code = ids.month_codes[ds.month - 1]
    yy = ds.year % 100
    letter = ids.suffix_letters["M"]
    listed = f"{root}{code}{yy % 10}{letter} {ids.yellow_key}"
    expired = f"{root}{code}{yy % 10}{letter}{yy:02d} {ids.yellow_key}"
    return (listed, expired)


def parse_bloomberg_ticker(ticker: str, ids: BloombergCurveIds, century: int = 2000
                           ) -> FuturesContract:
    """Inverse of :func:`bloomberg_ticker` for one hub. Raises :class:`ContractParseError`.

    For ``pnx`` hubs only the expired form carries a full year; the listed form (1-digit year) is
    ambiguous on its own and is rejected here.
    """
    body = ticker.strip()
    key = f" {ids.yellow_key}"
    if not body.endswith(key):
        raise ContractParseError(f"{ticker!r} does not end with {key!r}")
    body = body[: -len(key)]
    if ids.form == "pnx":
        root = ids.roots.get("M", "")
        letter = ids.suffix_letters["M"]
        m = re.fullmatch(rf"{re.escape(root)}([FGHJKMNQUVXZ])(\d)" + re.escape(letter) + r"(\d{2})",
                         body)
        if not m or m.group(1) not in ids.month_codes:
            raise ContractParseError(f"{ticker!r} is not an expired {ids.hub} pnx ticker "
                                     "(the listed 1-digit form cannot give the year)")
        yy = int(m.group(3))
        if yy % 10 != int(m.group(2)):
            raise ContractParseError(f"{ticker!r}: 1-digit and 2-digit years disagree")
        return FuturesContract.month(ids.hub, century + yy, ids.month_codes.index(m.group(1)) + 1)
    for tenor in FUTURES_TENORS:
        root = ids.roots.get(tenor)
        if not root or not body.startswith(root):
            continue
        rest = body[len(root):]
        if ids.period_suffix:
            letter = ids.suffix_letters[tenor]
            if not rest.endswith(letter):
                continue
            rest = rest[: -len(letter)]
        if len(rest) != 1 + ids.year_digits or not rest[1:].isdigit():
            continue
        code, year = rest[0], century + int(rest[1:])
        if tenor == "M" and code in ids.month_codes:
            return FuturesContract.month(ids.hub, year, ids.month_codes.index(code) + 1)
        if tenor == "Q" and code in ids.month_codes:
            month = ids.month_codes.index(code) + 1
            if month in (1, 4, 7, 10):
                return FuturesContract(_month_start(year, month), "Q", ids.hub)
        if tenor == "S":
            inv = {v: k for k, v in ids.season_codes.items()}
            if code in inv:
                return FuturesContract.season(ids.hub, year, inv[code])
        if tenor == "Y" and code == ids.year_code:
            return FuturesContract.calendar_year(ids.hub, year)
    raise ContractParseError(f"Cannot parse {ticker!r} as a {ids.hub} contract")


def hours_weights(months: Sequence[FuturesContract], tz: str = MARKET_TZ,
                  gas_day_start_hour: int = 6) -> np.ndarray:
    """Delivery-hours weights (sum to 1) for a strip of monthly contracts."""
    hours = np.array([m.delivery_hours(tz, gas_day_start_hour) for m in months], dtype=float)
    return hours / hours.sum()
