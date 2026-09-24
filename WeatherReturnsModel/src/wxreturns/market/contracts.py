"""Contract identity, expiry, delivery windows and the roll calendar.

The generic ``NG1 Comdty`` is not a tradeable return series, and the two
reasons are handled separately here.

**Roll gaps.** On roll day the generic jumps from the expiring contract to the
next one. That jump is a change of underlying, not a return, and in gas it is
routinely several percent. Chaining returns WITHIN a contract and stitching is
the fix; see `market/returns.py`.

**Window drift.** NG1 on 20 December references January gas. On 28 December,
after expiry, it references February. A fixed "next 15 days of weather" feature
therefore points at the wrong gas for part of every month, and it is worst in
the last week - exactly when the front contract is most weather-sensitive. So
the delivery window is resolved per date from the contract actually held.

Expiries are CONSTRUCTED here and VALIDATED against ``fut_ticker`` in
`data/bloomberg.py`. Construction alone is not trusted: exchange holidays move
expiries, and a rule that is right most of the time inserts a fake return into
the series several times a year.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wxreturns.config import get

log = logging.getLogger(__name__)

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
CODE_TO_MONTH = {code: i + 1 for i, code in enumerate(MONTH_CODES)}
MONTH_TO_CODE = {i + 1: code for i, code in enumerate(MONTH_CODES)}

_FIFTEENTH = 15


def _business_days_before(day: pd.Timestamp, count: int) -> pd.Timestamp:
    """Step back ``count`` weekdays. Exchange holidays are NOT applied.

    Deliberate: a wrong holiday calendar is worse than none, because it looks
    authoritative. `validate_roll_calendar` catches the difference against
    Bloomberg, which is the real answer.
    """
    return (day - pd.tseries.offsets.BDay(count)).normalize()


def expiry_date(commodity: str, month: int, year: int) -> pd.Timestamp:
    """Last trade date under the commodity's declared rule."""
    cfg = get("targets", "contracts")[commodity]
    rule = cfg["expiry_rule"]

    if rule == "n_business_days_before_month_start":
        offset = cfg["expiry_offset_business_days"]
        return _business_days_before(pd.Timestamp(year=year, month=month, day=1),
                                     offset)
    if rule == "business_day_before_fifteenth":
        return _business_days_before(
            pd.Timestamp(year=year, month=month, day=_FIFTEENTH), 1)
    if rule == "exchange_calendar":
        raise NotImplementedError(
            f"{commodity} expiries follow the exchange calendar and cannot be "
            "constructed from a rule. Pull them with blp.bdp(ticker, "
            "'last_tradeable_dt') and cache them before using this commodity."
        )
    raise ValueError(f"unknown expiry_rule {rule!r} for {commodity}")


def delivery_window(commodity: str, month: int, year: int
                    ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The span of weather a contract references.

    For gas this is the calendar delivery month. For ags it is the growing
    season of the crop year the contract belongs to, which is a different idea
    entirely and lives in `weather/crop.py`.
    """
    cfg = get("targets", "contracts")[commodity]
    kind = cfg["delivery_window"]

    if kind == "contract_month":
        start = pd.Timestamp(year=year, month=month, day=1)
        return start, start + pd.offsets.MonthEnd(1)
    if kind == "crop_year":
        from wxreturns.weather.crop import crop_year_window
        return crop_year_window(commodity, month, year)
    raise ValueError(f"unknown delivery_window {kind!r} for {commodity}")


def contract_sequence(commodity: str, start: pd.Timestamp, end: pd.Timestamp
                      ) -> pd.DataFrame:
    """Every listed contract expiring in the window, with its delivery span."""
    cfg = get("targets", "contracts")[commodity]
    months = [CODE_TO_MONTH[c] for c in cfg["contract_months"]]

    records: list[dict] = []
    for year in range(start.year, end.year + 2):
        for month in months:
            expiry = expiry_date(commodity, month, year)
            if expiry < start or expiry > end + pd.DateOffset(years=1):
                continue
            window_start, window_end = delivery_window(commodity, month, year)
            records.append({
                "commodity": commodity,
                "contract_month": month,
                "contract_year": year,
                "code": f"{MONTH_TO_CODE[month]}{str(year)[-2:]}",
                "expiry": expiry,
                "window_start": window_start,
                "window_end": window_end,
            })
    if not records:
        raise ValueError(
            f"no {commodity} contracts expire between {start.date()} and "
            f"{end.date()}")
    return (pd.DataFrame.from_records(records)
            .sort_values("expiry")
            .reset_index(drop=True))


def roll_calendar(commodity: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Which contract occupies each slot on each date.

    A contract is stepped off ``roll.days_before_expiry`` business days ahead
    of its last trade date, so the series is never holding an expiring contract
    into its final illiquid sessions.
    """
    if len(dates) == 0:
        raise ValueError("no dates given")
    lead = get("targets", "roll", "days_before_expiry")
    slots = get("targets", "slots")

    sequence = contract_sequence(commodity, dates.min(), dates.max())
    sequence["roll_off"] = sequence["expiry"].map(
        lambda d: _business_days_before(d, lead))

    records: list[dict] = []
    expiries = sequence["roll_off"].to_numpy()
    for when in dates:
        # The first contract not yet rolled off is the front; the next listed
        # one is front+1.
        position = int(np.searchsorted(expiries, np.datetime64(when), side="left"))
        for slot_name, slot in slots.items():
            index = position + slot["generic"] - 1
            if index >= len(sequence):
                continue
            row = sequence.iloc[index]
            records.append({
                "commodity": commodity,
                "slot": slot_name,
                "obs_date": when,
                "code": row["code"],
                "contract_month": int(row["contract_month"]),
                "contract_year": int(row["contract_year"]),
                "expiry": row["expiry"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "days_to_expiry": int((row["expiry"] - when).days),
            })

    if not records:
        raise ValueError(f"no {commodity} contract was live on any given date")
    frame = pd.DataFrame.from_records(records)
    region = get("targets", "contracts")[commodity]["region"]
    frame["region"] = region
    return frame


def weather_windows(commodity: str, dates: pd.DatetimeIndex,
                    mode: str | None = None, max_lead: int | None = None
                    ) -> pd.DataFrame:
    """Which span of weather to aggregate for each slot on each date.

    THE REACHABILITY PROBLEM, found on real data on 2026-09-07 and the reason
    this function exists at all.

    The natural choice is the contract's own delivery window (`contract_delivery`
    below), and for NG at short leads it is UNREACHABLE - not rarely, but
    never. NG expires three business days before delivery starts and this
    calendar rolls five business days before that, so the front contract always
    references a month beginning at least eight business days out. With the
    archive capped at seven days of lead, no forecast target ever lands inside
    the front contract's delivery month. The feature is empty by construction,
    not by accident of the sample.

    So the mode has to match the lead depth available:

      ``contract_delivery``  the contract's own month. Economically the right
                             object, and it needs leads of roughly 15+ days.
                             Available only once live collection has built up
                             the far leads history cannot supply.
      ``balance_of_month``   from tomorrow to the end of the current calendar
                             month. Works at leads 1-7. Defensible rather than
                             a fallback: near-term weather drives the storage
                             draw, and the storage trajectory is what the front
                             contract prices.
      ``forecast_horizon``   tomorrow to tomorrow + max_lead. The most direct
                             reading of "what changed in the forecast", with no
                             calendar structure imposed at all.
    """
    mode = mode or get("features", "revisions", "window_mode")
    if max_lead is None:
        max_lead = get("sources", "open_meteo", "previous_runs", "max_lead_days")

    calendar = roll_calendar(commodity, dates)
    if mode == "contract_delivery":
        reach = (calendar["window_start"] - calendar["obs_date"]).dt.days
        if (reach > max_lead).all():
            raise ValueError(
                f"{commodity}: no delivery window is within {max_lead} days of "
                f"any observation date (nearest is {int(reach.min())} days "
                "out), so a contract-window revision cannot be formed at all. "
                "This is structural, not a data gap: the contract expires "
                "before its own delivery month begins. Use "
                "window_mode: balance_of_month until live collection has built "
                "up leads beyond 15 days."
            )
        return calendar

    out = calendar.copy()
    if mode == "balance_of_month":
        out["window_start"] = out["obs_date"] + pd.Timedelta(days=1)
        out["window_end"] = out["obs_date"] + pd.offsets.MonthEnd(0)
        # On the last day or two of a month the balance is empty or negative;
        # roll onto the next month, which is what the market is trading by then.
        spent = out["window_end"] < out["window_start"]
        out.loc[spent, "window_end"] = (out.loc[spent, "obs_date"]
                                        + pd.offsets.MonthEnd(1))
    elif mode == "forecast_horizon":
        out["window_start"] = out["obs_date"] + pd.Timedelta(days=1)
        out["window_end"] = out["obs_date"] + pd.Timedelta(days=max_lead)
    else:
        raise ValueError(
            f"unknown window mode {mode!r}; expected contract_delivery, "
            "balance_of_month or forecast_horizon")

    out["window_mode"] = mode
    return out


def window_reachability(commodity: str, dates: pd.DatetimeIndex,
                        max_lead: int | None = None) -> dict:
    """How far each slot's delivery window sits beyond the forecast horizon.

    A diagnostic, so the reachability problem above is a number rather than a
    surprise in the middle of a fit.
    """
    if max_lead is None:
        max_lead = get("sources", "open_meteo", "previous_runs", "max_lead_days")
    calendar = roll_calendar(commodity, dates)
    calendar["days_to_window"] = (calendar["window_start"]
                                  - calendar["obs_date"]).dt.days

    report = {}
    for slot, block in calendar.groupby("slot"):
        reachable = int((block["days_to_window"] <= max_lead).sum())
        report[slot] = {
            "min_days_to_window": int(block["days_to_window"].min()),
            "max_days_to_window": int(block["days_to_window"].max()),
            "reachable_at_lead": max_lead,
            "reachable_rows": reachable,
            "total_rows": int(len(block)),
        }
    return report


def validate_roll_calendar(constructed: pd.DataFrame, observed: pd.DataFrame
                           ) -> dict:
    """Gate 3 check: does the constructed calendar match Bloomberg?

    ``observed`` comes from `RollCalendarFetcher`, i.e. from ``fut_ticker``.
    Any mismatch is a real disagreement about which contract was front, and it
    means a return in the series spans two different underlyings.
    """
    left = constructed[constructed["slot"] == "front"].copy()
    left["obs_date"] = pd.to_datetime(left["obs_date"])
    right = observed.copy()
    right["obs_date"] = pd.to_datetime(right["obs_date"])

    merged = left.merge(right[["obs_date", "contract"]], on="obs_date",
                        how="inner")
    if merged.empty:
        raise ValueError(
            "no overlapping dates between the constructed and observed roll "
            "calendars")

    # fut_ticker returns forms like 'NGF26 Comdty'; compare on the month code
    # and the year digits, which is the part both representations agree on.
    def _code_of(ticker: str) -> str:
        head = ticker.split(" ")[0]
        return head[-3:] if head[-2:].isdigit() else head[-2:]

    merged["observed_code"] = merged["contract"].map(_code_of)
    merged["matches"] = merged.apply(
        lambda r: r["code"][0] == r["observed_code"][0]
        and r["code"][-2:] == r["observed_code"][-2:], axis=1)

    mismatches = merged[~merged["matches"]]
    return {
        "checked": int(len(merged)),
        "matches": int(merged["matches"].sum()),
        "mismatches": int(len(mismatches)),
        "mismatch_dates": [str(d.date()) for d in
                           mismatches["obs_date"].head(10)],
        "passed": bool(mismatches.empty),
    }
