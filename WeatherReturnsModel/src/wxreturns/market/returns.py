"""Returns that could actually have been earned.

THE ROLL GAP. A generic settle series steps from the expiring contract to the
next one on roll day. That step is a change of underlying, and in gas it is
routinely several percent - larger than anything weather does in a day. Left
in, it becomes the biggest "return" in the sample, it recurs monthly, and it
lines up with the calendar, so any seasonal feature will fit it. Returns here
are therefore computed WITHIN a contract and the roll day return is dropped
rather than smoothed over.

THE FORWARD-RETURN CONVENTION. ``forward_return(t, h)`` is the return from the
close on t to the close h business days later. It is the thing being predicted,
so it is stamped at t and must never be shifted the other way. Features stamped
at t predict it; nothing stamped after t may touch it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wxreturns.config import get

log = logging.getLogger(__name__)


def _price_column(frame: pd.DataFrame) -> str:
    field = get("sources", "bloomberg", "settle_field").lower()
    if field in frame.columns:
        return field
    if "settle" in frame.columns:
        return "settle"
    raise KeyError(
        f"no price column found; expected {field!r} or 'settle' in "
        f"{sorted(frame.columns)}")


def daily_returns(settles: pd.DataFrame, roll: pd.DataFrame) -> pd.DataFrame:
    """Within-contract daily returns for each slot.

    ``settles`` is the generic series from `SettleFetcher`; ``roll`` says which
    contract each generic held on each date. A day whose contract differs from
    the previous day's is a roll and its return is dropped.
    """
    price = _price_column(settles)
    frame = settles.copy()
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])

    slots = get("targets", "slots")
    generic_to_slot = {slot["generic"]: name for name, slot in slots.items()}
    frame["slot"] = frame["generic"].map(generic_to_slot)
    frame = frame.dropna(subset=["slot"])

    keys = ["commodity", "slot", "obs_date"]
    merged = frame.merge(roll[keys + ["code", "days_to_expiry"]], on=keys,
                         how="left")

    unmatched = merged["code"].isna().sum()
    if unmatched:
        log.warning("%d settle rows have no roll-calendar entry and are dropped",
                    unmatched)
    merged = merged.dropna(subset=["code"])

    if get("targets", "return_type") != "log":
        raise ValueError("only log returns are implemented")

    merged = merged.sort_values(["commodity", "slot", "obs_date"])
    grouped = merged.groupby(["commodity", "slot"], sort=False)

    previous_price = grouped[price].shift(1)
    previous_code = grouped["code"].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["return"] = np.log(merged[price] / previous_price)

    # The roll day: the contract changed, so the price ratio spans two
    # different underlyings and is not a return anyone could have earned.
    is_roll = merged["code"] != previous_code
    merged.loc[is_roll, "return"] = np.nan
    merged["is_roll_day"] = is_roll

    non_positive = (merged[price] <= 0.0).sum()
    if non_positive:
        raise ValueError(
            f"{non_positive} non-positive settle(s); log returns are undefined. "
            "Check the field and the contract before proceeding.")

    return merged


def forward_returns(returns: pd.DataFrame, horizons: list[int] | None = None
                    ) -> pd.DataFrame:
    """Cumulative forward log return over each horizon, stamped at t.

    Computed by summing daily log returns forward, which keeps the roll-day
    NaNs propagating: a horizon spanning a roll has no clean tradeable return
    and is left missing rather than being bridged. Bridging would quietly
    reinsert the roll gap this module exists to remove.
    """
    horizons = horizons or get("targets", "horizons_business_days")
    frame = returns.sort_values(["commodity", "slot", "obs_date"]).copy()
    grouped = frame.groupby(["commodity", "slot"], sort=False)

    for horizon in horizons:
        # Sum of the next `horizon` daily returns, i.e. r_{t+1} + ... + r_{t+h}.
        forward = (grouped["return"]
                   .transform(lambda s, h=horizon:
                              s.shift(-1).rolling(h, min_periods=h).sum()
                              .shift(-(h - 1))))
        frame[f"fwd_return_{horizon}d"] = forward

    return frame


def realised_volatility(returns: pd.DataFrame, window: int | None = None
                        ) -> pd.DataFrame:
    """Forward realised volatility, annualised. The volatility model's target.

    Forward rather than trailing: the claim being tested is that today's
    ensemble spread predicts the volatility to come, not the one just past.
    """
    window = window or get("model", "volatility", "realised_vol_window_days")
    annual = get("model", "volatility", "annualisation_days")

    frame = returns.sort_values(["commodity", "slot", "obs_date"]).copy()
    grouped = frame.groupby(["commodity", "slot"], sort=False)
    forward = (grouped["return"]
               .transform(lambda s: s.shift(-1)
                          .rolling(window, min_periods=window).std()
                          .shift(-(window - 1))))
    frame["realised_vol_forward"] = forward * float(np.sqrt(annual))
    return frame


def transaction_cost(commodity: str) -> float:
    """One-way cost as a fraction of price, from targets.yaml."""
    costs = get("targets", "costs")
    return float(costs["by_commodity"].get(commodity, costs["default_one_way"]))


def net_pnl(signal: pd.Series, realised: pd.Series, commodity: str) -> pd.Series:
    """P&L of a unit signal net of cost, charged on position CHANGE.

    Charging per day held would penalise a slow signal for doing nothing, which
    is not how a book works. The cost is paid when the position moves.
    """
    positions = signal.fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    return positions * realised - turnover * transaction_cost(commodity)
