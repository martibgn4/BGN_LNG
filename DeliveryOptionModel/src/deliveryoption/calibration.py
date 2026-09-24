"""Historical calibration: pair correlations and the vol-by-maturity profile.

CORRELATION, per month pair (M, M+g). The number that drives a delivery-month
option is the correlation between M and M+g of the *same* hub, and it is not
a constant: it falls as M approaches expiry (the front month decouples on
prompt fundamentals) and across a season boundary. So instead of rolled
generics, every past month k supplies its own contracts (hub_k, hub_k+g for
each hub), and only the days when contract k was as far from fixing as M is
between today and the decision date are used. Returns never straddle two
anchors, so there is no roll jump in the sample.

Returns are taken over `return_days` business days. TTF settles ~17:30 CET and
NYMEX NG at 14:30 ET (20:30 CET): daily returns are asynchronous across hubs
and understate the inter-hub correlation (Epps effect). The daily estimate is
kept alongside as a diagnostic.

VOL PROFILE, per hub. Realised vol of a contract as a function of its days to
expiry (Samuelson effect): the front month moves most. The pricing job uses
only the *shape* of this curve, to decide how much of a contract's implied
variance (which runs to its option expiry) falls before the decision date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from DeliveryOptionModel.src.deliveryoption.market_data import hub_contract

BUSINESS_DAYS_PER_YEAR = 252
BDH_CHUNK = 50


def leg_names(hubs: list[str]) -> list[str]:
    return [f"{h}_{tag}" for h in hubs for tag in ("M", "M1")]


def pair_key(m0: pd.Period, m1: pd.Period) -> str:
    return f"{m0}_{m1}"


@dataclass(frozen=True)
class PairCalibration:
    m0: pd.Period
    m1: pd.Period
    corr: pd.DataFrame           # at the configured return horizon
    corr_daily: pd.DataFrame     # 1-day, for the asynchrony diagnostic
    realised_vol: pd.Series      # annualised, same windows
    n_obs: int
    n_anchors: int
    anchors: list[str]
    window_days: tuple[int, int]
    return_days: int
    stale_share: pd.Series       # share of zero returns per leg: stale far-dated prints


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def anchor_months(m0: pd.Period, as_of: pd.Timestamp, lookback_months: int,
                  same_pair: bool) -> list[pd.Period]:
    """The last `lookback_months` months up to today, whatever M is.

    Counting back from today rather than from M matters for long-dated pairs:
    the 48 months before a 2031 M have not expired yet. Anchors whose window
    has not finished are dropped later, in window_returns.
    """
    last = as_of.to_period("M")
    months = [last - j for j in range(lookback_months)]
    if same_pair:
        months = [m for m in months if m.month == m0.month]
    return sorted(months)


def fetch_history(hub_specs: dict, hubs: list[str], months: set[pd.Period],
                  start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """PX_LAST for every hub's contract in `months`, and each contract's expiry."""
    from xbbg import blp

    tickers = sorted({hub_contract(hub_specs[h], m) for h in hubs for m in months})
    expiry = blp.bdp(tickers, "LAST_TRADEABLE_DT")["last_tradeable_dt"]
    frames = []
    # Chunked: a hundred-plus tickers over several years in one request is slow
    # enough to time out.
    for i in range(0, len(tickers), BDH_CHUNK):
        px = blp.bdh(tickers[i:i + BDH_CHUNK], "PX_LAST", start, end)
        if px.empty:
            continue
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.droplevel(1)
        frames.append(px)
    px = pd.concat(frames, axis=1)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()   # the chunks' date sets differ, so the union is unsorted
    return px, pd.to_datetime(expiry)


# ---------------------------------------------------------------------------
# pair correlation
# ---------------------------------------------------------------------------

def window_returns(px: pd.DataFrame, expiry: pd.Series, hub_specs: dict, hubs: list[str],
                   anchors: list[pd.Period], gap: int, window_days: tuple[int, int],
                   return_days: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Stacked non-overlapping log returns, one block per anchor month."""
    lo, hi = window_days
    names = leg_names(hubs)
    blocks = []
    for k in anchors:
        cols = [hub_contract(hub_specs[h], m) for h in hubs for m in (k, k + gap)]
        if not all(c in px.columns and c in expiry.index for c in cols):
            continue
        fix = min(expiry[c] for c in cols[::2])     # the k contracts
        start, end = fix - pd.Timedelta(days=hi), fix - pd.Timedelta(days=lo)
        if end >= as_of:
            continue  # window not complete yet
        sub = px.loc[start:end, cols].dropna()
        # Sample back from the window end, so the observation nearest the
        # decision point is always in.
        sub = sub.iloc[::-1].iloc[::return_days].iloc[::-1]
        if len(sub) < 3:
            continue
        r = np.log(sub).diff().dropna()
        r.columns = names
        r.index = pd.MultiIndex.from_product([[str(k)], r.index], names=["anchor", "date"])
        blocks.append(r)
    if not blocks:
        raise ValueError("No complete anchor windows - widen lookback or the window")
    return pd.concat(blocks)


def calibrate_pair(px, expiry, hub_specs, hubs, m0: pd.Period, m1: pd.Period, anchors,
                   window_days, return_days, as_of) -> PairCalibration:
    gap = (m1 - m0).n
    r = window_returns(px, expiry, hub_specs, hubs, anchors, gap, window_days, return_days, as_of)
    r1 = window_returns(px, expiry, hub_specs, hubs, anchors, gap, window_days, 1, as_of)
    return PairCalibration(
        m0=m0, m1=m1,
        corr=r.corr(),
        corr_daily=r1.corr(),
        realised_vol=r.std() * np.sqrt(BUSINESS_DAYS_PER_YEAR / return_days),
        n_obs=len(r),
        n_anchors=r.index.get_level_values(0).nunique(),
        anchors=sorted(r.index.get_level_values(0).unique()),
        window_days=window_days,
        return_days=return_days,
        stale_share=(r == 0).mean(),
    )


# ---------------------------------------------------------------------------
# vol-by-maturity profile
# ---------------------------------------------------------------------------

def vol_profile(px: pd.DataFrame, expiry: pd.Series, hub_specs: dict, hubs: list[str],
                bucket_edges: list[int], return_days: int, as_of: pd.Timestamp) -> pd.DataFrame:
    """Realised vol per hub and days-to-expiry bucket, pooled over contracts.

    Uses the root mean square of returns rather than the standard deviation:
    with a handful of returns per contract per bucket, demeaning would bias the
    estimate down.
    """
    rows = []
    for h in hubs:
        root = hub_specs[h]["future_root"]
        rets, taus = [], []
        for tkr in px.columns:
            if not tkr.startswith(root) or tkr not in expiry.index:
                continue
            s = px[tkr].loc[:min(expiry[tkr], as_of)].dropna()
            s = s.iloc[::-1].iloc[::return_days].iloc[::-1]
            if len(s) < 2:
                continue
            r = np.log(s).diff().dropna()
            rets.append(r.to_numpy())
            taus.append((expiry[tkr] - r.index).days.to_numpy())
        r, tau = np.concatenate(rets), np.concatenate(taus)
        edges = list(bucket_edges) + [np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (tau >= lo) & (tau < hi)
            if not mask.any():
                continue
            rows.append({"hub": h, "tau_lo_days": lo, "tau_hi_days": hi,
                         "realised_vol": float(np.sqrt(np.mean(r[mask] ** 2)
                                                       * BUSINESS_DAYS_PER_YEAR / return_days)),
                         "n_obs": int(mask.sum())})
    return pd.DataFrame(rows)
