"""Implied REALISED Norwegian outage from the total exit flow (user spec, 2026-10-20).

For gas day D (Amsterdam date):
    capacity_proxy(D) = 95th percentile of the flows of the previous 30 days (D-30 .. D-1), needing >= 20 of them
    shortfall(D)      = capacity_proxy(D) - flow(D)
    implied_outage(D) = max(0, shortfall(D))                     [mcm/d]
    implied_outage_event(D) = 1 when shortfall(D) > threshold (10 mcm/d)

Strictly past data: the proxy uses ``shift(1)`` of the rolling window, so day D never sees its own flow. A flow for day D
is usable from D+1 07:00 Europe/Amsterdam (intraday nominations revise within the day), so every output for D is
published then: grade B. Parameters: configs/sources/flow_outage.yaml.

This module is pure: it takes the flow series and returns the observation rows. ``pipelines/implied_outage.py`` reads the
store and writes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.units import GasQuality
from ttf_forecaster.data.base import MARKET_TZ, OBSERVATION_SCHEMA, PitGrade, validate_frame

SOURCE = "implied_outage"
ENTITLEMENT = "licensed_bloomberg"  # derived from a Bloomberg series
PUBLICATION_HOUR = 7  # D+1 07:00 Europe/Amsterdam


def publication_of(gas_day: pd.Series) -> pd.Series:
    """D+1 07:00 Europe/Amsterdam, as UTC, for each gas day (a midnight-of-local-day timestamp in UTC)."""
    local = pd.DatetimeIndex(gas_day).tz_localize(None)
    pub_local = (local + pd.Timedelta(days=1, hours=PUBLICATION_HOUR)).tz_localize(
        MARKET_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return pd.Series(pub_local.tz_convert("UTC"), index=gas_day.index)


def local_days(flow: pd.DataFrame) -> pd.Series:
    """The gas day of each flow observation, as a naive Amsterdam date (the adapter stamps local midnight)."""
    ts = pd.to_datetime(flow["event_time"], utc=True)
    return ts.dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize()


def daily_flow(flow: pd.DataFrame) -> pd.Series:
    """One flow per gas day (Amsterdam), in mcm/d, sorted. A duplicated day keeps the latest publication."""
    f = flow.assign(day=local_days(flow)).sort_values(["day", "publication_time"])
    return f.groupby("day")["value"].last().astype(float)


def implied_outage(flow_mcmd: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Columns: capacity_proxy, flow, shortfall, outage_mcmd, event. Indexed by gas day (naive Amsterdam date).

    Days without a flow give NaN. The proxy needs ``min_periods`` past flows, otherwise it is NaN (no outage claimed).
    """
    window = int(cfg["window_days"])
    q = float(cfg["percentile"])
    min_p = int(cfg["min_periods"])
    threshold = float(cfg["threshold_mcmd"])
    full = flow_mcmd.asfreq("D")  # every calendar day; a gap stays NaN
    proxy = full.rolling(window, min_periods=min_p).quantile(q).shift(1)  # shift: strictly past days only
    shortfall = proxy - full
    out = pd.DataFrame({
        "capacity_proxy": proxy,
        "flow": full,
        "shortfall": shortfall,
        "outage_mcmd": shortfall.clip(lower=0.0).where(shortfall.notna()),
        "event": (shortfall > threshold).astype(float).where(shortfall.notna()),
    })
    return out


def observation_rows(flow: pd.DataFrame, cfg: Mapping[str, Any], gas_quality: GasQuality,
                     ingestion_time: pd.Timestamp) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows: flow GWh/d twin, implied outage mcm/d, its GWh/d twin, and the event flag.

    ``flow`` is the stored observation frame of ``cfg['flow_series']`` (mcm/d). Each row's publication is D+1 07:00
    Amsterdam; the flow twin is published with its own flow day (the same rule).
    """
    daily = daily_flow(flow)
    calc = implied_outage(daily, cfg)
    days = calc.index.intersection(daily.index)
    frames = []
    gcv = gas_quality.gcv_kwh_per_m3

    def block(series_id: str, values: pd.Series, unit: str, use_days: pd.DatetimeIndex) -> pd.DataFrame:
        v = values.reindex(use_days)
        ok = v.notna()
        idx = use_days[ok.to_numpy()]
        n = len(idx)
        event = pd.DatetimeIndex(idx).tz_localize(MARKET_TZ, ambiguous="NaT", nonexistent="shift_forward")
        event = event.tz_convert("UTC")
        return pd.DataFrame({
            "series_id": [series_id] * n,
            "event_time": event,
            "value": v[ok].to_numpy(dtype=float),
            "geography": ["NO"] * n,
            "publication_time": publication_of(pd.Series(idx, index=range(n))).to_numpy(),
            "ingestion_time": [ingestion_time] * n,
            "revision": np.zeros(n, dtype="int64"),
            "unit": [unit] * n,
            "source": [SOURCE] * n,
            "quality_flag": ["ok"] * n,
            "entitlement": [ENTITLEMENT] * n,
            "pit_grade": [PitGrade.B.value] * n,
            "publication_time_estimated": [False] * n,
            "source_timezone": [MARKET_TZ] * n,
        })

    frames.append(block(cfg["flow_gwhd_series"], daily * gcv, "GWh/d", days))
    frames.append(block(cfg["outage_series"], calc["outage_mcmd"], "MCM/d", days))
    frames.append(block(cfg["outage_gwhd_series"], calc["outage_mcmd"] * gcv, "GWh/d", days))
    frames.append(block(cfg["event_series"], calc["event"], "flag", days))
    out = pd.concat(frames, ignore_index=True)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["publication_time"] = pd.to_datetime(out["publication_time"], utc=True)
    out["ingestion_time"] = pd.to_datetime(out["ingestion_time"], utc=True)
    return validate_frame(out, OBSERVATION_SCHEMA)
