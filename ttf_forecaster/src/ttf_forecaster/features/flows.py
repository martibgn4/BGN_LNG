"""Pipeline-flow and LNG send-out feature family (``flows__*``), FeatureBuilder contract.

Each origin reads ``resolve_as_of(observations, origin)``. Outage series are vintaged daily
unavailable capacity whose ``event_time`` may lie in the future (planned maintenance published
ahead); forward-looking outage features read only rows already published by the origin.

Columns (GWh/d unless stated):

======================================  ===========================================================
flows__no_flow_gwhd                     last Norwegian pipeline flow (staleness-limited)
flows__no_flow_chg_{k}d_gwhd            flow minus flow k days earlier
flows__no_utilisation_pct               flow / technical capacity
flows__no_outage_adj_capacity_gwhd      capacity − unavailable on the last flow day
flows__no_unavailable_next{h}d_gwhd     mean published unavailable capacity over the next h days
flows__lng_sendout_gwhd                 last EU LNG send-out
flows__lng_sendout_chg_{k}d_gwhd        send-out change over k days
flows__lng_utilisation_pct              send-out / send-out capacity
flows__supply_surprise_gwhd             mean(last r days) − mean(previous b days) of total supply
flows__supply_cum_dev_{n}d_twh          Σ over last n days of (supply − prior-years seasonal normal)
======================================  ===========================================================

Total supply is the sum of ``supply_series``; a day missing any component is NaN (never 0).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    OBSERVATION_SCHEMA,
    DataBundle,
    PitGrade,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.data.fundamentals import grades_of, to_daily
from ttf_forecaster.features.point_in_time import resolve_as_of, to_utc_timestamp
from ttf_forecaster.features.storage import origin_local_day
from ttf_forecaster.fundamentals.scenarios import gas_year, seasonal_normal
from ttf_forecaster.settings import load_config

FAMILY = "flows"


def default_flows_params(config_dir: str | None = None) -> dict[str, Any]:
    """``flows_features`` block of ``sources/fundamentals.yaml``."""
    return dict(load_config("sources/fundamentals", config_dir)["flows_features"])


def _column_names(p: Mapping[str, Any]) -> list[str]:
    lags = [int(k) for k in p["change_lags_days"]]
    h, n = int(p["outage_forward_days"]), int(p["cumulative_window_days"])
    cols = ["no_flow_gwhd", *[f"no_flow_chg_{k}d_gwhd" for k in lags], "no_utilisation_pct",
            "no_outage_adj_capacity_gwhd", f"no_unavailable_next{h}d_gwhd", "lng_sendout_gwhd",
            *[f"lng_sendout_chg_{k}d_gwhd" for k in lags], "lng_utilisation_pct",
            "supply_surprise_gwhd", f"supply_cum_dev_{n}d_twh"]
    return [f"{FAMILY}__{c}" for c in cols]


def _past(s: pd.Series, day: pd.Timestamp) -> pd.Series:
    return s.loc[s.index <= day].dropna()


def _level_and_changes(s: pd.Series, day: pd.Timestamp, stale: int,
                       lags: list[int]) -> tuple[pd.Timestamp | None, float, dict[int, float]]:
    past = _past(s, day)
    if past.empty or (day - past.index[-1]).days > stale:
        return None, np.nan, {k: np.nan for k in lags}
    last = past.index[-1]
    v = float(past.iloc[-1])
    changes = {}
    for k in lags:
        ref = last - pd.Timedelta(days=k)
        changes[k] = v - float(past.loc[ref]) if ref in past.index else np.nan
    return last, v, changes


def _one_origin(resolved: pd.DataFrame, origin: pd.Timestamp, p: Mapping[str, Any],
                cols: list[str]) -> dict[str, float]:
    out = dict.fromkeys(cols, np.nan)
    day = origin_local_day(origin)
    stale = int(p["max_staleness_days"])
    lags = [int(k) for k in p["change_lags_days"]]
    f = lambda name: f"{FAMILY}__{name}"  # noqa: E731

    no = to_daily(resolved, p["norway_flow_series"])
    cap = to_daily(resolved, p["norway_capacity_series"])
    unav = to_daily(resolved, p["norway_unavailable_series"])
    last, v, ch = _level_and_changes(no, day, stale, lags)
    if last is not None:
        out[f("no_flow_gwhd")] = v
        for k in lags:
            out[f(f"no_flow_chg_{k}d_gwhd")] = ch[k]
        c = float(cap.get(last, np.nan))
        if np.isfinite(c) and c > 0:
            out[f("no_utilisation_pct")] = v / c * 100.0
            u = float(unav.get(last, np.nan))
            out[f("no_outage_adj_capacity_gwhd")] = c - u if np.isfinite(u) else np.nan
    h = int(p["outage_forward_days"])
    fwd_days = pd.date_range(day + pd.Timedelta(days=1), periods=h, freq="D")
    fwd = unav.reindex(fwd_days)
    if fwd.notna().all():  # every day of the window must have a published value
        out[f(f"no_unavailable_next{h}d_gwhd")] = float(fwd.mean())

    lng = to_daily(resolved, p["lng_sendout_series"])
    lcap = to_daily(resolved, p["lng_capacity_series"])
    last_l, vl, chl = _level_and_changes(lng, day, stale, lags)
    if last_l is not None:
        out[f("lng_sendout_gwhd")] = vl
        for k in lags:
            out[f(f"lng_sendout_chg_{k}d_gwhd")] = chl[k]
        c = float(lcap.get(last_l, np.nan))
        if np.isfinite(c) and c > 0:
            out[f("lng_utilisation_pct")] = vl / c * 100.0

    parts = [to_daily(resolved, s) for s in p["supply_series"]]
    if all(not s.empty for s in parts):
        supply = pd.concat(parts, axis=1)
        supply = supply.sum(axis=1, min_count=supply.shape[1]).where(supply.notna().all(axis=1))
        supply = _past(supply, day)
        if not supply.empty and (day - supply.index[-1]).days <= stale:
            end = supply.index[-1]
            full = supply.reindex(pd.date_range(end - pd.Timedelta(days=400), end, freq="D"))
            r, b = int(p["surprise_recent_days"]), int(p["surprise_baseline_days"])
            recent, base = full.iloc[-r:], full.iloc[-(r + b):-r]
            if recent.notna().all() and base.notna().sum() >= b // 2:
                out[f("supply_surprise_gwhd")] = float(recent.mean() - base.mean())
            n = int(p["cumulative_window_days"])
            window = full.iloc[-n:]
            if window.notna().all():
                before = int(min(gas_year(day), gas_year(window.index[0])))
                norm = seasonal_normal(supply, window.index, before,
                                       window_days=int(p["normal_window_days"]),
                                       min_years=int(p["normal_min_years"]))["mean"]
                if norm.notna().all():
                    n_name = f"supply_cum_dev_{n}d_twh"
                    out[f(n_name)] = float((window - norm.to_numpy()).sum() / 1000.0)
    return out


def _grades(observations: pd.DataFrame, p: Mapping[str, Any], cols: list[str]) -> dict[str, str]:
    def g(ids: list[str]) -> str:
        found = grades_of(observations, ids)
        return worst_grade(found).value if found else PitGrade.B.value

    no_ids = [p["norway_flow_series"], p["norway_capacity_series"]]
    out = {}
    for c in cols:
        name = c.split("__", 1)[1]
        if name.startswith("no_unavailable") or name.startswith("no_outage_adj"):
            out[c] = g([p["norway_unavailable_series"], *no_ids])
        elif name.startswith("no_"):
            out[c] = g(no_ids)
        elif name.startswith("lng_"):
            out[c] = g([p["lng_sendout_series"], p["lng_capacity_series"]])
        else:
            out[c] = g(list(p["supply_series"]))
    return out


def build_flows_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object] | None = None
) -> pd.DataFrame:
    """Flow features at each origin (FeatureBuilder contract); see module docstring."""
    p: dict[str, Any] = default_flows_params()
    p.update(params or {})
    idx = pd.DatetimeIndex([to_utc_timestamp(o) for o in origins], name="origin_time")
    cols = _column_names(p)
    ids = {p["norway_flow_series"], p["norway_capacity_series"], p["norway_unavailable_series"],
           p["lng_sendout_series"], p["lng_capacity_series"], *p["supply_series"]}
    obs = bundle.observations
    validate_frame(obs, OBSERVATION_SCHEMA)
    sub = obs.loc[obs["series_id"].isin(ids)]
    if len(idx):
        sub = sub.loc[sub["publication_time"] <= idx.max()]
    rows = [_one_origin(resolve_as_of(sub, o, validate=False), o, p, cols) for o in idx]
    frame = pd.DataFrame(rows, index=idx, columns=cols).astype(float)
    frame.attrs["pit_grade"] = _grades(sub, p, cols)
    return frame
