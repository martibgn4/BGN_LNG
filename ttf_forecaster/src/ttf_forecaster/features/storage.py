"""Storage feature family (``storage__*``), FeatureBuilder contract of ``data/base.py``.

Every origin is computed from ``resolve_as_of(observations, origin)``: the latest vintage published
by the origin. Seasonal normals and the projection profile use only gas years strictly before
``min(gas year of the origin, gas year of the last observed day)``. Values older than
``max_staleness_days`` are NaN, never carried forward. Unavailable inputs give NaN, never 0.

Columns (units in the name):

==========================================  ==================================================
storage__level_twh                          last published inventory
storage__full_pct                           inventory / working gas volume
storage__net_injection_gwhd                 injection − withdrawal on the last day (flows if
                                            published, else inventory first difference)
storage__net_injection_{w}d_gwhd            mean of the last ``net_flow_window_days`` days
storage__normal_full_pct                    same-day fill, mean of prior gas years
storage__deviation_pct_pts                  full_pct − normal_full_pct
storage__deviation_twh                      deviation × current WGV
storage__deviation_z                        deviation / across-year std of the normal
storage__normal_net_injection_gwhd          prior-year same-day net flow (% WGV) × current WGV
storage__injection_vs_normal_gwhd           net_injection_{w}d − normal_net_injection
storage__days_of_cover                      inventory / trailing mean demand (days)
storage__policy_gap_pct_pts                 policy target − full_pct (injection season only)
storage__policy_required_injection_gwhd     energy still needed / days to deadline
storage__proj_eos_<scenario>_pct            projected end-of-season fill (deterministic)
storage__proj_eos_dev_vs_normal_pct_pts     proj_eos_normal − seasonal normal at season end
storage__proj_eos_p10/p50/p90_pct           bootstrap quantiles (bootstrap scenario)
storage__prob_eos_below_<thr>pct            P(end fill < thr); NaN when thr not for this season
==========================================  ==================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    DataBundle,
    PitGrade,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.data.fundamentals import grades_of, to_daily
from ttf_forecaster.features.point_in_time import resolve_as_of, to_utc_timestamp
from ttf_forecaster.fundamentals.scenarios import (
    InsufficientHistory,
    estimate_net_flow_profile,
    gas_year,
    load_scenario_config,
    net_flow_pct_of_wgv,
    scenarios_from_config,
    seasonal_normal,
)
from ttf_forecaster.fundamentals.storage_projection import project_storage, season_end
from ttf_forecaster.settings import load_config

FAMILY = "storage"


def default_storage_params(config_dir: str | None = None) -> dict[str, Any]:
    """``storage_features`` + ``policy_storage_target`` from ``sources/fundamentals.yaml`` and the
    scenarios config, merged into one params mapping."""
    cfg = load_config("sources/fundamentals", config_dir)
    params = dict(cfg["storage_features"])
    params["policy_storage_target"] = cfg.get("policy_storage_target")
    params["scenario_config"] = load_scenario_config(config_dir)
    return params


def origin_local_day(origin: pd.Timestamp) -> pd.Timestamp:
    """Naive local calendar date of a tz-aware origin."""
    return to_utc_timestamp(origin).tz_convert(MARKET_TZ).tz_localize(None).normalize()


def _last_fresh(s: pd.Series, day: pd.Timestamp, max_stale: int) -> tuple[pd.Timestamp | None, float]:
    past = s.loc[s.index <= day].dropna()
    if past.empty or (day - past.index[-1]).days > max_stale:
        return None, np.nan
    return past.index[-1], float(past.iloc[-1])


def _policy_regime(policy: Mapping[str, Any] | None, day: pd.Timestamp) -> Mapping[str, Any] | None:
    if not policy:
        return None
    for reg in policy.get("regimes", []):
        if pd.Timestamp(reg["effective_from"]) <= day <= pd.Timestamp(reg["effective_to"]):
            return reg
    return None


def _column_names(params: Mapping[str, Any]) -> list[str]:
    w = int(params["net_flow_window_days"])
    cols = [
        "level_twh", "full_pct", "net_injection_gwhd", f"net_injection_{w}d_gwhd",
        "normal_full_pct", "deviation_pct_pts", "deviation_twh", "deviation_z",
        "normal_net_injection_gwhd", "injection_vs_normal_gwhd", "days_of_cover",
        "policy_gap_pct_pts", "policy_required_injection_gwhd",
    ]
    if params.get("projection_enabled", True):
        scen_cfg = params["scenario_config"]
        cols += [f"proj_eos_{s}_pct" for s in params["projection_scenarios"]]
        cols += ["proj_eos_dev_vs_normal_pct_pts"]
        cols += [f"proj_eos_p{int(round(q * 100))}_pct" for q in scen_cfg["bootstrap"]["quantiles"]]
        thr = sorted({float(t) for lst in scen_cfg["thresholds_pct"].values() for t in lst})
        cols += [f"prob_eos_below_{t:g}pct" for t in thr]
    return [f"{FAMILY}__{c}" for c in cols]


def _one_origin(resolved: pd.DataFrame, origin: pd.Timestamp, p: Mapping[str, Any],
                cols: list[str]) -> dict[str, float]:
    out = dict.fromkeys(cols, np.nan)
    day = origin_local_day(origin)
    stale = int(p["max_staleness_days"])
    w = int(p["net_flow_window_days"])
    inv = to_daily(resolved, p["inventory_series"])
    wgv_s = to_daily(resolved, p["working_gas_volume_series"])
    inj = to_daily(resolved, p["injection_series"])
    wd = to_daily(resolved, p["withdrawal_series"])
    dem = to_daily(resolved, p["demand_series"]) if p.get("demand_series") else pd.Series(dtype=float)

    last_day, level = _last_fresh(inv, day, stale)
    if last_day is None:
        return out
    _, wgv = _last_fresh(wgv_s, day, stale)
    f = lambda name: f"{FAMILY}__{name}"  # noqa: E731
    out[f("level_twh")] = level
    full = level / wgv * 100.0 if np.isfinite(wgv) and wgv > 0 else np.nan
    out[f("full_pct")] = full

    # net flow history (GWh/d): published injection/withdrawal if available, else inventory diff
    if not inj.empty and not wd.empty:
        net = (inj - wd).dropna()
    else:
        full_idx = pd.date_range(inv.index.min(), inv.index.max(), freq="D")
        net = (inv.reindex(full_idx).diff() * 1000.0).dropna()
    net = net.loc[net.index <= day]
    if not net.empty and (day - net.index[-1]).days <= stale:
        out[f("net_injection_gwhd")] = float(net.iloc[-1])
        recent = net.loc[net.index > net.index[-1] - pd.Timedelta(days=w)]
        out[f(f"net_injection_{w}d_gwhd")] = float(recent.mean())

    before = int(min(gas_year(day), gas_year(last_day)))
    fill_hist = (inv / wgv_s.reindex(inv.index) * 100.0).dropna()
    nwin, nmin = int(p["normal_window_days"]), int(p["normal_min_years"])
    norm = seasonal_normal(fill_hist, pd.DatetimeIndex([last_day]), before,
                           window_days=nwin, min_years=nmin).iloc[0]
    out[f("normal_full_pct")] = norm["mean"]
    dev = full - norm["mean"]
    out[f("deviation_pct_pts")] = dev
    out[f("deviation_twh")] = dev * wgv / 100.0
    out[f("deviation_z")] = dev / norm["std"] if norm["std"] and norm["std"] > 0 else np.nan

    net_pct = net_flow_pct_of_wgv(net, wgv_s).dropna() if not wgv_s.empty else pd.Series(dtype=float)
    if np.isfinite(wgv):
        nn = seasonal_normal(net_pct, pd.DatetimeIndex([last_day]), before,
                             window_days=nwin, min_years=nmin).iloc[0]
        out[f("normal_net_injection_gwhd")] = nn["mean"] / 100.0 * wgv * 1000.0
        out[f("injection_vs_normal_gwhd")] = (
            out[f(f"net_injection_{w}d_gwhd")] - out[f("normal_net_injection_gwhd")])

    if not dem.empty:
        d_last, _ = _last_fresh(dem, day, stale)
        if d_last is not None:
            win = dem.loc[(dem.index > d_last - pd.Timedelta(days=int(p["demand_window_days"])))
                          & (dem.index <= d_last)].dropna()
            mean_dem = float(win.mean())
            if mean_dem > 0:
                out[f("days_of_cover")] = level * 1000.0 / mean_dem

    reg = _policy_regime(p.get("policy_storage_target"), day)
    if reg is not None and np.isfinite(full):
        mm, dd = (int(x) for x in str(reg["deadline_mmdd"]).split("-"))
        deadline = pd.Timestamp(year=day.year, month=mm, day=dd)
        season_start = pd.Timestamp(year=day.year, month=4, day=1)
        if season_start <= day < deadline:
            target = float(reg["target_pct"])
            out[f("policy_gap_pct_pts")] = target - full
            days_left = (deadline - last_day).days
            needed_gwh = max(0.0, (target / 100.0 * wgv - level) * 1000.0)
            out[f("policy_required_injection_gwhd")] = needed_gwh / days_left

    if p.get("projection_enabled", True) and np.isfinite(wgv) and wgv > 0:
        _projection_features(out, p, net_pct, level, last_day, wgv, day, before, fill_hist)
    return out


def _projection_features(out: dict[str, float], p: Mapping[str, Any], net_pct: pd.Series,
                         level: float, last_day: pd.Timestamp, wgv: float, day: pd.Timestamp,
                         before: int, fill_hist: pd.Series) -> None:
    scen_cfg = p["scenario_config"]
    scenarios = scenarios_from_config(scen_cfg)
    try:
        profile = estimate_net_flow_profile(
            net_pct, int(min(before, gas_year(day))),
            smoothing_window_days=int(scen_cfg["profile"]["smoothing_window_days"]),
            min_years=int(scen_cfg["profile"]["min_years"]))
    except InsufficientHistory:
        return
    level = min(max(level, 0.0), wgv)
    seed_key = (day.toordinal(),)
    for name in p["projection_scenarios"]:
        boot = name == p.get("bootstrap_scenario", "normal")
        try:
            res = project_storage(level0_twh=level, start_day=last_day, wgv_twh=wgv,
                                  profile=profile, scenario=scenarios[name], origin_day=day,
                                  scen_cfg=scen_cfg, bootstrap=boot, seed_key=seed_key)
        except InsufficientHistory:
            continue
        out[f"{FAMILY}__proj_eos_{name}_pct"] = res.end_fill_pct
        if boot:
            for q, v in res.quantiles_fill_pct.items():
                out[f"{FAMILY}__proj_eos_p{int(round(q * 100))}_pct"] = v
            for t, prob in res.prob_below_pct.items():
                out[f"{FAMILY}__prob_eos_below_{t:g}pct"] = prob
        if name == "normal":
            end_day = season_end(day, scen_cfg.get("season_ends"))
            nm = seasonal_normal(fill_hist, pd.DatetimeIndex([end_day]), before,
                                 window_days=int(p["normal_window_days"]),
                                 min_years=int(p["normal_min_years"])).iloc[0]["mean"]
            out[f"{FAMILY}__proj_eos_dev_vs_normal_pct_pts"] = res.end_fill_pct - nm


def _grades(observations: pd.DataFrame, p: Mapping[str, Any], cols: list[str]) -> dict[str, str]:
    def g(ids: list[str]) -> str:
        found = grades_of(observations, [i for i in ids if i])
        return worst_grade(found).value if found else PitGrade.B.value

    core = [p["inventory_series"], p["working_gas_volume_series"]]
    flows = [p["injection_series"], p["withdrawal_series"], p["inventory_series"]]
    out = {}
    for c in cols:
        name = c.split("__", 1)[1]
        if name.startswith("days_of_cover"):
            out[c] = g([p["inventory_series"], p.get("demand_series", "")])
        elif "injection" in name or name.startswith(("proj_", "prob_")):
            out[c] = g(core + flows)
        else:
            out[c] = g(core)
    return out


def build_storage_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object] | None = None
) -> pd.DataFrame:
    """Storage features at each origin (FeatureBuilder contract).

    Args:
        bundle: all vintages; only ``bundle.observations`` is read.
        origins: tz-aware forecast origins.
        params: overrides of ``default_storage_params()`` (keys as in
            ``sources/fundamentals.yaml`` ``storage_features``).

    Returns:
        Frame indexed by ``origin_time`` (UTC) with ``storage__*`` columns and
        ``attrs["pit_grade"]`` (column -> worst input grade). Grade is B when no input rows exist
        (the configured grade of every storage series); a missing series yields NaN columns.
    """
    p: dict[str, Any] = default_storage_params()
    p.update(params or {})
    idx = pd.DatetimeIndex([to_utc_timestamp(o) for o in origins], name="origin_time")
    cols = _column_names(p)
    ids = [p["inventory_series"], p["working_gas_volume_series"], p["injection_series"],
           p["withdrawal_series"], p.get("demand_series")]
    obs = bundle.observations
    validate_frame(obs, OBSERVATION_SCHEMA)
    sub = obs.loc[obs["series_id"].isin([i for i in ids if i])]
    if len(idx):
        sub = sub.loc[sub["publication_time"] <= idx.max()]
    rows = []
    for origin in idx:
        resolved = resolve_as_of(sub, origin, validate=False)
        rows.append(_one_origin(resolved, origin, p, cols))
    frame = pd.DataFrame(rows, index=idx, columns=cols).astype(float)
    frame.attrs["pit_grade"] = _grades(sub, p, cols)
    return frame
