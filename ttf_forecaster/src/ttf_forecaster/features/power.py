"""Power-sector feature family (``power__*``), point-in-time.

Features (all at origin ``t``, from rows with ``publication_time <= t`` via the as-of resolvers):

* wind / solar forecast level and anomaly vs a prior-years normal (optionally capacity-scaled);
* residual load (load - wind - solar forecast): the transparent **power-sector gas-burn proxy**,
  level and day-on-day revision. It is an index in GW, not a gas volume: converting it needs the
  gas share of the marginal stack and an efficiency, both requires_user_input;
* French nuclear availability: forward schedule level, anomaly vs normal, revision vs one week ago;
* hydro proxy: reservoir level deviation from a prior-years normal, in percent;
* clean spark / clean dark spreads, their difference and the gas-to-coal switching gas price.

Prices are read as observation series (front-month settles) whose ``publication_time`` already
encodes the cross-commodity lag (PIT policy section 4). Efficiencies are ``requires_user_input``;
while null, the dependent features are NaN with a flag - never a guessed constant.

Conventions: EUR/MWh electric for spreads; ``eurusd`` = USD per EUR; API2 converted with the
configured MWh/t; emission factors in tCO2 per MWh thermal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    DataBundle,
    PitGrade,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.data.derived import load_derived_specs, with_derived_series
from ttf_forecaster.features.point_in_time import (
    latest_value_per_series,
    resolve_as_of,
    resolve_forecasts_as_of,
)
from ttf_forecaster.features.weather import (
    FlagLog,
    gas_year,
    gas_year_start,
    merged_params,
    origin_day,
    origin_index,
    window_offsets,
)
from ttf_forecaster.settings import get, load_config

FAMILY = "power"
POWER_CONFIG = "sources/power"


# ---------------------------------------------------------------------------------------------
# Pure economics (unit-tested by hand)
# ---------------------------------------------------------------------------------------------


def coal_eur_per_mwh_th(api2_usd_t: float, eurusd: float, mwh_per_t: float) -> float:
    """API2 USD/t -> EUR/MWh thermal. ``eurusd`` is USD per EUR."""
    return api2_usd_t / eurusd / mwh_per_t


def clean_spark_spread(power: float, gas: float, eua: float, eff_gas: float,
                       ef_gas: float) -> float:
    """CSS (EUR/MWh_e) = P_power - P_gas/eff - EUA * EF_gas/eff."""
    return power - gas / eff_gas - eua * ef_gas / eff_gas


def clean_dark_spread(power: float, coal_th: float, eua: float, eff_coal: float,
                      ef_coal: float) -> float:
    """CDS (EUR/MWh_e) = P_power - P_coal_th/eff - EUA * EF_coal/eff."""
    return power - coal_th / eff_coal - eua * ef_coal / eff_coal


def switching_gas_price(coal_th: float, eua: float, eff_gas: float, eff_coal: float,
                        ef_gas: float, ef_coal: float) -> float:
    """Gas price (EUR/MWh_th) at which CSS == CDS:
    ``eff_gas/eff_coal * (P_coal + EUA*EF_coal) - EUA*EF_gas``."""
    return eff_gas / eff_coal * (coal_th + eua * ef_coal) - eua * ef_gas


# ---------------------------------------------------------------------------------------------
# PIT helpers
# ---------------------------------------------------------------------------------------------


def _forecast_daily(frame: pd.DataFrame, series_id: str, as_of: pd.Timestamp,
                    lo: pd.Timestamp, hi: pd.Timestamp, min_points: int) -> pd.DataFrame:
    """Daily mean of the latest visible forecast per valid time for ``series_id`` (UTC days in
    [lo, hi]). Columns ``value, n, issue_min``; days with < ``min_points`` rows are NaN."""
    sub = frame.loc[(frame["series_id"] == series_id) & (frame["valid_time"] >= lo)
                    & (frame["valid_time"] < hi + pd.Timedelta(days=1))
                    & (frame["publication_time"] <= as_of)]
    if sub.empty:
        return pd.DataFrame(columns=["value", "n", "issue_min"])
    r = resolve_forecasts_as_of(sub, as_of, validate=False)
    day = r["valid_time"].dt.floor("D")
    g = r.assign(day=day).groupby("day")
    out = pd.DataFrame({"value": g["value"].mean(), "n": g["value"].count(),
                        "issue_min": g["issue_time"].min()})
    out.loc[out["n"] < min_points, "value"] = np.nan
    return out


def _window_mean(daily: pd.DataFrame, days: pd.DatetimeIndex) -> float:
    if daily.empty or not set(days) <= set(daily.index):
        return np.nan
    v = daily.loc[days, "value"]
    return float(v.mean()) if v.notna().all() else np.nan


def _observed(obs: pd.DataFrame, series_id: str, as_of: pd.Timestamp,
              lo: pd.Timestamp) -> pd.Series:
    sub = obs.loc[(obs["series_id"] == series_id) & (obs["event_time"] >= lo)
                  & (obs["publication_time"] <= as_of)]
    if sub.empty:
        return pd.Series(dtype=float)
    r = resolve_as_of(sub, as_of, validate=False)
    r = r.loc[r["event_time"] <= as_of]
    return pd.Series(r["value"].astype(float).to_numpy(),
                     index=pd.DatetimeIndex(r["event_time"]).floor("D"))


def prior_years_normal(history: pd.Series, target_days: pd.DatetimeIndex, cutoff: pd.Timestamp,
                       window_years: int, min_years: int, half_window: int,
                       capacity: pd.Series | None = None,
                       capacity_now: float | None = None) -> float:
    """Mean of ``history`` around the same calendar days in each of the previous ``window_years``
    years (+-``half_window`` days), using only days before ``cutoff`` (start of the origin's gas
    year). If ``capacity`` is given, each year's mean is scaled by ``capacity_now / capacity``
    (latest capacity at that time). NaN if fewer than ``min_years`` years contribute."""
    if history.empty:
        return np.nan
    year_means: list[float] = []
    k = pd.Timedelta(days=half_window)
    for y in range(1, window_years + 1):
        vals: list[float] = []
        ref = target_days[0] - pd.DateOffset(years=y)
        for d in target_days:
            dy = d - pd.DateOffset(years=y)
            lo, hi = dy - k, dy + k
            if hi >= cutoff:
                vals = []
                break
            seg = history.loc[(history.index >= lo) & (history.index <= hi)]
            vals.extend(seg.dropna().tolist())
        if not vals:
            continue
        m = float(np.mean(vals))
        if capacity is not None and capacity_now is not None:
            past = capacity.loc[capacity.index <= ref]
            if past.empty or past.iloc[-1] <= 0:
                continue
            m *= capacity_now / float(past.iloc[-1])
        year_means.append(m)
    return float(np.mean(year_means)) if len(year_means) >= min_years else np.nan


# ---------------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------------


def _econ(cfg: Mapping[str, Any], p: Mapping[str, Any], key: str) -> float | None:
    override = (p.get("economics") or {}).get(key)
    if override is not None:
        return float(override)
    node = get(cfg, "economics", key)
    return None if node.get("value") is None else float(node["value"])


def build_power_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object] | None = None
) -> pd.DataFrame:
    """Power features at each origin (``FeatureBuilder`` contract).

    Args:
        bundle: ``forecasts`` (wind/solar/load/nuclear schedules) and ``observations`` (actuals,
            capacities, reservoir levels, prices) in canonical schemas, internal series ids as in
            ``configs/sources/power.yaml``.
        origins: tz-aware origins.
        params: overrides of ``power.yaml::features``; ``economics`` {key: value} overrides
            ``power.yaml::economics`` values (e.g. efficiencies in tests, labelled synthetic).

    Returns:
        ``power__*`` frame indexed by ``origin_time``; ``attrs``: ``pit_grade``, ``flags``,
        ``units``, ``economics_status``.
    """
    cfg = load_config(POWER_CONFIG, params.get("config_dir") if params else None)  # type: ignore[arg-type]
    p = merged_params(get(cfg, "features"), params)
    idx = origin_index(origins)
    tz = str(p["market_tz"])
    # D-034: GW / GWh ids are derived from the raw MW components, point-in-time (data/derived.py).
    bundle = with_derived_series(bundle, load_derived_specs(cfg))
    fc, obs = bundle.forecasts, bundle.observations
    if len(fc):
        validate_frame(fc, FORECAST_SCHEMA)
    if len(obs):
        validate_frame(obs, OBSERVATION_SCHEMA)
    norms = p["normals"]
    start_month = int(norms["gas_year_start_month"])
    max_age = pd.Timedelta(hours=float(p["max_forecast_age_hours"]))
    hist_span = pd.DateOffset(years=int(norms["window_years"]) + 1)

    eff_g, eff_c = _econ(cfg, p, "gas_plant_efficiency"), _econ(cfg, p, "coal_plant_efficiency")
    ef_g, ef_c = _econ(cfg, p, "gas_emission_factor_t_per_mwh_th"), _econ(
        cfg, p, "coal_emission_factor_t_per_mwh_th")
    coal_cv = _econ(cfg, p, "coal_energy_mwh_per_t")

    def grade_of(frame: pd.DataFrame, ids: Sequence[str], default: PitGrade) -> PitGrade:
        sub = frame.loc[frame["series_id"].isin(ids)] if len(frame) else frame
        return default if sub.empty else worst_grade(list(sub["pit_grade"].unique()))

    flags = FlagLog()
    grades: dict[str, str] = {}
    units: dict[str, str] = {}
    rows: list[dict[str, float]] = []

    def put(out: dict[str, float], col: str, val: float, grade: PitGrade, unit: str,
            origin: pd.Timestamp, reason: str) -> None:
        out[col] = val
        grades[col] = grade.value
        units[col] = unit
        if np.isnan(val):
            flags.add(col, origin, reason)

    def fresh(daily: pd.DataFrame, days: pd.DatetimeIndex, origin: pd.Timestamp) -> bool:
        return (not daily.empty and set(days) <= set(daily.index)
                and bool((daily.loc[days, "issue_min"] >= origin - max_age).all()))

    for origin in idx:
        o_day = origin_day(origin, tz)
        cutoff = gas_year_start(gas_year(origin.tz_convert(tz), start_month), tz, start_month)
        out: dict[str, float] = {}

        def days_of(spec: Mapping[str, Any]) -> pd.DatetimeIndex:
            s, e = window_offsets(spec, o_day, tz)
            return pd.date_range(o_day + pd.Timedelta(days=s), o_day + pd.Timedelta(days=e),
                                 freq="D")

        # --- renewables ---------------------------------------------------------------------
        ren = p["renewables"]
        rdays = days_of(ren["window"])
        ren_level: dict[tuple[str, str], float] = {}
        for geo in ren["geographies"]:
            for tech in ren["techs"]:
                sid = ren["forecast_template"].format(geo=geo, tech=tech)
                aid = ren["actual_template"].format(geo=geo, tech=tech)
                cid = ren["capacity_template"].format(geo=geo, tech=tech)
                g = grade_of(fc, [sid], PitGrade.A)
                daily = _forecast_daily(fc, sid, origin, rdays[0], rdays[-1],
                                        int(ren["min_points_per_day"]))
                lvl = _window_mean(daily, rdays) if fresh(daily, rdays, origin) else np.nan
                ren_level[(geo, tech)] = lvl
                put(out, f"{FAMILY}__{geo}_{tech}_fc_gw", lvl, g, "GW", origin,
                    "missing_or_stale_forecast")
                hist = _observed(obs, aid, origin, origin - hist_span)
                cap = _observed(obs, cid, origin, pd.Timestamp("1900-01-01", tz="UTC"))
                cap_now = float(cap.iloc[-1]) if len(cap) else None
                normal = prior_years_normal(
                    hist, rdays, cutoff, int(norms["window_years"]), int(norms["min_years"]),
                    int(norms["smoothing_half_window_days"]),
                    cap if len(cap) else None, cap_now)
                ga = worst_grade([g, grade_of(obs, [aid, cid], PitGrade.B)])
                col = f"{FAMILY}__{geo}_{tech}_fc_anom_gw"
                put(out, col, lvl - normal, ga, "GW", origin,
                    "normal_unavailable" if np.isnan(normal) else "missing_or_stale_forecast")
                if not len(cap) and not np.isnan(normal):
                    flags.add(col, origin, "normal_not_capacity_adjusted")

        # --- residual load (gas-burn proxy) -----------------------------------------------------
        rl = p["residual_load"]
        ldays = days_of(rl["window"])
        step = pd.Timedelta(days=int(rl["revision_step_days"]))
        for geo in rl["geographies"]:
            lid = rl["load_template"].format(geo=geo)
            ids = [lid] + [ren["forecast_template"].format(geo=geo, tech=t) for t in ren["techs"]]
            g = grade_of(fc, ids, PitGrade.A)

            def residual(as_of: pd.Timestamp, check_fresh: bool) -> float:
                vals = []
                for sid in ids:
                    d = _forecast_daily(fc, sid, as_of, ldays[0], ldays[-1],
                                        int(ren["min_points_per_day"]))
                    if check_fresh and not fresh(d, ldays, as_of):
                        return np.nan
                    vals.append(_window_mean(d, ldays))
                return vals[0] - sum(vals[1:])

            now = residual(origin, True)
            prev = residual(origin - step, False)
            put(out, f"{FAMILY}__{geo}_residual_load_fc_gw", now, g, "GW", origin,
                "missing_or_stale_component")
            put(out, f"{FAMILY}__{geo}_residual_load_rev_gw", now - prev, g, "GW", origin,
                "missing_component_in_either_state")

        # --- French nuclear ---------------------------------------------------------------------
        nu = p["nuclear"]
        ndays = days_of(nu["window"])
        nstep = pd.Timedelta(days=int(nu["revision_step_days"]))
        g = grade_of(fc, [nu["forecast_series"]], PitGrade.A)
        nd = _forecast_daily(fc, nu["forecast_series"], origin, ndays[0], ndays[-1], 1)
        nlvl = _window_mean(nd, ndays)
        npd = _forecast_daily(fc, nu["forecast_series"], origin - nstep, ndays[0], ndays[-1], 1)
        nprev = _window_mean(npd, ndays)
        no_new = (not nd.empty and not npd.empty and set(ndays) <= set(nd.index)
                  and set(ndays) <= set(npd.index)
                  and bool((nd.loc[ndays, "issue_min"] == npd.loc[ndays, "issue_min"]).all()))
        put(out, f"{FAMILY}__fr_nuclear_avail_fc_gw", nlvl, g, "GW", origin, "missing_schedule")
        # An unchanged schedule IS information for REMIT (no new message = no change), so a zero
        # revision is legitimate here, unlike weather runs; flagged for transparency only.
        put(out, f"{FAMILY}__fr_nuclear_avail_rev_gw", nlvl - nprev, g, "GW", origin,
            "missing_schedule_in_either_state")
        if no_new:
            flags.add(f"{FAMILY}__fr_nuclear_avail_rev_gw", origin, "no_new_message_zero_revision")
        nhist = _observed(obs, nu["actual_series"], origin, origin - hist_span)
        nnorm = prior_years_normal(nhist, ndays, cutoff, int(norms["window_years"]),
                                   int(norms["min_years"]),
                                   int(norms["smoothing_half_window_days"]))
        put(out, f"{FAMILY}__fr_nuclear_avail_anom_gw", nlvl - nnorm,
            worst_grade([g, grade_of(obs, [nu["actual_series"]], PitGrade.B)]), "GW", origin,
            "normal_unavailable" if np.isnan(nnorm) else "missing_schedule")

        # --- hydro proxy -------------------------------------------------------------------------
        hy = p["hydro"]
        for sid in hy["series"]:
            name = sid.split(".")[1]
            h = _observed(obs, sid, origin, origin - hist_span)
            col = f"{FAMILY}__hydro_reservoir_anom_pct_{name}"
            gh = grade_of(obs, [sid], PitGrade.B)
            if h.empty or (o_day - h.index.max()).days > int(hy["max_staleness_days"]):
                put(out, col, np.nan, gh, "pct", origin, "stale_or_missing")
                continue
            last_day = pd.DatetimeIndex([h.index.max()])
            hn = prior_years_normal(h, last_day, cutoff, int(norms["window_years"]),
                                    int(norms["min_years"]), int(norms["smoothing_half_window_days"]))
            val = 100.0 * (float(h.iloc[-1]) - hn) / hn if hn and not np.isnan(hn) else np.nan
            put(out, col, val, gh, "pct", origin, "normal_unavailable")

        # --- spreads and switching ----------------------------------------------------------------
        pr = p["prices"]
        ids = [pr["power"], pr["gas"], pr["coal"], pr["carbon"], pr["eurusd"]]
        sub = obs.loc[obs["series_id"].isin(ids)
                      & (obs["event_time"] >= origin - pd.Timedelta(days=30))]
        latest = (latest_value_per_series(resolve_as_of(sub, origin, validate=False), origin,
                                          max_staleness=pd.Timedelta(
                                              days=int(pr["max_staleness_days"])))
                  if len(sub) else pd.Series(dtype=float))
        pw, gs, co, ca, fx = (float(latest.get(s, np.nan)) for s in ids)
        gp = worst_grade([grade_of(obs, ids, PitGrade.A)])
        econ_missing = [k for k, v in {"gas_plant_efficiency": eff_g,
                                       "coal_plant_efficiency": eff_c,
                                       "gas_emission_factor": ef_g,
                                       "coal_emission_factor": ef_c,
                                       "coal_energy_mwh_per_t": coal_cv}.items() if v is None]
        reason = ("requires_user_input:" + ",".join(econ_missing)) if econ_missing else \
            "missing_or_stale_price"
        if econ_missing:
            css = cds = sw = np.nan
        else:
            coal_th = coal_eur_per_mwh_th(co, fx, coal_cv)  # type: ignore[arg-type]
            css = clean_spark_spread(pw, gs, ca, eff_g, ef_g)  # type: ignore[arg-type]
            cds = clean_dark_spread(pw, coal_th, ca, eff_c, ef_c)  # type: ignore[arg-type]
            sw = switching_gas_price(coal_th, ca, eff_g, eff_c, ef_g, ef_c)  # type: ignore[arg-type]
        put(out, f"{FAMILY}__clean_spark_eur_mwh", css, gp, "EUR/MWh_e", origin, reason)
        put(out, f"{FAMILY}__clean_dark_eur_mwh", cds, gp, "EUR/MWh_e", origin, reason)
        put(out, f"{FAMILY}__spark_minus_dark_eur_mwh", css - cds, gp, "EUR/MWh_e", origin, reason)
        put(out, f"{FAMILY}__switch_gas_price_eur_mwh", sw, gp, "EUR/MWh_th", origin, reason)
        put(out, f"{FAMILY}__gas_minus_switch_eur_mwh", gs - sw, gp, "EUR/MWh_th", origin, reason)
        rows.append(out)

    frame = pd.DataFrame(rows, index=idx)
    frame.attrs["pit_grade"] = {c: grades[c] for c in frame.columns}
    frame.attrs["flags"] = flags.flags
    frame.attrs["units"] = {c: units[c] for c in frame.columns}
    frame.attrs["economics_status"] = {
        k: ("params_override" if (p.get("economics") or {}).get(k) is not None
            else get(cfg, "economics", k, "status"))
        for k in get(cfg, "economics")
    }
    return frame


__all__ = [
    "build_power_features",
    "clean_dark_spread",
    "clean_spark_spread",
    "coal_eur_per_mwh_th",
    "prior_years_normal",
    "switching_gas_price",
]
