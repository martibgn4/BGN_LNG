"""Weather feature family (``weather__*``), point-in-time.

Thesis (WeatherReturnsModel, research spec H2): forecast **levels** are priced; forecast
**revisions** are the news; ensemble spread speaks to volatility, not direction. Levels and
anomalies are therefore built as controls, revisions as the candidate signal.

Every value at origin ``t`` is computed from rows with ``publication_time <= t`` only, read through
``features/point_in_time.py``:

* forecast state at ``t``      = ``resolve_forecasts_as_of(forecasts, t)``
* forecast state one step back = ``resolve_forecasts_as_of(forecasts, t - step)``
* observations                  = ``resolve_as_of(observations, t)`` (ERA5, grade B, lag D+5)

A **revision** is ``DD(window | state t) - DD(window | state t - step)`` over the *same* fixed valid
days, so it is lead-matched (each valid day is compared at lead L vs L + step). It is NaN, with a
flag, when either state lacks a valid day, when no newer run arrived (the issue time did not
advance: a zero here would be a fabricated "no change"), or when the latest run is stale.

**Normals** use observed weather from gas years strictly before the origin's gas year, resolved as
of the start of that gas year (deterministic and conservative), with a minimum year count.

Degree days are computed per point and then weighted (the HDD function is convex in temperature).
Observed-weather features carry the ``obs_`` prefix and are never a forecast substitute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    DataBundle,
    PitGrade,
    PointInTimeViolation,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.data.weather import (
    WeatherRegion,
    load_weather_config,
    point_series_id,
    region_from_config,
)
from ttf_forecaster.features.point_in_time import resolve_as_of, resolve_forecasts_as_of
from ttf_forecaster.settings import ConfigError, _deep_merge, get

FAMILY = "weather"
_STAT_MEMBERS = frozenset({"mean", "spread", "median", "p10", "p25", "p75", "p90", "control"})


# ---------------------------------------------------------------------------------------------
# Small shared helpers (also used by features/power.py)
# ---------------------------------------------------------------------------------------------


def heating_degree_days(temp_c: np.ndarray | pd.Series, reference_c: float,
                        threshold_c: float) -> np.ndarray:
    """HDD (degC·day) per day: ``reference - T`` if ``T <= threshold`` else 0; NaN stays NaN.

    Eurostat convention: reference 18 C, threshold 15 C. ``threshold == reference`` gives the
    simple ``max(0, reference - T)`` form.
    """
    t = np.asarray(temp_c, dtype=float)
    out = np.where(t <= threshold_c, reference_c - t, 0.0)
    return np.where(np.isnan(t), np.nan, out)


def cooling_degree_days(temp_c: np.ndarray | pd.Series, reference_c: float,
                        threshold_c: float) -> np.ndarray:
    """CDD (degC·day): ``T - reference`` if ``T >= threshold`` else 0 (Eurostat: 21 C / 24 C)."""
    t = np.asarray(temp_c, dtype=float)
    out = np.where(t >= threshold_c, t - reference_c, 0.0)
    return np.where(np.isnan(t), np.nan, out)


def origin_index(origins: pd.DatetimeIndex | Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
    """Origins as a tz-aware UTC index named ``origin_time``; naive origins are rejected."""
    idx = pd.DatetimeIndex(origins)
    if idx.tz is None:
        raise PointInTimeViolation("origins must be tz-aware (e.g. 19:00 Europe/Amsterdam)")
    return idx.tz_convert("UTC").rename("origin_time")


def origin_day(origin: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    """The origin's local calendar date as a UTC-midnight timestamp (daily valid_time convention:
    daily values are stamped at 00:00 UTC of their UTC day)."""
    return pd.Timestamp(origin.tz_convert(market_tz).date()).tz_localize("UTC")


def gas_year(ts: pd.Timestamp, start_month: int = 10) -> int:
    """Gas year label: the calendar year in which it starts (Oct-Sep for start_month=10)."""
    return ts.year if ts.month >= start_month else ts.year - 1


def gas_year_start(year: int, market_tz: str, start_month: int = 10) -> pd.Timestamp:
    """Local midnight of the gas year's first day, in UTC."""
    return pd.Timestamp(year=year, month=start_month, day=1).tz_localize(market_tz).tz_convert("UTC")


def noleap_doy(ts: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Day of year on a 365-day calendar; 29 Feb maps to 28 Feb (doy 59)."""
    idx = pd.DatetimeIndex(ts)
    month, day = idx.month.to_numpy(), idx.day.to_numpy()
    day = np.where((month == 2) & (day == 29), 28, day)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    return cum[month - 1] + day


@dataclass
class FlagLog:
    """Per-column, per-origin reasons a value is NaN (``attrs['flags']``)."""

    flags: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, column: str, origin: pd.Timestamp, reason: str) -> None:
        self.flags.setdefault(column, {})[origin.isoformat()] = reason


def window_offsets(spec: Mapping[str, Any], o_day: pd.Timestamp, market_tz: str) -> tuple[int, int]:
    """(start, end) day offsets of a window spec; ``end: weekday_sunday`` = next Sunday (capped)."""
    start = int(spec["start"])
    end_raw = spec["end"]
    if end_raw == "weekday_sunday":
        wd = o_day.weekday()  # Mon=0..Sun=6 of the local origin date
        end = 7 if wd == 6 else 6 - wd
        end = max(start, min(end, int(spec.get("end_max", end))))
    else:
        end = int(end_raw)
    if end < start:
        raise ConfigError(f"window {dict(spec)} has end < start")
    return start, end


def window_max_end(spec: Mapping[str, Any]) -> int:
    """Largest end offset a window spec can take (for lead validation)."""
    if spec["end"] == "weekday_sunday":
        return int(spec.get("end_max", 7))
    return int(spec["end"])


def merged_params(defaults: Mapping[str, Any], params: Mapping[str, object] | None) -> dict[str, Any]:
    """Config defaults deep-merged with call-time params (params win)."""
    return _deep_merge(dict(defaults), dict(params or {}))


# ---------------------------------------------------------------------------------------------
# Daily weighted degree days from resolved rows
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _DDSpec:
    hdd_ref: float
    hdd_thr: float
    cdd_ref: float
    cdd_thr: float
    preaggregated: bool
    series_point: Mapping[str, str]
    weights: Mapping[str, float]


def _daily_dd(rows: pd.DataFrame, spec: _DDSpec, time_col: str, issue_col: str) -> pd.DataFrame:
    """Weighted daily HDD/CDD per (member, day) from resolved rows.

    A day is NaN unless every point has a non-null value (no silent re-weighting over a subset).
    Returns columns ``member, day, hdd, cdd, issue_min, issue_max``.
    """
    cols = ["member", "day", "hdd", "cdd", "issue_min", "issue_max"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    r = pd.DataFrame(
        {
            "member": rows["member"].to_numpy() if "member" in rows else "obs",
            "day": rows[time_col].dt.floor("D").to_numpy(),
            "value": rows["value"].astype(float).to_numpy(),
            "issue": rows[issue_col].to_numpy(),
        }
    )
    r["day"] = pd.DatetimeIndex(r["day"]).tz_localize("UTC") if pd.DatetimeIndex(
        r["day"]).tz is None else r["day"]
    if spec.preaggregated:
        r["hdd"] = r["value"]
        r["cdd"] = np.nan
        r["w"] = 1.0
        n_points = 1
    else:
        r["point"] = rows["series_id"].map(spec.series_point).to_numpy()
        r["w"] = r["point"].map(spec.weights).astype(float)
        r["hdd"] = heating_degree_days(r["value"], spec.hdd_ref, spec.hdd_thr)
        r["cdd"] = cooling_degree_days(r["value"], spec.cdd_ref, spec.cdd_thr)
        n_points = len(spec.weights)
    r["whdd"] = r["w"] * r["hdd"]
    r["wcdd"] = r["w"] * r["cdd"]
    g = r.groupby(["member", "day"])
    out = pd.DataFrame(
        {
            "hdd": g["whdd"].sum(min_count=1),
            "cdd": g["wcdd"].sum(min_count=1),
            "n": g["value"].count(),
            "issue_min": g["issue"].min(),
            "issue_max": g["issue"].max(),
        }
    ).reset_index()
    incomplete = out["n"] < n_points
    out.loc[incomplete, ["hdd", "cdd"]] = np.nan
    if spec.preaggregated:
        out["cdd"] = np.nan
    return out[cols]


# ---------------------------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------------------------


def seasonal_normal_table(daily: pd.DataFrame, value_cols: Sequence[str], gas_year_label: int,
                          window_years: int, min_years: int, half_window: int,
                          start_month: int) -> pd.DataFrame | None:
    """Day-of-year normal (365 rows, circularly smoothed) from ``daily`` (columns ``day`` + values)
    using only gas years ``[G - window_years, G - 1]``. ``None`` if fewer than ``min_years`` years
    have data. ``daily`` must already be resolved as of a time no later than the origin."""
    if daily.empty:
        return None
    d = daily.dropna(subset=list(value_cols), how="all").copy()
    days = pd.DatetimeIndex(d["day"])
    gy = np.where(days.month >= start_month, days.year, days.year - 1)
    keep = (gy < gas_year_label) & (gy >= gas_year_label - window_years)
    d = d.loc[keep]
    if d.empty or len(np.unique(gy[keep])) < min_years:
        return None
    d["doy"] = noleap_doy(d["day"])
    raw = d.groupby("doy")[list(value_cols)].mean().reindex(range(1, 366))
    k = int(half_window)
    padded = pd.concat([raw.iloc[-k:], raw, raw.iloc[:k]]) if k else raw
    smooth = padded.rolling(2 * k + 1, center=True, min_periods=1).mean()
    out = smooth.iloc[k: k + 365] if k else smooth
    out.index = range(1, 366)
    return out


# ---------------------------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------------------------


def _feature_params(params: Mapping[str, object] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_weather_config(params.get("config_dir") if params else None)  # type: ignore[arg-type]
    return cfg, merged_params(get(cfg, "features"), params)


def _validate_windows(p: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    windows = p["windows"]
    max_lead = int(p.get("max_lead_days", get(cfg, "open_meteo", "previous_runs", "max_lead_days")))
    step = int(p["revision"]["step_days"])
    live_only = set((p.get("live_only_windows") or {}).keys())
    allow_live = bool(p.get("allow_live_only_windows", False))
    for group in ("level_windows", "ensemble_windows"):
        for w in p.get(group) or []:
            if w in live_only and not allow_live:
                raise ConfigError(f"{group}: window {w} needs leads > {max_lead} days and can only "
                                  "be live-accumulated; set allow_live_only_windows to use it")
    for w in list(p["revision"]["windows"]) + list(p.get("level_windows") or []):
        if w not in windows and w not in live_only:
            raise ConfigError(f"unknown weather window {w!r}")
    for w in p["revision"]["windows"]:
        if w in live_only and not allow_live:
            raise ConfigError(f"revision window {w} is live-only")
        spec = windows.get(w) or p["live_only_windows"][w]
        if window_max_end(spec) + step > max_lead and not allow_live:
            raise ConfigError(
                f"revision window {w} ends at D+{window_max_end(spec)}; with step {step} the "
                f"previous state needs lead {window_max_end(spec) + step} > max lead {max_lead}. "
                "Open-Meteo returns nulls there silently - shorten the window."
            )
    for w in p.get("level_windows") or []:
        spec = windows.get(w) or p["live_only_windows"][w]
        if window_max_end(spec) > max_lead and not allow_live:
            raise ConfigError(f"level window {w} exceeds max lead {max_lead} days")


def _region_spec(cfg: Mapping[str, Any], p: Mapping[str, Any]) -> tuple[WeatherRegion, _DDSpec,
                                                                         list[str], list[str], str]:
    region = p.get("region_obj") or region_from_config(cfg, str(p["region"]))
    dd = p["degree_days"]
    if dd.get("convention") not in ("eurostat", "simple"):
        raise ConfigError(f"degree_days.convention {dd.get('convention')!r} not supported")
    hdd_thr = float(dd["hdd"]["threshold_c"]) if dd["convention"] == "eurostat" else float(
        dd["hdd"]["reference_c"])
    cdd_thr = float(dd["cdd"]["threshold_c"]) if dd["convention"] == "eurostat" else float(
        dd["cdd"]["reference_c"])
    if region.input == "preaggregated_hdd":
        spec = _DDSpec(float(dd["hdd"]["reference_c"]), hdd_thr, float(dd["cdd"]["reference_c"]),
                       cdd_thr, True, {}, {})
        fc_ids = [str(region.hdd_forecast_series_id)]
        obs_ids = [str(region.hdd_observed_series_id)] if region.hdd_observed_series_id else []
        return region, spec, fc_ids, obs_ids, "preaggregated"
    if p.get("weights"):
        w = {str(k): float(v) for k, v in dict(p["weights"]).items()}
        missing = {pt.point_id for pt in region.points} - set(w)
        if missing or any(v <= 0 for v in w.values()):
            raise ConfigError(f"params.weights must give a positive weight for every point; "
                              f"missing {sorted(missing)}")
        tot = sum(w[pt.point_id] for pt in region.points)
        weights = {pt.point_id: w[pt.point_id] / tot for pt in region.points}
        label = "params_override"
    else:
        weights, label = region.weights(
            allow_synthetic_equal=bool(p.get("allow_synthetic_equal_weights", False))
        )
    fc_ids = [point_series_id(cfg, pt.point_id, "forecast") for pt in region.points]
    obs_ids = [point_series_id(cfg, pt.point_id, "observed") for pt in region.points]
    series_point = {sid: pt.point_id for sid, pt in zip(fc_ids, region.points, strict=True)}
    series_point.update({sid: pt.point_id for sid, pt in zip(obs_ids, region.points, strict=True)})
    spec = _DDSpec(float(dd["hdd"]["reference_c"]), hdd_thr, float(dd["cdd"]["reference_c"]),
                   cdd_thr, False, series_point, weights)
    return region, spec, fc_ids, obs_ids, label


def _grade(frame: pd.DataFrame, default: PitGrade) -> PitGrade:
    if frame.empty:
        return default
    return worst_grade(list(frame["pit_grade"].unique()))


def build_weather_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object] | None = None
) -> pd.DataFrame:
    """Weather features at each origin (``FeatureBuilder`` contract).

    Args:
        bundle: all vintages; uses ``forecasts`` (point temperatures or a pre-aggregated HDD
            series) and ``observations`` (ERA5-type observed temperature / HDD).
        origins: tz-aware forecast origins.
        params: overrides of ``configs/sources/weather.yaml::features``. Notable keys:
            ``region``; ``weights`` ({point_id: weight}, labelled ``params_override``);
            ``allow_synthetic_equal_weights`` (synthetic tests only); ``deterministic`` and
            ``ensemble`` ({model, member/min_members}); ``degree_days``; ``windows``;
            ``revision`` ({step_days, windows, max_run_age_hours}); ``level_windows``;
            ``ensemble_windows``; ``normals``; ``observed``; ``allow_live_only_windows``.

    Returns:
        Frame indexed by ``origin_time`` with ``weather__*`` columns (degC·day sums over the
        window). ``attrs``: ``pit_grade`` {column: grade}, ``flags`` {column: {origin: reason}},
        ``weights_label``, ``heating_season_only`` [columns], ``units`` {column: unit}.

    Raises:
        WeightsUnavailable: region weights not confirmed and synthetic weights not allowed.
        ConfigError: window/lead arithmetic impossible (revision beyond the 7-day archive).
    """
    cfg, p = _feature_params(params)
    _validate_windows(p, cfg)
    idx = origin_index(origins)
    tz = str(p["market_tz"])
    region, spec, fc_ids, obs_ids, weights_label = _region_spec(cfg, p)
    det_model, det_member = str(p["deterministic"]["model"]), str(p["deterministic"]["member"])
    ens_model = str(p["ensemble"]["model"])
    min_members = int(p["ensemble"]["min_members"])
    step = pd.Timedelta(days=int(p["revision"]["step_days"]))
    max_age = pd.Timedelta(hours=float(p["revision"]["max_run_age_hours"]))
    norms = p["normals"]
    start_month = int(norms["gas_year_start_month"])
    obs_p = p["observed"]

    fc_all = bundle.forecasts
    if len(fc_all):
        validate_frame(fc_all, FORECAST_SCHEMA)
    fc = fc_all.loc[fc_all["series_id"].isin(fc_ids)]
    det = fc.loc[(fc["model"] == det_model) & (fc["member"] == det_member)]
    ens = fc.loc[(fc["model"] == ens_model) & ~fc["member"].isin(_STAT_MEMBERS)]
    obs_all = bundle.observations
    if len(obs_all):
        validate_frame(obs_all, OBSERVATION_SCHEMA)
    obs = obs_all.loc[obs_all["series_id"].isin(obs_ids)]

    g_det = _grade(det, PitGrade.A)
    g_ens = _grade(ens, PitGrade.A)
    g_obs = _grade(obs, PitGrade.B)
    g_anom = worst_grade([g_det, g_obs])

    windows = {**(p.get("live_only_windows") or {}), **p["windows"]}
    flags = FlagLog()
    rows: list[dict[str, float]] = []
    normal_cache: dict[int, pd.DataFrame | None] = {}
    grades: dict[str, PitGrade] = {}
    heating_only: set[str] = set()
    units: dict[str, str] = {}

    def normal_for(g: int) -> pd.DataFrame | None:
        if g not in normal_cache:
            as_of = gas_year_start(g, tz, start_month)
            lo = gas_year_start(g - int(norms["window_years"]), tz, start_month)
            sub = obs.loc[(obs["event_time"] >= lo) & (obs["event_time"] < as_of)]
            resolved = resolve_as_of(sub, as_of, validate=False) if len(sub) else sub
            daily = _daily_dd(resolved, spec, "event_time", "publication_time")
            normal_cache[g] = seasonal_normal_table(
                daily, ["hdd", "cdd"], g, int(norms["window_years"]), int(norms["min_years"]),
                int(norms["smoothing_half_window_days"]), start_month)
        return normal_cache[g]

    def state(frame: pd.DataFrame, as_of: pd.Timestamp, lo: pd.Timestamp,
              hi: pd.Timestamp) -> pd.DataFrame:
        sub = frame.loc[(frame["valid_time"] >= lo) & (frame["valid_time"] <= hi)
                        & (frame["publication_time"] <= as_of)]
        if sub.empty:
            return _daily_dd(sub, spec, "valid_time", "issue_time")
        resolved = resolve_forecasts_as_of(sub, as_of, validate=False)
        return _daily_dd(resolved, spec, "valid_time", "issue_time")

    def register(col: str, grade: PitGrade, unit: str, heating: bool) -> None:
        grades[col] = grade
        units[col] = unit
        if heating:
            heating_only.add(col)

    max_off = max(window_max_end(windows[w]) for w in
                  set(p["revision"]["windows"]) | set(p.get("level_windows") or [])
                  | set(p.get("ensemble_windows") or []))

    for origin in idx:
        o_day = origin_day(origin, tz)
        out: dict[str, float] = {}
        lo, hi = o_day + pd.Timedelta(days=1), o_day + pd.Timedelta(days=max_off)
        now = state(det, origin, lo, hi).set_index("day")
        prev = state(det, origin - step, lo, hi).set_index("day")
        normal = normal_for(gas_year(origin.tz_convert(tz), start_month))

        def window_days(w: str) -> pd.DatetimeIndex:
            s, e = window_offsets(windows[w], o_day, tz)
            return pd.date_range(o_day + pd.Timedelta(days=s), o_day + pd.Timedelta(days=e),
                                 freq="D")

        def window_sum(frame: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> float:
            if frame.empty or not set(days) <= set(frame.index):
                return np.nan
            v = frame.loc[days, col]
            return float(v.sum()) if v.notna().all() else np.nan

        # Levels and anomalies
        for w in p.get("level_windows") or []:
            days = window_days(w)
            stale = (not now.empty and set(days) <= set(now.index)
                     and (now.loc[days, "issue_min"] < origin - max_age).any())
            for kind in ("hdd", "cdd"):
                col, acol = f"{FAMILY}__{kind}_fc_{w}", f"{FAMILY}__{kind}_fc_anom_{w}"
                register(col, g_det, "degC_day", kind == "hdd")
                register(acol, g_anom, "degC_day", kind == "hdd")
                lvl = np.nan if stale else window_sum(now, days, kind)
                if np.isnan(lvl):
                    flags.add(col, origin, "stale_run" if stale else "missing_valid_day")
                out[col] = lvl
                if normal is None:
                    out[acol] = np.nan
                    flags.add(acol, origin, "normal_unavailable_min_years")
                else:
                    out[acol] = lvl - float(normal.loc[noleap_doy(days), kind].sum())
        # Revisions (lead-matched, same valid days)
        for w in p["revision"]["windows"]:
            days = window_days(w)
            reason = ""
            if now.empty or prev.empty or not (set(days) <= set(now.index)
                                               and set(days) <= set(prev.index)):
                reason = "missing_valid_day_or_lead"
            elif (now.loc[days, "issue_min"] < origin - max_age).any():
                reason = "stale_run"
            elif (now.loc[days, "issue_min"] <= prev.loc[days, "issue_max"]).any():
                reason = "no_newer_run"
            for kind in ("hdd", "cdd"):
                col = f"{FAMILY}__{kind}_rev_{w}"
                register(col, g_det, "degC_day", kind == "hdd")
                if reason:
                    out[col] = np.nan
                    flags.add(col, origin, reason)
                    continue
                a, b = window_sum(now, days, kind), window_sum(prev, days, kind)
                out[col] = a - b
                if np.isnan(out[col]):
                    flags.add(col, origin, "missing_value")
        # Ensemble mean and spread
        for w in p.get("ensemble_windows") or []:
            days = window_days(w)
            mcol, scol = f"{FAMILY}__hdd_ens_mean_{w}", f"{FAMILY}__hdd_ens_spread_{w}"
            register(mcol, g_ens, "degC_day", True)
            register(scol, g_ens, "degC_day", True)
            e = state(ens, origin, days[0], days[-1])
            sums = pd.Series(dtype=float)
            if not e.empty:
                e = e.loc[e["day"].isin(days)]
                if len(e) and (e["issue_min"] >= origin - max_age).all():
                    piv = e.pivot_table(index="member", columns="day", values="hdd",
                                        aggfunc="first", dropna=False)
                    piv = piv.reindex(columns=days)
                    sums = piv.loc[piv.notna().all(axis=1)].sum(axis=1)
            if len(sums) < max(min_members, 2):
                out[mcol] = out[scol] = np.nan
                flags.add(mcol, origin, f"members<{min_members}")
                flags.add(scol, origin, f"members<{min_members}")
            else:
                out[mcol] = float(sums.mean())
                out[scol] = float(sums.std(ddof=1))
        # Observed (lagged, grade B) - clearly separated
        _observed_block(out, obs, spec, origin, o_day, tz, normal, obs_p, start_month, flags,
                        register, g_obs)
        rows.append(out)

    frame = pd.DataFrame(rows, index=idx)
    frame.attrs["pit_grade"] = {c: grades[c].value for c in frame.columns}
    frame.attrs["flags"] = flags.flags
    frame.attrs["weights_label"] = weights_label
    frame.attrs["heating_season_only"] = sorted(c for c in frame.columns if c in heating_only)
    frame.attrs["units"] = {c: units[c] for c in frame.columns}
    frame.attrs["region"] = region.region_id
    return frame


def _observed_block(out: dict[str, float], obs: pd.DataFrame, spec: _DDSpec,
                    origin: pd.Timestamp, o_day: pd.Timestamp, tz: str,
                    normal: pd.DataFrame | None, obs_p: Mapping[str, Any], start_month: int,
                    flags: FlagLog, register: Any, g_obs: PitGrade) -> None:
    recent = [int(n) for n in obs_p["recent_window_days"]]
    stale_days = int(obs_p["max_staleness_days"])
    gys = gas_year_start(gas_year(origin.tz_convert(tz), start_month), tz, start_month)
    gys_day = origin_day(gys, tz)
    cols = [f"{FAMILY}__obs_hdd_anom_{n}d" for n in recent] + [
        f"{FAMILY}__obs_hdd_anom_cum_gas_year"]
    for c in cols:
        register(c, g_obs, "degC_day", True)
    lo = min(gys_day, o_day - pd.Timedelta(days=max(recent) + stale_days + 2))
    sub = obs.loc[(obs["event_time"] >= lo) & (obs["publication_time"] <= origin)]
    if sub.empty or normal is None:
        for c in cols:
            out[c] = np.nan
            flags.add(c, origin, "no_observations" if sub.empty else "normal_unavailable_min_years")
        return
    resolved = resolve_as_of(sub, origin, validate=False)
    daily = _daily_dd(resolved, spec, "event_time", "publication_time").set_index("day")["hdd"]
    daily = daily.dropna()
    if daily.empty or (o_day - daily.index.max()).days > stale_days:
        for c in cols:
            out[c] = np.nan
            flags.add(c, origin, "observations_stale")
        return
    end = daily.index.max()
    anom = daily - normal["hdd"].reindex(noleap_doy(daily.index)).to_numpy()
    for n, c in zip(recent, cols[:-1], strict=True):
        days = pd.date_range(end - pd.Timedelta(days=n - 1), end, freq="D")
        out[c] = float(anom.loc[days].sum()) if set(days) <= set(anom.index) else np.nan
        if np.isnan(out[c]):
            flags.add(c, origin, "observation_gap")
    if end < gys_day:
        out[cols[-1]] = np.nan
        flags.add(cols[-1], origin, "no_observed_day_in_gas_year_yet")
    else:
        days = pd.date_range(gys_day, end, freq="D")
        ok = set(days) <= set(anom.index)
        out[cols[-1]] = float(anom.loc[days].sum()) if ok else np.nan
        if not ok:
            flags.add(cols[-1], origin, "observation_gap")


__all__ = [
    "FlagLog",
    "build_weather_features",
    "cooling_degree_days",
    "gas_year",
    "gas_year_start",
    "heating_degree_days",
    "merged_params",
    "noleap_doy",
    "origin_day",
    "origin_index",
    "seasonal_normal_table",
    "window_max_end",
    "window_offsets",
]
