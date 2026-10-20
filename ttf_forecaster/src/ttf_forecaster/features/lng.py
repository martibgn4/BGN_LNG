"""LNG feature family (``lng__*``): arrivals, on-water, routing, trade mix, arb, diversions, slots.

Every value at origin ``t`` is computed from data with ``publication_time <= t`` through the
as-of resolvers in ``features/point_in_time.py``; unavailable values are NaN, never 0.
``frame.attrs["pit_grade"]`` maps each column to the worst grade of the rows it used (or, when
no row was ever available, to the grade its source family is declared with).

Inputs expected in the :class:`~ttf_forecaster.data.base.DataBundle`:

* observations — Bloomberg LNG aggregates (``configs/sources/bloomberg_lng.yaml``), Spark netbacks
  and terminal slots (``configs/sources/spark.yaml``), vessel-derived statistics and realised
  arrivals from ``data/spark.py`` (``destination_observations``, ``realised_arrival_observations``);
* forecasts — PIT arrival forecasts ``lng.arrivals.<geo>.*`` from ``build_arrival_forecasts``.

Grade C rows (e.g. destination features from a non-vintaged vessel history) are excluded unless
``params["allow_grade_c"]`` is true, which is only for a separately labelled contaminated run.

Default parameters live in :data:`DEFAULT_PARAMS`; the proposed home is an ``lng:`` block of
``configs/features.yaml`` (owned by the feature-engineering agent), passed in as ``params``.
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
from ttf_forecaster.data.spark import (
    UNDECLARED_SHARE_SERIES,
    arrival_series_id,
    diversion_series_id,
    realised_series_id,
)
from ttf_forecaster.features.point_in_time import (
    resolve_as_of,
    resolve_forecasts_as_of,
    to_utc_timestamp,
)

FAMILY = "lng"
SEP = "__"

ORIGIN_SERIES = {
    "usa": "lng.usa.afloat_20d_vessels_count",
    "qatar": "lng.qatar.afloat_20d_vessels_count",
    "australia": "lng.australia.afloat_20d_vessels_count",
    "nigeria": "lng.nigeria.afloat_20d_vessels_count",
    "russia": "lng.russia.afloat_20d_vessels_count",
    "other": "lng.other.afloat_20d_vessels_count",
}
ONWATER_SERIES = {
    "onwater20_volume_t": "lng.global.afloat_20d_volume_t",
    "onwater20_vessels": "lng.global.afloat_20d_vessels_count",
    "onwater30_volume_t": "lng.global.afloat_30d_volume_t",
    "onwater30_vessels": "lng.global.afloat_30d_vessels_count",
}
TRANSIT_SERIES = {
    "panama": "lng.panama.laden_transits_count",
    "suez": "lng.suez.laden_transits_count",
    "cape": "lng.cape.laden_transits_count",
    "nsr": "lng.nsr.laden_transits_count",
}
FLOW_NWE = "lng.amer_to_nwe.flow_t_per_d"
FLOW_ASIA = ("lng.amer_to_nasia.flow_t_per_d", "lng.amer_to_sasia.flow_t_per_d")
SLOT_PREFIX = "lng.slots."

DEFAULT_PARAMS: dict[str, Any] = {
    "allow_grade_c": False,               # True only for a labelled contaminated experiment
    "arrival_geos": ["eu", "nwe"],
    "arrival_horizons_days": [7, 14, 30, 45],
    "arrival_volume_unit": "m3_LNG",
    "arrival_change_days": 7,
    "arrival_max_staleness_days": 3,
    "obs_max_staleness_days": 7,          # Bloomberg aggregates: older latest value -> NaN
    "onwater_change_days": 7,
    "transit_window_days": 30,
    "flow_window_days": 30,
    "window_min_coverage": 0.8,           # share of calendar days present in a rolling window
    # monthIdx semantics unconfirmed (spark.yaml: requires_discovery) -> explicit, overridable
    "arb_series_id": "route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu",
    "arb_change_days": 7,
    "arb_max_staleness_days": 5,
    "diversion_region": "eu",
    "diversion_lookback_days": 14,
    "derived_max_staleness_days": 3,
    "error_geo": "nwe",
    "error_horizon_days": 14,
    "error_search_issues": 10,
    "slots_terminals": None,              # None -> every terminal observed at the origin
    "slots_bucket": "m1",
    "slots_change_days": 20,
    "slots_max_staleness_days": 5,
}

# Declared grade of each source family, used only when a column never saw a row.
_DEFAULT_GRADE = {"arrivals": "C", "diversions": "C", "destination": "C", "onwater": "B",
                  "transits": "B", "amer": "B", "arb": "A", "slots": "A"}


def _col(name: str) -> str:
    return f"{FAMILY}{SEP}{name}"


class _Acc:
    """Accumulates values and the grades of the rows behind them, per column."""

    def __init__(self, origins: pd.DatetimeIndex) -> None:
        self.origins = origins
        self.values: dict[str, np.ndarray] = {}
        self.grades: dict[str, set[str]] = {}

    def declare(self, name: str) -> None:
        if name not in self.values:
            self.values[name] = np.full(len(self.origins), np.nan)
            self.grades[name] = set()

    def put(self, name: str, i: int, value: float, grades: Sequence[str]) -> None:
        self.declare(name)
        if value is None or not np.isfinite(value):
            return
        self.values[name][i] = float(value)
        self.grades[name].update(str(g) for g in grades)

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame({_col(k): v for k, v in self.values.items()}, index=self.origins)
        out.index.name = "origin_time"
        grades = {}
        for k, gs in self.grades.items():
            fallback = next((g for pre, g in _DEFAULT_GRADE.items() if k.startswith(pre)), "C")
            grades[_col(k)] = worst_grade(sorted(gs)).value if gs else fallback
        out.attrs["pit_grade"] = grades
        return out


# ---- per-series helpers on an as-of-resolved view ---------------------------------------------


def _series(resolved: pd.DataFrame, series_id: str) -> tuple[pd.Series, list[str]]:
    sub = resolved.loc[resolved["series_id"] == series_id]
    s = pd.Series(sub["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(sub["event_time"]))
    return s.sort_index(), sub["pit_grade"].unique().tolist()


def _latest(s: pd.Series, as_of: pd.Timestamp, max_stale: pd.Timedelta) -> tuple[float, Any]:
    past = s.loc[s.index <= as_of].dropna()
    if past.empty or as_of - past.index[-1] > max_stale:
        return np.nan, None
    return float(past.iloc[-1]), past.index[-1]


def _change(s: pd.Series, as_of: pd.Timestamp, max_stale: pd.Timedelta, days: int) -> float:
    val, ev = _latest(s, as_of, max_stale)
    if ev is None:
        return np.nan
    prior = s.loc[s.index <= ev - pd.Timedelta(days=days)].dropna()
    if prior.empty or (ev - prior.index[-1]) > pd.Timedelta(days=days) + max_stale:
        return np.nan
    return val - float(prior.iloc[-1])


def _window_sum(s: pd.Series, as_of: pd.Timestamp, max_stale: pd.Timedelta, days: int,
                min_cov: float) -> float:
    """Sum over the ``days`` calendar days ending at the latest visible event (NaN if sparse)."""
    _, ev = _latest(s, as_of, max_stale)
    if ev is None:
        return np.nan
    win = s.loc[(s.index > ev - pd.Timedelta(days=days)) & (s.index <= ev)].dropna()
    if win.index.normalize().nunique() < min_cov * days:
        return np.nan
    return float(win.sum())


def _ratio(num: float, den: float) -> float:
    if not (np.isfinite(num) and np.isfinite(den)) or den <= 0:
        return np.nan
    return num / den


# ---- builder -------------------------------------------------------------------------------------


def build_lng_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                       params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Build ``lng__*`` features for each origin (FeatureBuilder contract, ``data/base.py``).

    Args:
        bundle: all vintages; this function applies the as-of rule itself.
        origins: tz-aware UTC origin timestamps (returned index, same order).
        params: overrides of :data:`DEFAULT_PARAMS`.

    Returns:
        Frame indexed by ``origins`` with ``lng__<name>`` float columns and
        ``attrs["pit_grade"]``.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    origins = pd.DatetimeIndex(origins)
    if origins.tz is None or str(origins.tz) != "UTC":
        raise ValueError("origins must be tz-aware UTC")
    allowed = None if p["allow_grade_c"] else frozenset({PitGrade.A.value, PitGrade.B.value})

    obs = bundle.observations
    lng_obs = obs.loc[obs["series_id"].str.startswith(("lng.", "route."))]
    validate_frame(lng_obs, OBSERVATION_SCHEMA)
    fc = bundle.forecasts
    lng_fc = fc.loc[fc["series_id"].str.startswith("lng.arrivals.")]
    validate_frame(lng_fc, FORECAST_SCHEMA)

    geos = [str(g).lower() for g in p["arrival_geos"]]
    horizons = [int(h) for h in p["arrival_horizons_days"]]
    unit = str(p["arrival_volume_unit"])
    stale_obs = pd.Timedelta(days=p["obs_max_staleness_days"])
    stale_arr = pd.Timedelta(days=p["arrival_max_staleness_days"])
    stale_der = pd.Timedelta(days=p["derived_max_staleness_days"])
    stale_arb = pd.Timedelta(days=p["arb_max_staleness_days"])
    stale_slot = pd.Timedelta(days=p["slots_max_staleness_days"])
    cov = float(p["window_min_coverage"])
    acc = _Acc(origins)

    # declare every column up front so the frame shape never depends on data availability
    names: list[str] = []
    for g in geos:
        for h in horizons:
            names += [f"arrivals_{g}_cargoes_{h}d", f"arrivals_{g}_volume_{h}d",
                      f"arrivals_{g}_cargoes_{h}d_chg{p['arrival_change_days']}d"]
    eg, eh = str(p["error_geo"]).lower(), int(p["error_horizon_days"])
    names.append(f"arrivals_{eg}_cargoes_{eh}d_err")
    for k in ONWATER_SERIES:
        names += [k, f"{k}_chg{p['onwater_change_days']}d"]
    names += [f"onwater20_share_{o}" for o in ORIGIN_SERIES]
    tw, fw = int(p["transit_window_days"]), int(p["flow_window_days"])
    names += [f"transits_{c}_sum{tw}d" for c in TRANSIT_SERIES] + [f"transits_cape_share{tw}d"]
    names += [f"amer_nwe_share{fw}d"]
    names += ["arb_nea_minus_nwe_usd_mmbtu", f"arb_nea_minus_nwe_chg{p['arb_change_days']}d"]
    dr, dl = str(p["diversion_region"]).lower(), int(p["diversion_lookback_days"])
    names += [f"diversions_{dr}_net_{dl}d", f"diversions_{dr}_away_{dl}d",
              "destination_undeclared_share"]
    sb = str(p["slots_bucket"])
    names += [f"slots_{sb}_available", f"slots_{sb}_available_chg{p['slots_change_days']}d"]
    for n in names:
        acc.declare(n)

    for i, origin in enumerate(origins):
        t = to_utc_timestamp(origin)
        r = resolve_as_of(lng_obs, t, allowed_grades=allowed, validate=False)
        f = resolve_forecasts_as_of(lng_fc, t, allowed_grades=allowed, validate=False)

        # 1. expected arrivals by geo and horizon (latest visible issue)
        for g in geos:
            for h in horizons:
                for measure, name in (("cargoes", f"arrivals_{g}_cargoes_{h}d"),
                                      ("volume", f"arrivals_{g}_volume_{h}d")):
                    sid = arrival_series_id(g, h, measure, unit if measure == "volume" else None)
                    runs = f.loc[f["series_id"] == sid].sort_values("issue_time")
                    if runs.empty or t - runs["issue_time"].iloc[-1] > stale_arr:
                        continue
                    last = runs.iloc[-1]
                    acc.put(name, i, last["value"], [last["pit_grade"]])
                    if measure == "cargoes":
                        cd = int(p["arrival_change_days"])
                        prior = runs.loc[runs["issue_time"]
                                         <= last["issue_time"] - pd.Timedelta(days=cd)]
                        if len(prior):
                            pr = prior.iloc[-1]
                            if last["issue_time"] - pr["issue_time"] <= pd.Timedelta(
                                    days=cd) + stale_arr:
                                acc.put(f"{name}_chg{cd}d", i, last["value"] - pr["value"],
                                        [last["pit_grade"], pr["pit_grade"]])

        # 2. arrival forecast error, only once the realisation window is fully published
        real, real_g = _series(r, realised_series_id(eg))
        runs = f.loc[(f["series_id"] == arrival_series_id(eg, eh))
                     & (f["valid_time"] <= t)].sort_values("issue_time", ascending=False)
        for _, run in runs.head(int(p["error_search_issues"])).iterrows():
            days = pd.date_range(run["valid_time"] - pd.Timedelta(days=eh), periods=eh,
                                 freq="D")
            window = real.reindex(days)
            if window.isna().any():
                continue
            acc.put(f"arrivals_{eg}_cargoes_{eh}d_err", i, run["value"] - window.sum(),
                    [run["pit_grade"], *real_g])
            break

        # 3. on-water levels and changes
        for k, sid in ONWATER_SERIES.items():
            s, gr = _series(r, sid)
            v, _ = _latest(s, t, stale_obs)
            acc.put(k, i, v, gr)
            cd = int(p["onwater_change_days"])
            acc.put(f"{k}_chg{cd}d", i, _change(s, t, stale_obs, cd), gr)

        # 4. origin mix of >=20d afloat vessels (same event date for all origins)
        latest_by_origin: dict[str, tuple[float, Any, list[str]]] = {}
        for o, sid in ORIGIN_SERIES.items():
            s, gr = _series(r, sid)
            v, ev = _latest(s, t, stale_obs)
            latest_by_origin[o] = (v, ev, gr)
        evs = {ev for _, ev, _ in latest_by_origin.values()}
        if None not in evs and len(evs) == 1:
            total = sum(v for v, _, _ in latest_by_origin.values())
            all_g = [g for _, _, gr in latest_by_origin.values() for g in gr]
            for o, (v, _, _) in latest_by_origin.items():
                acc.put(f"onwater20_share_{o}", i, _ratio(v, total), all_g)

        # 5. chokepoint rolling sums and Cape share
        sums: dict[str, float] = {}
        tgr: list[str] = []
        for c, sid in TRANSIT_SERIES.items():
            s, gr = _series(r, sid)
            sums[c] = _window_sum(s, t, stale_obs, tw, cov)
            tgr += gr
            acc.put(f"transits_{c}_sum{tw}d", i, sums[c], gr)
        acc.put(f"transits_cape_share{tw}d", i,
                _ratio(sums["cape"], sums["cape"] + sums["suez"] + sums["panama"]), tgr)

        # 6. Americas -> NWE share of Americas -> NWE + Asia (rolling sums; legs are mostly zero)
        s_nwe, g_nwe = _series(r, FLOW_NWE)
        nwe = _window_sum(s_nwe, t, stale_obs, fw, cov)
        asia, g_asia = 0.0, []
        for sid in FLOW_ASIA:
            s, gr = _series(r, sid)
            asia += _window_sum(s, t, stale_obs, fw, cov)
            g_asia += gr
        acc.put(f"amer_nwe_share{fw}d", i, _ratio(nwe, nwe + asia), g_nwe + g_asia)

        # 7. freight-adjusted JKM-TTF arb (Spark netbacks, NEA minus NWE netback)
        s, gr = _series(r, str(p["arb_series_id"]))
        v, _ = _latest(s, t, stale_arb)
        acc.put("arb_nea_minus_nwe_usd_mmbtu", i, v, gr)
        cd = int(p["arb_change_days"])
        acc.put(f"arb_nea_minus_nwe_chg{cd}d", i, _change(s, t, stale_arb, cd), gr)

        # 8. diversions and destination uncertainty (vessel-snapshot derived)
        s_in, g_in = _series(r, diversion_series_id(dr, "into", dl))
        s_aw, g_aw = _series(r, diversion_series_id(dr, "away", dl))
        v_in, _ = _latest(s_in, t, stale_der)
        v_aw, _ = _latest(s_aw, t, stale_der)
        acc.put(f"diversions_{dr}_net_{dl}d", i, v_in - v_aw, g_in + g_aw)
        acc.put(f"diversions_{dr}_away_{dl}d", i, v_aw, g_aw)
        s, gr = _series(r, UNDECLARED_SHARE_SERIES)
        acc.put("destination_undeclared_share", i, _latest(s, t, stale_der)[0], gr)

        # 9. terminal-slot congestion: unsold slots summed over terminals (falling = tightening)
        suffix = f".available_{sb}_count"
        slot_ids = sorted(x for x in r["series_id"].unique()
                          if x.startswith(SLOT_PREFIX) and x.endswith(suffix))
        if p["slots_terminals"] is not None:
            wanted = [f"{SLOT_PREFIX}{str(term).lower()}{suffix}" for term in p["slots_terminals"]]
            slot_ids = wanted if all(w in set(slot_ids) for w in wanted) else []
        if slot_ids:
            lvl, chg, sgr = 0.0, 0.0, []
            scd = int(p["slots_change_days"])
            for sid in slot_ids:
                s, gr = _series(r, sid)
                lvl += _latest(s, t, stale_slot)[0]
                chg += _change(s, t, stale_slot, scd)
                sgr += gr
            acc.put(f"slots_{sb}_available", i, lvl, sgr)
            acc.put(f"slots_{sb}_available_chg{scd}d", i, chg, sgr)

    return acc.frame()
