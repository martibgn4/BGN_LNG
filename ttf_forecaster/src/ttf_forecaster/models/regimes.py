"""Point-in-time market regimes for TTF: rule-based physical regimes, Markov switching, CUSUM.

Everything in this module that can reach a model as a FEATURE is causal: the value stamped at time
``t`` is computable from data with ``publication_time <= t`` (``docs/point_in_time_policy.md``
§5.6). Concretely:

* **Rule-based regimes** {oversupplied, normal, tight, crisis} are read from a signed tightness
  score built from inputs available at ``t``: storage fill (% of WGV) z vs a prior-gas-year
  normal and, optionally, the freight-adjusted NEA-NWE LNG arb (the PHYSICAL inputs); front-month
  price and realised-vol percentiles vs a trailing window ending at ``t`` (MARKET STRESS, combined
  contribution capped); M1-M2 backwardation as a deviation from its same-month prior-gas-year
  normal. Tight and crisis require a physical input to score tight by itself. Score thresholds are fixed ex-ante in
  ``configs/regimes.yaml``; hysteresis and a minimum duration are applied forward in time only.
* **Markov switching** (statsmodels ``MarkovRegression``, switching variance) is refit on an
  expanding window on a schedule anchored at the series start and exposes **filtered**
  probabilities ``P(state_t | y_<=t)`` only. Smoothed probabilities exist solely in
  :func:`markov_smoothed_probabilities_diagnostic`, which no feature path calls.
* **CUSUM** change-point statistics use returns standardised by moments estimated strictly before
  ``t``; an alarm is stamped at the time it would have fired.

The single exception is :func:`evaluation_only_period_labels` (historical crisis window
2021-07..2023-06, research spec §8 criterion 2). It is hindsight by construction and must never be
used as a feature; :func:`build_regime_features` refuses any column with its prefix.

Daily grid: all regime quantities are computed on one grid of front-month settlement availability
times (one row per trading date, stamped with the settle's ``publication_time``) and then sampled
as-of at the requested origins. The labels therefore do not depend on which origins are requested.

Curves have no resolver in ``features/point_in_time.py``; the curve read here applies the same rule
by hand (``publication_time <= as_of``, grade C excluded) and takes the **first** vintage of each
(contract, trade date), which is known from its own publication onward.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    DataBundle,
    NonPositivePriceError,
    PitGrade,
    PointInTimeViolation,
    validate_frame,
    worst_grade,
)
from ttf_forecaster.features.point_in_time import resolve_as_of, to_utc_timestamp
from ttf_forecaster.settings import load_config

logger = logging.getLogger(__name__)

FAMILY = "regime"
EVALUATION_ONLY_PREFIX = "eval_"
DEFAULT_LABELS: tuple[str, ...] = ("oversupplied", "normal", "tight", "crisis")
INPUT_COLUMNS: tuple[str, ...] = (
    "storage_z",
    "price_percentile",
    "vol_percentile",
    "m1m2_dev",
    "jkm_ttf",
)
PHYSICAL_INPUTS: tuple[str, ...] = ("storage_z", "jkm_ttf")
"""Physical-balance indicators. Tight/crisis need one of them to score tight by itself."""
MARKET_STRESS_INPUTS: tuple[str, ...] = ("price_percentile", "vol_percentile")
"""Curve-only "market stress" indicators; their combined positive score is capped."""
_CLEAN_GRADES = frozenset({PitGrade.A.value, PitGrade.B.value})


class RegimeConfigError(ValueError):
    """``configs/regimes.yaml`` (or a params override) is missing a key or inconsistent."""


class OptionalDependencyMissing(ImportError):
    """An optional package (e.g. ``hmmlearn``) is required for the requested model."""


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_regime_config(
    overrides: Mapping[str, Any] | None = None, config_dir: str | Path | None = None
) -> dict[str, Any]:
    """Load ``configs/regimes.yaml`` and deep-merge ``overrides`` on top.

    Raises:
        RegimeConfigError: if a required section is missing or score thresholds are not ordered.
    """
    cfg = load_config("regimes", str(config_dir) if config_dir else None)
    cfg = _deep_merge(cfg, overrides or {})
    for section in ("inputs", "rules", "markov", "changepoint", "evaluation_only"):
        if section not in cfg:
            raise RegimeConfigError(f"regimes config lacks section {section!r}")
    th = _req(cfg, "rules", "score_thresholds")
    lo, ti, cr = (
        th["oversupplied_at_or_below"],
        th["tight_at_or_above"],
        th["crisis_at_or_above"],
    )
    if not lo < ti < cr:
        raise RegimeConfigError(
            f"score thresholds must satisfy oversupplied < tight < crisis, got {lo}, {ti}, {cr}"
        )
    if len(_labels(cfg)) != 4:
        raise RegimeConfigError("rules.labels must list exactly 4 labels, loose -> tight")
    return cfg


def _req(cfg: Mapping[str, Any], *keys: str) -> Any:
    node: Any = cfg
    for i, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            raise RegimeConfigError(f"missing regimes config key {'.'.join(keys[: i + 1])}")
        node = node[key]
    return node


def _labels(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(cfg.get("rules", {}).get("labels", DEFAULT_LABELS))


# ---------------------------------------------------------------------------------------------
# Inputs from the bundle (PIT)
# ---------------------------------------------------------------------------------------------


def _local_dates(ts: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    return idx.tz_convert(MARKET_TZ).tz_localize(None).normalize()


def front_month_panel(
    curves: pd.DataFrame,
    contracts: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    as_of: pd.Timestamp | str | None = None,
    allowed_grades: frozenset[str] = _CLEAN_GRADES,
) -> pd.DataFrame:
    """Daily front-month (M1) and next-month (M2) settlements with within-contract log returns.

    Args:
        curves: CURVE_SCHEMA rows (any hubs/fields; filtered to ``inputs.hub``, monthly, settle).
        contracts: CONTRACT_METADATA_SCHEMA rows; ``last_trade_date`` used when present, otherwise
            ``delivery_start - BDay(inputs.expiry_bdays_before_delivery)`` (ex-ante rule).
        cfg: full regimes config.
        as_of: only rows with ``publication_time <= as_of`` are read.
        allowed_grades: PIT grades admitted (grade C excluded by default).

    Returns:
        Frame indexed by ``available_time`` (UTC; the latest first-vintage publication time on the
        trade date) with ``trade_date`` (local), ``m1_contract``, ``m1_price``, ``m2_price``,
        ``m1_logret`` (ln of the SAME contract's settle vs the previous trade date, never across a
        roll), ``m1_delivery_month`` (1-12), ``same_season`` and ``pit_grade``.

    Raises:
        NonPositivePriceError: a log return is requested on a settle <= 0.
    """
    icfg = _req(cfg, "inputs")
    hub = icfg["hub"]
    c = curves.loc[
        (curves["hub"] == hub) & (curves["tenor_type"] == "M") & (curves["field"] == "settle")
    ]
    c = c.loc[c["pit_grade"].isin(allowed_grades)]
    if as_of is not None:
        c = c.loc[c["publication_time"] <= to_utc_timestamp(as_of)]
    cols = ["trade_date", "m1_contract", "m1_price", "m2_price", "m1_logret", "same_season",
            "pit_grade"]
    if c.empty:
        out = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="available_time"))
        return out
    # First vintage per (contract, trade date): known from its own publication onward.
    c = c.sort_values(["contract_id", "trade_date", "revision", "publication_time"])
    c = c.drop_duplicates(["contract_id", "trade_date"], keep="first").copy()
    c["trade_local"] = _local_dates(c["trade_date"])
    c["deliv_local"] = _local_dates(c["delivery_start"])

    meta = c.drop_duplicates("contract_id").set_index("contract_id")["deliv_local"].sort_values()
    rule_ltd = meta - pd.offsets.BDay(int(icfg["expiry_bdays_before_delivery"]))
    if contracts is not None and len(contracts) and "last_trade_date" in contracts.columns:
        vendor = contracts.drop_duplicates("contract_id").set_index("contract_id")["last_trade_date"]
        vendor = pd.Series(_local_dates(vendor), index=vendor.index)
        ltd = vendor.reindex(meta.index).fillna(rule_ltd)
    else:
        ltd = rule_ltd

    prices = c.pivot(index="trade_local", columns="contract_id", values="value")
    prices = prices.reindex(columns=meta.index)  # columns ordered by delivery start
    pub = c.groupby("trade_local")["publication_time"].max()
    grade = c.groupby("trade_local")["pit_grade"].agg(lambda g: worst_grade(list(g)).value)

    dates = prices.index
    roll = pd.offsets.BDay(int(icfg["roll_trading_days_before_expiry"]))
    ltd_arr = ltd.to_numpy(dtype="datetime64[ns]")
    deliv = meta.to_numpy(dtype="datetime64[ns]")
    winter = set(int(m) for m in icfg["winter_months"])
    pvals = prices.to_numpy(dtype=float)
    n = len(dates)
    m1_contract = np.full(n, None, dtype=object)
    m1_month = np.full(n, np.nan)
    m1 = np.full(n, np.nan)
    m2 = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    same = np.zeros(n, dtype=bool)
    for i, d in enumerate(dates):
        threshold = np.datetime64(d + roll, "ns")
        eligible = np.flatnonzero(ltd_arr > threshold)
        if eligible.size == 0:
            continue
        j = int(eligible[0])
        m1_contract[i] = meta.index[j]
        m1_month[i] = pd.Timestamp(deliv[j]).month
        m1[i] = pvals[i, j]
        if j + 1 < len(deliv):
            nxt = pd.Timestamp(deliv[j]) + pd.DateOffset(months=1)
            if pd.Timestamp(deliv[j + 1]) == nxt:
                m2[i] = pvals[i, j + 1]
                same[i] = (pd.Timestamp(deliv[j]).month in winter) == (nxt.month in winter)
        if i > 0:
            prev = pvals[i - 1, j]
            if np.isfinite(prev) and np.isfinite(m1[i]):
                if prev <= 0 or m1[i] <= 0:
                    raise NonPositivePriceError(
                        f"log return on non-positive settle for {meta.index[j]} at {d.date()}"
                    )
                ret[i] = np.log(m1[i]) - np.log(prev)
    out = pd.DataFrame(
        {
            "trade_date": dates,
            "m1_contract": m1_contract,
            "m1_delivery_month": m1_month,
            "m1_price": m1,
            "m2_price": m2,
            "m1_logret": ret,
            "same_season": same,
            "pit_grade": grade.reindex(dates).to_numpy(),
        },
        index=pd.DatetimeIndex(pub.reindex(dates).to_numpy(), name="available_time"),
    )
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out


def price_regime_inputs(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Market-stress percentiles and the seasonally-adjusted M1-M2 backwardation.

    ``price_percentile`` / ``vol_percentile`` are "market stress" measures, not physical ones.
    ``m1m2_dev`` is ``(M1-M2)/M2`` minus its mean, over PRIOR gas years (of the trade date), for
    the same M1 delivery calendar month (:func:`m1m2_seasonal_deviation`). All windows are
    trailing, so each row is computable at its index time.
    """
    icfg = _req(cfg, "inputs")
    pp = icfg["price_percentile"]
    rv = icfg["realised_vol"]
    price = panel["m1_price"].astype(float)
    price_pct = price.rolling(int(pp["window_trading_days"]), min_periods=int(pp["min_periods"]))
    vol = panel["m1_logret"].astype(float).rolling(
        int(rv["vol_window_trading_days"]), min_periods=int(rv["vol_min_periods"])
    ).std() * np.sqrt(float(rv["annualisation"]))
    vol_pct = vol.rolling(
        int(rv["percentile_window_trading_days"]), min_periods=int(rv["percentile_min_periods"])
    ).rank(pct=True)
    back = (panel["m1_price"] - panel["m2_price"]) / panel["m2_price"]
    dev, normal = m1m2_seasonal_deviation(
        back.astype(float), panel["m1_delivery_month"], pd.DatetimeIndex(panel["trade_date"]),
        int(icfg["m1m2"]["min_prior_gas_years"]),
    )
    return pd.DataFrame(
        {
            "price_percentile": price_pct.rank(pct=True),
            "realised_vol": vol,
            "vol_percentile": vol_pct,
            "m1m2_backwardation": back.astype(float),
            "m1m2_normal": normal,
            "m1m2_dev": dev,
        },
        index=panel.index,
    )


def m1m2_seasonal_deviation(
    back: pd.Series, m1_month: pd.Series, trade_dates: pd.DatetimeIndex, min_years: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deviation of M1-M2 backwardation from its prior-gas-years mean for the same M1 month.

    For row ``t`` with front delivery month ``m`` and trade-date gas year ``g``: normal = mean
    over gas years ``< g`` of the per-gas-year mean backwardation on rows whose M1 month is ``m``
    (such rows are all earlier than ``t``). This removes the structural summer-winter carry
    (e.g. Sep vs Oct) instead of discarding those months. NaN until ``min_years`` prior gas years
    exist for that month.

    Returns:
        ``(deviation, normal)`` arrays aligned with ``back``.
    """
    gy = _gas_year(pd.DatetimeIndex(trade_dates))
    frame = pd.DataFrame({"b": back.to_numpy(dtype=float),
                          "m": pd.Series(m1_month).to_numpy(dtype=float), "g": gy})
    per = frame.dropna().groupby(["m", "g"])["b"].mean()
    months = set(per.index.get_level_values("m"))
    dev = np.full(len(frame), np.nan)
    normal = np.full(len(frame), np.nan)
    cache: dict[tuple[float, int], float] = {}
    for i, (b, m, g) in enumerate(frame.itertuples(index=False, name=None)):
        if not (np.isfinite(b) and np.isfinite(m)):
            continue
        key = (m, int(g))
        if key not in cache:
            hist = per.xs(m, level="m") if m in months else per.iloc[:0]
            hist = hist.loc[hist.index < g]
            cache[key] = float(hist.mean()) if len(hist) >= min_years else float("nan")
        normal[i] = cache[key]
        dev[i] = b - cache[key]
    return dev, normal


def _gas_year(local_dates: pd.DatetimeIndex) -> np.ndarray:
    return np.where(local_dates.month >= 10, local_dates.year, local_dates.year - 1)


def storage_seasonal_z(
    observations: pd.DataFrame,
    times: pd.DatetimeIndex,
    cfg: Mapping[str, Any],
    *,
    allowed_grades: frozenset[str] = _CLEAN_GRADES,
) -> pd.DataFrame:
    """Storage fill (% of working gas volume) vs its seasonal normal, as known at each of ``times``.

    At each time ``tau`` inventory and WGV are resolved as of ``tau`` (``resolve_as_of``: latest
    visible revision). Fill = inventory / WGV x 100, with WGV aligned by gas day and carried
    forward only from earlier days (capacity is a slow-moving published series). The latest fill
    with ``event_time <= tau`` (stale beyond ``max_staleness_days`` -> NaN) is compared with the
    mean, over gas years strictly before ``min(gas_year(tau), gas_year(last_event))`` (so on 1-2
    Oct the last published September day never enters its own normal; policy §5.3), of fills
    within ``seasonal_window_days`` of the same calendar day. Scale = cross-year std of those
    per-year means, floored at ``min_scale_pct_pts``. No WGV -> NaN (no TWh fallback).

    Returns:
        Frame indexed by ``times`` with ``storage_fill_pct``, ``storage_normal_pct``,
        ``storage_dev_pct_pts``, ``storage_z`` and ``storage_n_years``.
    """
    scfg = _req(cfg, "inputs", "storage")
    sid, wid = scfg["series_id"], scfg["wgv_series_id"]
    cols = ["storage_fill_pct", "storage_normal_pct", "storage_dev_pct_pts", "storage_z",
            "storage_n_years"]
    out = pd.DataFrame(np.nan, index=times, columns=cols)
    obs = observations.loc[observations["series_id"].isin([sid, wid])]
    if obs.empty or len(times) == 0 or not (obs["series_id"] == wid).any():
        return out
    validate_frame(obs, OBSERVATION_SCHEMA)
    window = int(scfg["seasonal_window_days"])
    min_years = int(scfg["min_prior_gas_years"])
    floor = float(scfg["min_scale_pct_pts"])
    stale = pd.Timedelta(days=float(scfg["max_staleness_days"]))
    pub_sorted = np.sort(pd.DatetimeIndex(obs["publication_time"]).as_unit("ns").asi8)
    cache: dict[tuple[int, int], tuple[Any, ...]] = {}
    tau_gy = _gas_year(_local_dates(times))
    rows = []
    for k, tau in enumerate(times):
        n_visible = int(np.searchsorted(pub_sorted, tau.as_unit("ns").value, side="right"))
        key = (n_visible, int(tau_gy[k]))
        if key not in cache:
            res = resolve_as_of(obs, tau, allowed_grades=allowed_grades, validate=False)
            cache[key] = _storage_stats(res, tau, sid, wid, int(tau_gy[k]), window, min_years,
                                        floor)
        v, normal, n_years, scale, last_event = cache[key]
        if not np.isfinite(v) or (tau - pd.Timestamp(last_event, tz="UTC")) > stale:
            rows.append((np.nan, np.nan, np.nan, np.nan, n_years))
            continue
        dev = v - normal
        rows.append((v, normal, dev, dev / scale if np.isfinite(scale) else np.nan, n_years))
    out.loc[:, cols] = np.asarray(rows, dtype=float)
    return out


def _storage_stats(
    res: pd.DataFrame, tau: pd.Timestamp, sid: str, wid: str, gy_tau: int, window: int,
    min_years: int, floor: float,
) -> tuple[float, float, float, float, Any]:
    nan = float("nan")
    res = res.loc[res["event_time"] <= tau]
    inv = res.loc[res["series_id"] == sid].set_index("event_time")["value"].astype(float)
    wgv = res.loc[res["series_id"] == wid].set_index("event_time")["value"].astype(float)
    if inv.empty or wgv.empty:
        return nan, nan, 0.0, nan, np.datetime64("NaT")
    inv = inv.sort_index()
    wgv_aligned = wgv.sort_index().reindex(inv.index.union(wgv.index)).ffill().reindex(inv.index)
    fill = (inv / wgv_aligned.where(wgv_aligned > 0) * 100.0).dropna()
    if fill.empty:
        return nan, nan, 0.0, nan, np.datetime64("NaT")
    last_event = fill.index[-1]
    v = float(fill.iloc[-1])
    ev_local = _local_dates(fill.index)
    gys = _gas_year(ev_local)
    before = min(gy_tau, int(gys[-1]))  # the last published day's gas year never enters its normal
    doy_last = ev_local[-1].dayofyear
    dist = np.abs(ev_local.dayofyear.to_numpy() - doy_last)
    dist = np.minimum(dist, 365 - dist)
    mask = (gys < before) & (dist <= window)
    if not mask.any():
        return v, nan, 0.0, nan, last_event.to_datetime64()
    per_year = pd.Series(fill.to_numpy()[mask]).groupby(gys[mask]).mean()
    n_years = float(len(per_year))
    if n_years < min_years:
        return v, nan, n_years, nan, last_event.to_datetime64()
    std = float(per_year.std(ddof=1)) if n_years >= 2 else 0.0
    return v, float(per_year.mean()), n_years, max(std, floor), last_event.to_datetime64()


def latest_observation_series(
    observations: pd.DataFrame,
    series_id: str,
    times: pd.DatetimeIndex,
    max_staleness_days: float,
    *,
    allowed_grades: frozenset[str] = _CLEAN_GRADES,
) -> pd.Series:
    """Latest as-of-resolved value of ``series_id`` at each of ``times`` (NaN if stale/absent)."""
    obs = observations.loc[observations["series_id"] == series_id]
    out = pd.Series(np.nan, index=times, name=series_id)
    if obs.empty:
        return out
    validate_frame(obs, OBSERVATION_SCHEMA)
    stale = pd.Timedelta(days=float(max_staleness_days))
    for tau in times:
        res = resolve_as_of(obs, tau, allowed_grades=allowed_grades, validate=False)
        past = res.loc[res["event_time"] <= tau]
        if past.empty:
            continue
        last = past.sort_values("event_time").iloc[-1]
        if tau - last["event_time"] <= stale:
            out.loc[tau] = float(last["value"])
    return out


def compute_regime_inputs(
    bundle: DataBundle,
    cfg: Mapping[str, Any],
    *,
    as_of: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """All rule inputs plus ``m1_logret`` on the daily availability grid, and their PIT grades.

    Returns:
        ``(inputs, grades)``: ``inputs`` indexed by availability time (UTC) with
        :data:`INPUT_COLUMNS` plus ``m1_logret``, ``realised_vol`` and storage diagnostics;
        ``grades`` maps each input column to the worst PIT grade of the rows it was built from.
    """
    panel = front_month_panel(bundle.curves, bundle.contracts, cfg, as_of=as_of)
    if panel.empty:
        raise PointInTimeViolation(
            "no TTF monthly settlements visible" + (f" as of {as_of}" if as_of is not None else "")
        )
    inputs = price_regime_inputs(panel, cfg)
    inputs["m1_logret"] = panel["m1_logret"].astype(float)
    price_grade = worst_grade(list(panel["pit_grade"].dropna().unique())).value
    grades = {k: price_grade for k in ("price_percentile", "vol_percentile", "m1m2_dev",
                                       "m1m2_backwardation", "m1_logret", "realised_vol")}
    obs = bundle.observations
    if as_of is not None:
        obs = obs.loc[obs["publication_time"] <= to_utc_timestamp(as_of)]
    st = storage_seasonal_z(obs, inputs.index, cfg)
    inputs = inputs.join(st)
    sids = [cfg["inputs"]["storage"]["series_id"], cfg["inputs"]["storage"]["wgv_series_id"]]
    st_rows = obs.loc[obs["series_id"].isin(sids) & obs["pit_grade"].isin(_CLEAN_GRADES)]
    if len(st_rows):
        grades["storage_z"] = worst_grade(list(st_rows["pit_grade"].unique())).value
    jcfg = cfg["inputs"].get("jkm_ttf", {})
    inputs["jkm_ttf"] = np.nan
    if jcfg.get("enabled"):
        if not jcfg.get("series_id"):
            raise RegimeConfigError("inputs.jkm_ttf.enabled is true but series_id is null")
        inputs["jkm_ttf"] = latest_observation_series(
            obs, jcfg["series_id"], inputs.index, jcfg["max_staleness_days"]
        )
        j_rows = obs.loc[obs["series_id"] == jcfg["series_id"]]
        if len(j_rows):
            grades["jkm_ttf"] = worst_grade(list(j_rows["pit_grade"].unique())).value
    return inputs, grades


# ---------------------------------------------------------------------------------------------
# Rule-based regimes
# ---------------------------------------------------------------------------------------------


def indicator_scores(inputs: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Signed tightness score per indicator (NaN where the input is NaN or the column absent).

    Each indicator maps its input to {-1, 0, +1, +2} via the ex-ante thresholds in
    ``rules.indicators``. ``jkm_ttf`` scores only when ``inputs.jkm_ttf.enabled``.
    """
    ind = _req(cfg, "rules", "indicators")
    jkm_on = bool(cfg.get("inputs", {}).get("jkm_ttf", {}).get("enabled", False))
    out = pd.DataFrame(index=inputs.index)
    for name in INPUT_COLUMNS:
        if name == "jkm_ttf" and not jkm_on:
            continue
        rule = ind.get(name)
        if rule is None:
            continue
        x = inputs[name].astype(float) if name in inputs else pd.Series(np.nan, index=inputs.index)
        s = pd.Series(0.0, index=inputs.index)
        if "tight_at_or_above" in rule:
            s = s.mask(x >= rule["tight_at_or_above"], 1.0)
        if "tight_at_or_below" in rule:
            s = s.mask(x <= rule["tight_at_or_below"], 1.0)
        if "crisis_at_or_above" in rule:
            s = s.mask(x >= rule["crisis_at_or_above"], 2.0)
        if "crisis_at_or_below" in rule:
            s = s.mask(x <= rule["crisis_at_or_below"], 2.0)
        if "loose_at_or_above" in rule:
            s = s.mask(x >= rule["loose_at_or_above"], -1.0)
        if "loose_at_or_below" in rule:
            s = s.mask(x <= rule["loose_at_or_below"], -1.0)
        out[name] = s.where(x.notna())
    return out


def classify_score(score: float, cfg: Mapping[str, Any]) -> float:
    """Map a summed score to a regime code (0 oversupplied .. 3 crisis); NaN stays NaN."""
    if not np.isfinite(score):
        return float("nan")
    th = _req(cfg, "rules", "score_thresholds")
    if score >= th["crisis_at_or_above"]:
        return 3.0
    if score >= th["tight_at_or_above"]:
        return 2.0
    if score <= th["oversupplied_at_or_below"]:
        return 0.0
    return 1.0


def apply_hysteresis(
    score: pd.Series, cfg: Mapping[str, Any], max_code: pd.Series | None = None
) -> pd.Series:
    """Causal anti-flicker filter on a score series; returns regime codes.

    Rules (``rules.hysteresis``), evaluated forward in time only:

    1. *Score band* (Schmitt trigger around ``normal``): a state away from normal is ENTERED at
       its plain threshold but only EXITED (towards normal) once the score is ``band`` points past
       that threshold, i.e. tight is left when ``score + band`` no longer classifies as tight.
       A move from tight to crisis (or normal to tight) is an entry and uses the plain threshold.
    2. *Minimum duration*: a new candidate must persist ``min_duration`` consecutive observations
       before the published state switches. The switch is stamped on the day the condition is
       met, never back-dated.

    ``max_code`` (optional, same index) caps the candidate at each step, e.g. at ``normal`` when
    no physical indicator scores tight (``rules.require_physical_for``). When the cap removes the
    evidence for the current state, leaving becomes the candidate, still subject to min_duration.

    A NaN score publishes NaN and clears any pending candidate; the last confirmed state is kept.
    """
    hcfg = _req(cfg, "rules", "hysteresis")
    band = float(hcfg["score_band"])
    normal = 1.0
    min_dur = max(int(hcfg["min_duration"]), 1)
    out = np.full(len(score), np.nan)
    state: float | None = None
    pending: float | None = None
    count = 0
    caps = (np.full(len(score), 3.0) if max_code is None
            else max_code.reindex(score.index).to_numpy(dtype=float))
    for i, s in enumerate(score.to_numpy(dtype=float)):
        if not np.isfinite(s):
            pending, count = None, 0
            continue
        cap = caps[i] if np.isfinite(caps[i]) else 3.0
        raw = min(classify_score(s, cfg), cap)
        if state is None:
            state = raw
        else:
            if raw > state:  # tighter: entering if from normal/tight, exiting if from oversupplied
                cand = raw if state >= normal else max(state, classify_score(s - band, cfg))
            elif raw < state:  # looser: exiting if from tight/crisis, entering if from normal
                cand = raw if state <= normal else min(state, classify_score(s + band, cfg))
            else:
                cand = state
            cand = min(cand, cap)
            if cand == state:
                pending, count = None, 0
            else:
                if cand == pending:
                    count += 1
                else:
                    pending, count = cand, 1
                if count >= min_dur:
                    state, pending, count = cand, None, 0
        out[i] = state
    return pd.Series(out, index=score.index, name="code")


def _run_length(codes: pd.Series) -> pd.Series:
    vals = codes.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    run = 0
    prev = np.nan
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            run, prev = 0, np.nan
            continue
        run = run + 1 if v == prev else 1
        prev = v
        out[i] = run
    return pd.Series(out, index=codes.index)


def classify_regimes(inputs: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rule-based regimes from an input frame indexed by availability time.

    Returns:
        Frame with ``score`` (sum of non-NaN indicator scores, the price+vol "market stress"
        part capped at ``rules.market_stress_cap``), ``n_indicators``, ``n_physical``,
        ``physical_tight``, ``raw_code`` (no hysteresis; diagnostic), ``code`` (after
        hysteresis), ``label`` and ``days_in_regime``. ``score``/codes are NaN when fewer than
        ``rules.min_indicators`` inputs, or fewer than ``rules.min_physical_indicators`` physical
        inputs, are available. Labels in ``rules.require_physical_for`` are reachable only while
        a physical indicator (storage_z, jkm_ttf) scores tight by itself: the curve alone can
        never produce tight or crisis.
    """
    rules = _req(cfg, "rules")
    scores = indicator_scores(inputs, cfg)
    n = scores.notna().sum(axis=1)
    market = [c for c in MARKET_STRESS_INPUTS if c in scores]
    physical = [c for c in PHYSICAL_INPUTS if c in scores]
    other = [c for c in scores.columns if c not in market]
    market_sum = scores[market].sum(axis=1, min_count=1).clip(
        upper=float(rules["market_stress_cap"]))
    total = pd.concat([market_sum, scores[other]], axis=1).sum(axis=1, min_count=1)
    if physical:
        n_phys = scores[physical].notna().sum(axis=1)
        phys_tight = (scores[physical] >= 1).any(axis=1)
    else:
        n_phys = pd.Series(0, index=scores.index)
        phys_tight = pd.Series(False, index=scores.index)
    enough = (n >= int(rules["min_indicators"])) & (
        n_phys >= int(rules["min_physical_indicators"]))
    total = total.where(enough)
    labels = _labels(cfg)
    gated = [labels.index(lab) for lab in rules.get("require_physical_for", [])]
    max_code = pd.Series(3.0, index=scores.index)
    if gated:
        max_code = max_code.where(phys_tight, float(min(gated) - 1))
    raw = pd.Series([min(classify_score(v, cfg), c) for v, c in zip(total, max_code)],
                    index=scores.index, dtype=float)
    code = apply_hysteresis(total, cfg, max_code=max_code)
    label = code.map(lambda c: labels[int(c)] if np.isfinite(c) else None)
    return pd.DataFrame(
        {
            "score": total,
            "n_indicators": n.astype(float),
            "n_physical": n_phys.astype(float),
            "physical_tight": phys_tight.astype(float),
            "raw_code": raw.astype(float),
            "code": code,
            "label": label,
            "days_in_regime": _run_length(code),
        },
        index=inputs.index,
    )


# ---------------------------------------------------------------------------------------------
# Markov switching (filtered only) and optional HMM
# ---------------------------------------------------------------------------------------------


def _markov_model(y: np.ndarray, mcfg: Mapping[str, Any]) -> Any:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    return MarkovRegression(
        y,
        k_regimes=int(mcfg["k_regimes"]),
        trend=str(mcfg["trend"]),
        switching_variance=bool(mcfg["switching_variance"]),
    )


def _nobs_by_k(probs: Any, nobs: int, k: int) -> np.ndarray:
    """statsmodels returns (nobs, k) for ndarray endog; guard against a transposed layout."""
    arr = np.asarray(probs, dtype=float)
    if arr.shape == (nobs, k):
        return arr
    if arr.shape == (k, nobs):
        return arr.T
    raise ValueError(f"unexpected Markov probability shape {arr.shape}, wanted ({nobs}, {k})")


def _variance_order(model: Any, params: np.ndarray, k: int) -> np.ndarray:
    names = list(model.param_names)
    if f"sigma2[{k - 1}]" in names:
        sig = np.array([params[names.index(f"sigma2[{i}]")] for i in range(k)])
    else:  # non-switching variance: order by mean instead
        sig = np.array([params[names.index(f"const[{i}]")] for i in range(k)])
    return np.argsort(sig)


def markov_filtered_probabilities(returns: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Expanding-window Markov-switching FILTERED state probabilities ``P(s_t | y_<=t)``.

    Refit positions are ``min_train_obs - 1 + k * refit_every`` counted from the first non-NaN
    return (anchored at the start, so appending data never moves a past refit). Parameters fitted
    on ``y[0..p]`` are used for the filtered probabilities at positions ``p .. p+refit_every-1``;
    the Hamilton filter at position ``s`` uses only ``y[0..s]``. If a refit fails the previous
    parameters are kept (``ms_refit_ok`` = False).

    Returns:
        Frame on ``returns.index`` with ``ms_p_state<i>`` (states ordered by fitted variance,
        ascending), ``ms_p_high_vol`` (= the highest-variance state) and ``ms_refit_ok``.
    """
    mcfg = _req(cfg, "markov")
    k = int(mcfg["k_regimes"])
    cols = [f"ms_p_state{i}" for i in range(k)]
    out = pd.DataFrame(np.nan, index=returns.index, columns=[*cols, "ms_p_high_vol"])
    out["ms_refit_ok"] = np.nan
    y_all = returns.astype(float).dropna()
    y = y_all.to_numpy() * float(mcfg["return_scale"])
    n = len(y)
    m = int(mcfg["min_train_obs"])
    step = int(mcfg["refit_every"])
    if n < m:
        return out
    probs = np.full((n, k), np.nan)
    ok = np.full(n, np.nan)
    params: np.ndarray | None = None
    for p in range(m - 1, n, step):
        end = min(p + step, n)  # exclusive
        fit_ok = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Warm start from the previous fit (itself estimated on data <= an earlier
                # refit point), which is causal and cuts EM iterations.
                res = _markov_model(y[: p + 1], mcfg).fit(
                    start_params=params, disp=False, maxiter=int(mcfg["max_iter"])
                )
            new = np.asarray(res.params, dtype=float)
            if not np.all(np.isfinite(new)):
                raise ValueError("non-finite Markov parameters")
            params = new
        except (ValueError, np.linalg.LinAlgError) as exc:
            fit_ok = False
            logger.warning("Markov refit at position %d failed (%s); keeping previous params", p,
                           exc)
        if params is None:
            continue
        model = _markov_model(y[:end], mcfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filt = model.filter(params)
        fp = _nobs_by_k(filt.filtered_marginal_probabilities, end, k)
        order = _variance_order(model, params, k)
        probs[p:end] = fp[p:end][:, order]
        ok[p:end] = float(fit_ok)
    out.loc[y_all.index, cols] = probs
    out.loc[y_all.index, "ms_p_high_vol"] = probs[:, -1]
    out.loc[y_all.index, "ms_refit_ok"] = ok
    return out


def markov_smoothed_probabilities_diagnostic(
    returns: pd.Series, cfg: Mapping[str, Any]
) -> pd.DataFrame:
    """DIAGNOSTIC ONLY - contains hindsight. Never use as a feature or for evaluation slicing.

    Fits one Markov-switching model on the FULL sample and returns its smoothed probabilities
    ``P(s_t | y_1..y_T)`` next to the same fit's in-sample filtered probabilities, for charts and
    for checking how much the smoother "knows" that the filter could not.

    Returns:
        Frame on ``returns.index`` with ``diag_smoothed_p_high_vol`` and
        ``diag_insample_filtered_p_high_vol``.
    """
    mcfg = _req(cfg, "markov")
    k = int(mcfg["k_regimes"])
    y_all = returns.astype(float).dropna()
    y = y_all.to_numpy() * float(mcfg["return_scale"])
    model = _markov_model(y, mcfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False, maxiter=int(mcfg["max_iter"]))
    order = _variance_order(model, np.asarray(res.params), k)

    sm = _nobs_by_k(res.smoothed_marginal_probabilities, len(y), k)[:, order]
    fi = _nobs_by_k(res.filtered_marginal_probabilities, len(y), k)[:, order]
    out = pd.DataFrame(np.nan, index=returns.index,
                       columns=["diag_smoothed_p_high_vol", "diag_insample_filtered_p_high_vol"])
    out.loc[y_all.index, "diag_smoothed_p_high_vol"] = sm[:, -1]
    out.loc[y_all.index, "diag_insample_filtered_p_high_vol"] = fi[:, -1]
    return out


def forward_filter(
    log_emission: np.ndarray, startprob: np.ndarray, transmat: np.ndarray
) -> np.ndarray:
    """Normalised HMM forward recursion: filtered ``P(s_t | x_<=t)`` for each ``t``.

    Args:
        log_emission: ``(T, K)`` log densities ``log p(x_t | s_t = k)``.
        startprob: ``(K,)`` initial distribution.
        transmat: ``(K, K)`` row-stochastic, ``transmat[i, j] = P(s_t = j | s_{t-1} = i)``.
    """
    t_len, k = log_emission.shape
    if startprob.shape != (k,) or transmat.shape != (k, k):
        raise ValueError("startprob/transmat shapes do not match log_emission")
    out = np.empty((t_len, k))
    prior = startprob.astype(float)
    for t in range(t_len):
        le = log_emission[t]
        w = prior * np.exp(le - np.max(le))
        total = w.sum()
        out[t] = w / total if total > 0 else np.full(k, 1.0 / k)
        prior = out[t] @ transmat
    return out


def hmm_filtered_probabilities(returns: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Gaussian-HMM FILTERED probabilities on an expanding, start-anchored refit schedule.

    ``hmmlearn``'s ``predict_proba`` is forward-backward (smoothed), so it is NOT used; the
    fitted parameters are run through :func:`forward_filter` instead.

    Raises:
        OptionalDependencyMissing: ``hmmlearn`` is not installed.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalDependencyMissing(
            "hmm_filtered_probabilities needs the optional package 'hmmlearn', which is not "
            "installed (modelling_log D-005). Use markov_filtered_probabilities instead, or ask "
            "the user to approve installing hmmlearn."
        ) from exc
    hcfg = _req(cfg, "hmm")
    k = int(hcfg["n_states"])
    y_all = returns.astype(float).dropna()
    y = y_all.to_numpy() * float(cfg.get("markov", {}).get("return_scale", 100.0))
    n, m, step = len(y), int(hcfg["min_train_obs"]), int(hcfg["refit_every"])
    probs = np.full((n, k), np.nan)
    for p in range(m - 1, n, step):  # pragma: no cover - needs hmmlearn
        end = min(p + step, n)
        hmm = GaussianHMM(n_components=k, covariance_type=str(hcfg["covariance_type"]),
                          n_iter=int(hcfg["n_iter"]), random_state=int(hcfg["random_state"]))
        hmm.fit(y[: p + 1].reshape(-1, 1))
        means = hmm.means_.ravel()
        var = np.asarray(hmm.covars_).reshape(k, -1)[:, 0]
        x = y[:end, None]
        le = -0.5 * (np.log(2 * np.pi * var)[None, :] + (x - means[None, :]) ** 2 / var[None, :])
        fp = forward_filter(le, hmm.startprob_, hmm.transmat_)
        probs[p:end] = fp[p:end][:, np.argsort(var)]
    cols = [f"hmm_p_state{i}" for i in range(k)]
    out = pd.DataFrame(np.nan, index=returns.index, columns=cols)
    out.loc[y_all.index, cols] = probs
    out["hmm_p_high_vol"] = out[cols[-1]]
    return out


# ---------------------------------------------------------------------------------------------
# Causal change-point detection
# ---------------------------------------------------------------------------------------------


def cusum_alarms(returns: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Two-sided Page CUSUM on standardised returns plus a one-sided variance CUSUM.

    ``z_t = (r_t - mean_{t-1}) / std_{t-1}`` with moments from the trailing
    ``standardise_window`` returns strictly before ``t``. Statistics:
    ``S+_t = max(0, S+_{t-1} + z_t - k_mean)``, ``S-_t = max(0, S-_{t-1} - z_t - k_mean)``,
    ``V_t = max(0, V_{t-1} + z_t^2 - 1 - k_var)``. An alarm is stamped at ``t`` when a statistic
    exceeds its ``h``; that statistic then resets to 0. NaN ``z`` leaves the statistics unchanged
    and raises no alarm.

    Returns:
        Frame on ``returns.index`` with ``cusum_pos``, ``cusum_neg``, ``cusum_var``,
        ``alarm_mean_up``, ``alarm_mean_down``, ``alarm_var_up`` (0/1) and
        ``days_since_alarm`` (rows since any alarm, capped; NaN before the first valid ``z``).
    """
    ccfg = _req(cfg, "changepoint")
    r = returns.astype(float)
    roll = r.rolling(int(ccfg["standardise_window"]),
                     min_periods=int(ccfg["standardise_min_periods"]))
    mu = roll.mean().shift(1)
    sd = roll.std().shift(1)
    z = ((r - mu) / sd.where(sd > 0)).to_numpy()
    km, hm = float(ccfg["k_mean"]), float(ccfg["h_mean"])
    kv, hv = float(ccfg["k_var"]), float(ccfg["h_var"])
    cap = float(ccfg["days_since_alarm_cap"])
    n = len(z)
    pos = np.full(n, np.nan)
    neg = np.full(n, np.nan)
    var = np.full(n, np.nan)
    a_up = np.zeros(n)
    a_dn = np.zeros(n)
    a_v = np.zeros(n)
    since = np.full(n, np.nan)
    sp = sn = sv = 0.0
    started = False
    last_alarm: int | None = None
    for t in range(n):
        if np.isfinite(z[t]):
            started = True
            sp = max(0.0, sp + z[t] - km)
            sn = max(0.0, sn - z[t] - km)
            sv = max(0.0, sv + z[t] ** 2 - 1.0 - kv)
            if sp > hm:
                a_up[t], sp = 1.0, 0.0
            if sn > hm:
                a_dn[t], sn = 1.0, 0.0
            if sv > hv:
                a_v[t], sv = 1.0, 0.0
            if a_up[t] or a_dn[t] or a_v[t]:
                last_alarm = t
        if started:
            pos[t], neg[t], var[t] = sp, sn, sv
            since[t] = cap if last_alarm is None else min(float(t - last_alarm), cap)
    return pd.DataFrame(
        {
            "cusum_pos": pos,
            "cusum_neg": neg,
            "cusum_var": var,
            "alarm_mean_up": a_up,
            "alarm_mean_down": a_dn,
            "alarm_var_up": a_v,
            "days_since_alarm": since,
        },
        index=returns.index,
    )


# ---------------------------------------------------------------------------------------------
# Outputs: daily regime series, transitions (T9c), evaluation-only labels, feature builder
# ---------------------------------------------------------------------------------------------


def regime_series_from_inputs(
    inputs: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    include_markov: bool | None = None,
    include_changepoint: bool | None = None,
) -> pd.DataFrame:
    """Rule regimes (+ optional filtered Markov probabilities and CUSUM) on an input grid.

    ``inputs`` must be indexed by tz-aware UTC availability times; each row must be computable at
    its index time (the caller's responsibility when not built by :func:`compute_regime_inputs`).
    ``m1_logret`` is required for the Markov / CUSUM parts.
    """
    if not isinstance(inputs.index, pd.DatetimeIndex) or inputs.index.tz is None:
        raise PointInTimeViolation("regime inputs must be indexed by tz-aware UTC timestamps")
    if not inputs.index.is_monotonic_increasing:
        raise PointInTimeViolation("regime inputs index must be sorted ascending")
    out = classify_regimes(inputs, cfg)
    use_ms = cfg["markov"].get("enabled", True) if include_markov is None else include_markov
    use_cp = (cfg["changepoint"].get("enabled", True) if include_changepoint is None
              else include_changepoint)
    if (use_ms or use_cp) and "m1_logret" not in inputs:
        raise RegimeConfigError("m1_logret column required for Markov / change-point outputs")
    if use_ms:
        out = out.join(markov_filtered_probabilities(inputs["m1_logret"], cfg))
    if use_cp:
        out = out.join(cusum_alarms(inputs["m1_logret"], cfg).add_prefix("cp_"))
    return out


def regime_label_series(
    bundle: DataBundle,
    cfg: Mapping[str, Any] | None = None,
    *,
    as_of: pd.Timestamp | str | None = None,
    include_markov: bool | None = None,
    include_changepoint: bool | None = None,
) -> pd.DataFrame:
    """Daily regime series for targets (T9c) and conditional evaluation.

    Each row is "the regime evaluated with information at that date": computed only from data
    published by the row's index time (settle availability of that trade date). Rows before ``as_of``
    are identical whether or not later data exists in ``bundle``.

    Returns:
        Frame indexed by availability time (UTC) with ``trade_date`` (local) and the columns of
        :func:`regime_series_from_inputs` plus the inputs themselves.
    """
    cfg = load_regime_config() if cfg is None else cfg
    inputs, _ = compute_regime_inputs(bundle, cfg, as_of=as_of)
    reg = regime_series_from_inputs(inputs, cfg, include_markov=include_markov,
                                    include_changepoint=include_changepoint)
    panel_dates = _local_dates(inputs.index)
    reg.insert(0, "trade_date", panel_dates)
    return reg.join(inputs.drop(columns=[c for c in inputs.columns if c in reg.columns]))


def regime_transition_series(codes: pd.Series, horizon: int) -> pd.DataFrame:
    """Transition indicator for target T9c: ``1[regime(t+h) != regime(t)]``.

    Args:
        codes: regime codes on the daily grid (one row per trading day), each evaluated with
            information at its own date, e.g. ``regime_label_series(...)["code"]``.
        horizon: ``h`` in grid rows (trading days).

    Returns:
        Frame indexed like ``codes`` with ``regime_t``, ``regime_t_plus_h``, ``value`` (1.0 / 0.0,
        NaN when either end is unknown or beyond the data) and ``label_known_time`` (the index
        time of row ``t+h``; NaT if beyond the data). The market-data agent wraps this into
        TARGET_SCHEMA.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 trading day")
    c = codes.astype(float)
    fut = c.shift(-horizon)
    known = pd.Series(codes.index, index=codes.index).shift(-horizon)
    value = (fut != c).astype(float).where(c.notna() & fut.notna())
    return pd.DataFrame(
        {"regime_t": c, "regime_t_plus_h": fut, "value": value, "label_known_time": known},
        index=codes.index,
    )


def evaluation_only_period_labels(
    index: pd.DatetimeIndex, cfg: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """EVALUATION ONLY - NEVER A FEATURE. Hindsight period flags for slicing OOS metrics.

    Flags origins inside the historical crisis window (default 2021-07-01..2023-06-30 inclusive,
    Europe/Amsterdam local dates) used by research spec §8 criterion 2 ("R² > 0 including and
    excluding 2021-07..2023-06"). The window was chosen with knowledge of what happened, which is
    exactly what point_in_time_policy.md §5.6 forbids in a feature. Column names start with
    :data:`EVALUATION_ONLY_PREFIX`; :func:`build_regime_features` rejects such columns.
    """
    cfg = load_regime_config() if cfg is None else cfg
    w = _req(cfg, "evaluation_only", "crisis_window")
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise PointInTimeViolation("evaluation labels need a tz-aware DatetimeIndex")
    local = _local_dates(index)
    name = str(w["name"])
    if not name.startswith(EVALUATION_ONLY_PREFIX):
        raise RegimeConfigError(f"evaluation-only column {name!r} must start with "
                                f"{EVALUATION_ONLY_PREFIX!r}")
    inside = (local >= pd.Timestamp(w["start"])) & (local <= pd.Timestamp(w["end"]))
    return pd.DataFrame({name: inside}, index=index)


_FEATURE_MAP: dict[str, str] = {
    "score": "score",
    "n_indicators": "n_indicators",
    "code": "code",
    "days_in_regime": "days_in_regime",
    "ms_p_high_vol": "ms_p_high_vol",
    "cp_cusum_var": "cp_cusum_var",
    "cp_days_since_alarm": "cp_days_since_alarm",
}


def build_regime_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: dict[str, object]
) -> pd.DataFrame:
    """FeatureBuilder for the ``regime`` family (``data/base.py``).

    Inputs come **from the bundle by default** (:func:`compute_regime_inputs`: TTF monthly
    settles from ``bundle.curves``, storage from ``bundle.observations``), read only up to
    ``max(origins)``. Alternatively ``params["input_frame"]`` supplies already-built columns
    (:data:`INPUT_COLUMNS` and ``m1_logret``) indexed by their availability time, together with
    ``params["input_pit_grades"]`` (column -> grade); the caller then vouches for their PIT-ness.

    Regimes are computed on the daily grid and sampled as-of each origin (last grid row with
    time <= origin; older than ``features.max_label_staleness_days`` -> NaN). Only FILTERED Markov
    probabilities are used.

    Params (all optional): ``overrides`` (dict deep-merged into ``configs/regimes.yaml``),
    ``include_markov``, ``include_changepoint``, ``input_frame``, ``input_pit_grades``,
    ``max_label_staleness_days`` (default 7).

    Returns:
        Frame indexed by ``origins`` (name ``origin_time``) with columns ``regime__score``,
        ``regime__n_indicators``, ``regime__code``, ``regime__is_<label>`` (one-hot, NaN when
        unknown), ``regime__days_in_regime`` and, when enabled, ``regime__ms_p_high_vol``,
        ``regime__cp_cusum_var``, ``regime__cp_days_since_alarm``. ``attrs["pit_grade"]`` maps
        each column to the worst grade of its inputs.
    """
    if not isinstance(origins, pd.DatetimeIndex) or origins.tz is None:
        raise PointInTimeViolation("origins must be a tz-aware DatetimeIndex")
    origins = origins.tz_convert("UTC")
    overrides = params.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise RegimeConfigError("params['overrides'] must be a mapping")
    cfg = load_regime_config(overrides)
    include_ms = params.get("include_markov")
    include_cp = params.get("include_changepoint")
    use_ms = bool(cfg["markov"].get("enabled", True) if include_ms is None else include_ms)
    use_cp = bool(cfg["changepoint"].get("enabled", True) if include_cp is None else include_cp)
    horizon_end = origins.max() if len(origins) else None

    input_frame = params.get("input_frame")
    if input_frame is not None:
        if not isinstance(input_frame, pd.DataFrame):
            raise RegimeConfigError("params['input_frame'] must be a DataFrame")
        grades_in = params.get("input_pit_grades")
        if not isinstance(grades_in, Mapping):
            raise RegimeConfigError("params['input_pit_grades'] is required with input_frame")
        leaked = [c for c in input_frame.columns if str(c).startswith(EVALUATION_ONLY_PREFIX)]
        if leaked:
            raise PointInTimeViolation(
                f"evaluation-only columns {leaked} passed as regime inputs; they contain hindsight"
            )
        inputs = input_frame.copy()
        for target, alias in (cfg["inputs"].get("input_frame_aliases") or {}).items():
            if target not in inputs and alias in inputs:
                inputs[target] = inputs[alias]
                if alias in grades_in and target not in grades_in:
                    grades_in = {**grades_in, target: grades_in[alias]}
        if horizon_end is not None:
            inputs = inputs.loc[inputs.index <= horizon_end]
        for col in INPUT_COLUMNS:
            if col not in inputs:
                inputs[col] = np.nan
        grades = {str(k): str(v) for k, v in grades_in.items()}
    else:
        inputs, grades = compute_regime_inputs(bundle, cfg, as_of=horizon_end)

    reg = regime_series_from_inputs(inputs, cfg, include_markov=use_ms,
                                    include_changepoint=use_cp)
    bad = [c for c in reg.columns if str(c).startswith(EVALUATION_ONLY_PREFIX)]
    if bad:
        raise PointInTimeViolation(f"evaluation-only columns reached the feature path: {bad}")

    staleness = pd.Timedelta(days=float(params.get("max_label_staleness_days", 7)))  # type: ignore[arg-type]
    pos = reg.index.searchsorted(origins, side="right") - 1
    valid = pos >= 0
    grid_time = pd.DatetimeIndex(
        np.where(valid, reg.index.to_numpy()[np.clip(pos, 0, None)], np.datetime64("NaT")),
        tz="UTC",
    ) if len(reg) else pd.DatetimeIndex([pd.NaT] * len(origins), tz="UTC")
    fresh = valid & ((origins - grid_time) <= staleness)
    sampled = reg.iloc[np.clip(pos, 0, None)].reset_index(drop=True) if len(reg) else None

    labels = _labels(cfg)
    out = pd.DataFrame(index=pd.DatetimeIndex(origins, name="origin_time"))
    wanted = ["score", "n_indicators", "code", "days_in_regime"]
    if use_ms:
        wanted.append("ms_p_high_vol")
    if use_cp:
        wanted += ["cp_cusum_var", "cp_days_since_alarm"]
    for src in wanted:
        vals = (sampled[src].to_numpy(dtype=float) if sampled is not None
                else np.full(len(origins), np.nan))
        out[f"{FAMILY}__{_FEATURE_MAP[src]}"] = np.where(fresh, vals, np.nan)
    code = out[f"{FAMILY}__code"]
    for i, lab in enumerate(labels):
        out[f"{FAMILY}__is_{lab}"] = (code == i).astype(float).where(code.notna())

    rule_inputs = [c for c in INPUT_COLUMNS if c in grades and inputs[c].notna().any()]
    price_grade = grades.get("m1_logret", grades.get("price_percentile", PitGrade.A.value))
    rule_grade = worst_grade([grades[c] for c in rule_inputs] or [price_grade]).value
    pit: dict[str, str] = {}
    for col in out.columns:
        if col.endswith(("__ms_p_high_vol", "__cp_cusum_var", "__cp_days_since_alarm")):
            pit[col] = PitGrade(price_grade).value
        else:
            pit[col] = rule_grade
    out.attrs["pit_grade"] = pit
    out.attrs["family"] = FAMILY
    return out
