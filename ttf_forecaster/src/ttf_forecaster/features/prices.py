"""Price, curve and cross-commodity feature families.

All three are computed in **trading-date space** (one row per local trading date of the hub), each
row stamped with ``available_time`` = the latest publication time of the rows it uses, and then
joined onto the origins with ``panel.align_to_origins`` (cumulative-max availability, bounded
staleness). Reads go through ``point_in_time.resolve_curves_as_of`` / ``resolve_as_of`` at the last
origin. That is PIT-exact for settlements and conservative if a settlement is ever revised: a
revised value is used only from the revision's publication time, never back-filled.

Cross-commodity prices are **paired by session date** (JKM/HH/Brent/FX of local date *d* with TTF
of date *d*) and become usable at the **max** of their publication times. Because the adapters
stamp JKM/HH/Brent/FX with the next-European-session rule (PIT policy §4), a same-date pair is
not usable at the TTF origin of date *d*, only at *d+1*. Nothing is ever aligned by calendar date
alone.

Continuous series: the front contract follows ``target_definitions.md`` §1 (nearest month with
last trade date more than ``roll_offset_bdays`` business days ahead). Daily returns are within one
contract (the one held from the previous session), spliced at the roll; a level difference across
a roll is never used as a return. The return series can be delegated to ``contracts/rolls.py``
via ``params["return_series_provider"]`` once it exists (see ``front_return_series``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.units import MMBTU_PER_MWH, UnitError, convert_energy
from ttf_forecaster.data.base import (
    MARKET_TZ,
    DataBundle,
    DataError,
    NonPositivePriceError,
    PitGrade,
    worst_grade,
)
from ttf_forecaster.features.calendar import (
    front_months,
    holiday_array,
    month_range,
    monthly_last_trade_dates,
)
from ttf_forecaster.features.panel import (
    DEFAULT_ALLOWED_GRADES,
    align_to_origins,
    as_origin_index,
    final_vintage,
)
from ttf_forecaster.features.point_in_time import resolve_as_of, resolve_curves_as_of
from ttf_forecaster.features.registry import load_dotted

log = logging.getLogger(__name__)

PRICE, CURVE, XC = "price", "curve", "cross_commodity"
SERIES_COLUMNS = ("date", "event_time", "available_time", "front_month", "front_settle",
                  "log_return")

# unit -> (currency, energy unit, multiplier to whole currency units)
_PRICE_UNITS: dict[str, tuple[str, str, float]] = {
    "EUR/MWh": ("EUR", "MWh", 1.0),
    "USD/MMBtu": ("USD", "MMBtu", 1.0),
    "GBp/therm": ("GBP", "therm", 0.01),
    "GBP/therm": ("GBP", "therm", 1.0),
}


# ---------------------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------------------


def price_to_eur_mwh(value: Any, unit: str, ccy_per_eur: Any = None) -> Any:
    """Convert a gas price to EUR/MWh.

    Args:
        value: scalar, array or Series.
        unit: one of ``EUR/MWh, USD/MMBtu, GBp/therm, GBP/therm``.
        ccy_per_eur: FX as units of the price currency per 1 EUR (EURUSD, EURGBP convention),
            of the **same session** as ``value``. Not needed for EUR prices.
    """
    try:
        ccy, energy, mult = _PRICE_UNITS[unit]
    except KeyError as exc:
        raise UnitError(f"unsupported gas price unit {unit!r}; known {sorted(_PRICE_UNITS)}") \
            from exc
    per_mwh = value * mult / convert_energy(1.0, energy, "MWh")
    if ccy == "EUR":
        return per_mwh
    if ccy_per_eur is None:
        raise UnitError(f"converting {unit} needs a {ccy}-per-EUR FX rate")
    return per_mwh / ccy_per_eur


def eur_mwh_to_usd_mmbtu_vec(price_eur_mwh: Any, usd_per_eur: Any) -> Any:
    """Vectorised ``units.eur_mwh_to_usd_mmbtu``."""
    return price_eur_mwh * usd_per_eur / MMBTU_PER_MWH


# ---------------------------------------------------------------------------------------------
# Curve preparation
# ---------------------------------------------------------------------------------------------


def _local_naive(ts: pd.Series, tzs: pd.Series) -> pd.Series:
    """Per-row local naive timestamps using each row's ``source_timezone``."""
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    for tz in tzs.unique():
        m = tzs == tz
        out.loc[m] = ts.loc[m].dt.tz_convert(tz).dt.tz_localize(None).dt.as_unit("ns")
    return out


def prepare_curve(
    curves: pd.DataFrame,
    hub: str,
    *,
    field: str = "settle",
    tenor_type: str = "M",
    allowed_grades: frozenset[str] | None = DEFAULT_ALLOWED_GRADES,
) -> pd.DataFrame:
    """Final-vintage curve rows of one hub/field/tenor with local ``date`` and delivery ``month``.

    ``date`` = local trading date (row ``source_timezone``); ``month`` = local month start of
    ``delivery_start``. Returns an empty frame (same columns) when nothing matches.
    """
    df = curves.loc[
        (curves["hub"] == hub) & (curves["field"] == field) & (curves["tenor_type"] == tenor_type)
    ]
    cols = ["date", "month", "value", "unit", "available_time", "pit_grade", "contract_id"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = final_vintage(df, ["hub", "contract_id", "trade_date", "field"],
                       allowed_grades=allowed_grades)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["date"] = _local_naive(df["trade_date"], df["source_timezone"]).dt.normalize()
    df["month"] = _local_naive(df["delivery_start"], df["source_timezone"]).dt.to_period(
        "M").dt.to_timestamp()
    return df[cols].reset_index(drop=True)


def _wide(prep: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(date x delivery month) values and per-date availability (max publication)."""
    if prep[["date", "month"]].duplicated().any():
        raise DataError("two curve rows for the same (trade date, delivery month)")
    wide = prep.pivot(index="date", columns="month", values="value").sort_index()
    wide = wide.astype(float)
    avail = prep.groupby("date")["available_time"].max().sort_index()
    return wide, avail


def _grade(prep: pd.DataFrame) -> str:
    g = prep["pit_grade"].dropna().unique().tolist() if len(prep) else []
    return worst_grade(g).value if g else PitGrade.A.value


def _single_unit(prep: pd.DataFrame, hub: str) -> str:
    units = prep["unit"].dropna().unique().tolist()
    if len(units) != 1:
        raise UnitError(f"{hub}: expected one price unit in curve rows, found {units}")
    return str(units[0])


def _event_time(dates: pd.DatetimeIndex, tz: str = MARKET_TZ) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates).tz_localize(tz).tz_convert("UTC")


def _check_positive(arr: np.ndarray, what: str) -> None:
    bad = np.isfinite(arr) & (arr <= 0)
    if bad.any():
        raise NonPositivePriceError(f"{int(bad.sum())} non-positive {what} prices; log undefined")


# ---------------------------------------------------------------------------------------------
# Front / spliced return series
# ---------------------------------------------------------------------------------------------


def _schedule(dates: pd.DatetimeIndex, cols: pd.DatetimeIndex, contracts: pd.DataFrame | None,
              p: Mapping[str, Any]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.Series]:
    hol = holiday_array(p.get("holidays"))
    start = min(dates.min(), cols.min()) if len(cols) else dates.min()
    end = max(dates.max(), cols.max()) if len(cols) else dates.max()
    months = month_range(start, end, pad_months=int(p.get("curve_pad_months", 30)))
    ltd = monthly_last_trade_dates(
        months, hub=str(p.get("hub", "TTF")), contracts=contracts,
        bdays_before=int(p["expiry_bdays_before_delivery"]), holidays=hol,
    )
    fm = front_months(dates, ltd, roll_offset_bdays=int(p["roll_offset_bdays"]), holidays=hol)
    return months, fm, ltd


def front_return_series(
    curves: pd.DataFrame, contracts: pd.DataFrame | None, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Spliced within-contract daily log returns and the unadjusted front level, per trade date.

    If ``params["return_series_provider"]`` is a dotted path, that callable
    ``(curves, contracts, params) -> DataFrame`` is used instead and must return
    ``SERIES_COLUMNS``. Otherwise the reference implementation here is used:

    * ``front_month(t)``: front by the roll rule; ``front_settle(t)`` = its settlement (EUR/MWh).
    * ``log_return(t) = ln F_c(t) - ln F_c(t_prev)`` with ``c = front_month(t_prev)`` (the
      contract held into session t); NaN if either settlement is missing.

    Returns:
        Frame with ``SERIES_COLUMNS``; ``attrs["pit_grade"]`` is the worst grade of the rows used.
    """
    provider = params.get("return_series_provider")
    if provider:
        out = load_dotted(str(provider))(curves, contracts, params)
        missing = [c for c in SERIES_COLUMNS if c not in out.columns]
        if missing:
            raise DataError(f"return_series_provider {provider} output lacks {missing}")
        return out
    hub = str(params.get("hub", "TTF"))
    prep = prepare_curve(curves, hub, allowed_grades=_allowed(params))
    if prep.empty:
        raise DataError(f"no {hub} monthly settlements in bundle.curves")
    wide, avail = _wide(prep)
    dates = pd.DatetimeIndex(wide.index)
    months, fm, _ = _schedule(dates, pd.DatetimeIndex(wide.columns), contracts, params)
    w = wide.reindex(columns=months).to_numpy(dtype=float)
    _check_positive(w, hub)
    rows = np.arange(len(dates))
    col = months.get_indexer(fm)
    level = w[rows, col]
    ret = np.full(len(dates), np.nan)
    if len(dates) > 1:
        held = col[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[1:] = np.log(w[rows[1:], held]) - np.log(w[rows[:-1], held])
    out = pd.DataFrame({
        "date": dates,
        "event_time": _event_time(dates),
        "available_time": avail.reindex(dates).to_numpy(),
        "front_month": fm,
        "front_settle": level,
        "log_return": ret,
    })
    out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    out.attrs["pit_grade"] = _grade(prep)
    return out


def visible_curves(curves: pd.DataFrame, origins: pd.DatetimeIndex,
                   params: Mapping[str, Any]) -> pd.DataFrame:
    """The sanctioned curve read: ``resolve_curves_as_of`` at the **last** origin.

    Every row published after the last origin is excluded here; rows published between origins
    are handled by ``available_time`` + ``align_to_origins`` downstream. Raises ``DataError`` on
    conflicting duplicate vintages (modelling_log D-015).
    """
    if len(origins) == 0:
        return curves.iloc[0:0]
    return resolve_curves_as_of(curves, origins.max(), allowed_grades=_allowed(params))


def visible_observations(observations: pd.DataFrame, series_ids: list[str],
                         origins: pd.DatetimeIndex, params: Mapping[str, Any]) -> pd.DataFrame:
    """``resolve_as_of`` at the last origin, restricted to ``series_ids``."""
    sub = observations.loc[observations["series_id"].isin(series_ids)]
    if len(origins) == 0 or sub.empty:
        return sub.iloc[0:0]
    return resolve_as_of(sub, origins.max(), allowed_grades=_allowed(params))


def _allowed(p: Mapping[str, Any]) -> frozenset[str] | None:
    g = p.get("allowed_grades", sorted(DEFAULT_ALLOWED_GRADES))
    return None if g is None else frozenset(str(x) for x in g)  # type: ignore[union-attr]


def _staleness(p: Mapping[str, Any]) -> pd.Timedelta | None:
    d = p.get("max_staleness_days")
    return None if d is None else pd.Timedelta(days=float(d))


# ---------------------------------------------------------------------------------------------
# price family
# ---------------------------------------------------------------------------------------------


def price_features_from_series(series: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Trading-date-space price features from a spliced series (``SERIES_COLUMNS``).

    Columns (unprefixed): ``ret_{k}d`` (sum of last k spliced log returns, all k required),
    ``rvol_{w}d`` (annualised stdev of daily returns), ``mom_{w}d`` (sum over w, >= frac*w
    present), ``z_{w}d`` (z-score of the back-adjusted log level vs its w-day mean/std),
    ``front_settle_eur_mwh`` (unadjusted level, a level not a return).
    """
    s = series.sort_values("date").reset_index(drop=True)
    r = s["log_return"].astype(float)
    frac = float(params.get("min_periods_frac", 0.8))
    ann = float(params.get("annualisation_days", 252))
    out = pd.DataFrame({"event_time": s["event_time"], "available_time": s["available_time"]})
    for k in params.get("return_windows", [1, 5, 20]):
        out[f"ret_{k}d"] = r.rolling(int(k), min_periods=int(k)).sum()
    for w in params.get("vol_windows", [20, 60]):
        mp = max(2, int(np.ceil(frac * int(w))))
        out[f"rvol_{w}d"] = r.rolling(int(w), min_periods=mp).std(ddof=1) * np.sqrt(ann)
    for w in params.get("momentum_windows", [60, 120]):
        mp = int(np.ceil(frac * int(w)))
        out[f"mom_{w}d"] = r.rolling(int(w), min_periods=mp).sum()
    log_level = r.fillna(0.0).cumsum()  # back-adjusted (ratio) log level; origin is arbitrary
    for w in params.get("zscore_windows", [20, 60]):
        mp = max(2, int(np.ceil(frac * int(w))))
        roll = log_level.rolling(int(w), min_periods=mp)
        sd = roll.std(ddof=1)
        out[f"z_{w}d"] = (log_level - roll.mean()) / sd.where(sd > 0)
    out["front_settle_eur_mwh"] = s["front_settle"].astype(float)
    return out


def _front_field_features(curves: pd.DataFrame, series: pd.DataFrame,
                          params: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Volume / open-interest features of the front contract (NaN when the field is absent)."""
    hub = str(params.get("hub", "TTF"))
    vw = int(params.get("volume_window", 20))
    ok = int(params.get("oi_change_days", 5))
    names = {"volume": f"volume_rel_{vw}d", "open_interest": f"oi_logchg_{ok}d"}
    frames: dict[str, pd.DataFrame] = {}
    grades: dict[str, str] = {}
    dates = pd.DatetimeIndex(series["date"])
    for field, name in names.items():
        prep = prepare_curve(curves, hub, field=field, allowed_grades=_allowed(params))
        grades[name] = _grade(prep)
        if prep.empty:
            frames[name] = pd.DataFrame(columns=["event_time", "available_time", name])
            continue
        wide, avail = _wide(prep)
        wide = wide.reindex(index=dates)
        fm = pd.DatetimeIndex(series["front_month"])
        months = pd.DatetimeIndex(sorted(set(wide.columns) | set(fm)))
        w = wide.reindex(columns=months).to_numpy(dtype=float)
        col = months.get_indexer(fm)
        rows = np.arange(len(dates))
        with np.errstate(divide="ignore", invalid="ignore"):
            lw = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), np.nan)
        if field == "volume":
            v = pd.Series(w[rows, col])
            mean = v.rolling(vw, min_periods=int(np.ceil(0.8 * vw))).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.log(v.where(v > 0) / mean.where(mean > 0)).to_numpy()
        else:
            val = np.full(len(dates), np.nan)
            if len(dates) > ok:
                val[ok:] = lw[rows[ok:], col[ok:]] - lw[rows[:-ok], col[ok:]]
        frames[name] = pd.DataFrame({
            "event_time": series["event_time"].to_numpy(),
            "available_time": pd.to_datetime(avail.reindex(dates).to_numpy(), utc=True),
            name: val,
        }).dropna(subset=["available_time"])
    return frames, grades


def build_price_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: Mapping[str, Any]
) -> pd.DataFrame:
    """``price`` family. See ``price_features_from_series`` and ``_front_field_features``.

    Params: ``hub, roll_offset_bdays, expiry_bdays_before_delivery, holidays, return_windows,
    vol_windows, momentum_windows, zscore_windows, min_periods_frac, annualisation_days,
    volume_window, oi_change_days, max_staleness_days, allowed_grades,
    return_series_provider``.
    """
    idx = as_origin_index(origins)
    curves = visible_curves(bundle.curves, idx, params)
    series = front_return_series(curves, bundle.contracts, params)
    feats = price_features_from_series(series, params)
    cols = [c for c in feats.columns if c not in ("event_time", "available_time")]
    stale = _staleness(params)
    out = align_to_origins(feats, idx, cols, max_staleness=stale)
    grade = str(series.attrs.get("pit_grade", PitGrade.A.value))
    grades = {c: grade for c in cols}
    frames, fgrades = _front_field_features(curves, series, params)
    for name, fr in frames.items():
        out[name] = align_to_origins(fr, idx, [name], max_staleness=stale)[name]
        grades[name] = worst_grade([grade, fgrades[name]]).value
    return _finish(out, PRICE, grades)


def _finish(out: pd.DataFrame, family: str, grades: Mapping[str, str]) -> pd.DataFrame:
    out = out.add_prefix(f"{family}__")
    out.index.name = "origin_time"
    out.attrs["pit_grade"] = {f"{family}__{k}": v for k, v in grades.items()}
    return out


# ---------------------------------------------------------------------------------------------
# curve family
# ---------------------------------------------------------------------------------------------


def month_hours(months: pd.DatetimeIndex, tz: str = MARKET_TZ) -> np.ndarray:
    """Clock hours in each delivery month in ``tz`` (DST-aware), used as strip weights."""
    m = pd.DatetimeIndex(months)
    start = m.tz_localize(tz)
    end = (m + pd.offsets.MonthBegin(1)).tz_localize(tz)
    return ((end - start) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def _next_month_on_or_after(dates: pd.DatetimeIndex, month: int) -> pd.DatetimeIndex:
    """First ``month``-1st strictly after each date (a season that has not started)."""
    y = dates.year.to_numpy()
    this = pd.to_datetime({"year": y, "month": month, "day": 1})
    yr = np.where(dates < pd.DatetimeIndex(this), y, y + 1)
    return pd.DatetimeIndex(pd.to_datetime({"year": yr, "month": month, "day": 1}))


def prior_gas_year_deviation(
    values: pd.Series,
    front: pd.DatetimeIndex,
    *,
    gas_year_start_month: int = 10,
    min_years: int = 1,
) -> pd.Series:
    """``values`` minus its mean over all dates of **strictly earlier gas years** that had the same
    front-month calendar month (PIT policy §5.3: seasonal normals use only prior gas years).

    Args:
        values: indexed by naive local trading date.
        front: front delivery month per date (same order).
        min_years: NaN until at least this many prior gas years contribute.
    """
    dates = pd.DatetimeIndex(values.index)
    gy = np.where(dates.month >= gas_year_start_month, dates.year, dates.year - 1)
    df = pd.DataFrame({"v": values.to_numpy(dtype=float), "gy": gy,
                       "mo": pd.DatetimeIndex(front).month})
    agg = df.dropna(subset=["v"]).groupby(["mo", "gy"])["v"].agg(["sum", "count"]).sort_index()
    cum = agg.groupby(level="mo").cumsum() - agg  # sums over strictly earlier gas years
    nyears = agg.groupby(level="mo").cumcount()
    normal = (cum["sum"] / cum["count"].where(cum["count"] > 0)).where(nyears >= min_years)
    key = pd.MultiIndex.from_arrays([df["mo"], df["gy"]])
    mean = normal.reindex(key).to_numpy()
    return pd.Series(df["v"].to_numpy() - mean, index=values.index)


def curve_features_from_wide(
    wide: pd.DataFrame,
    front: pd.DatetimeIndex,
    months: pd.DatetimeIndex,
    *,
    season_wide: pd.DataFrame | None = None,
    tz: str = MARKET_TZ,
    gas_year_start_month: int = 10,
    seasonal_min_years: int = 1,
) -> pd.DataFrame:
    """Curve-shape features (EUR/MWh unless noted) from a (date x delivery-month) settle matrix.

    ``M_k`` = settlement of the month ``k-1`` months after the front month on that date (NaN if not
    settled). Seasonal strips are hours-weighted means of their six months (NaN if any month is
    missing), else the listed season contract from ``season_wide`` (date x season start).

    Columns: m1_m2, m1_m2_dev ((M1-M2)/M2 minus its mean over prior gas years for the same front
    calendar month, dimensionless), slope_1_6 (M6-M1), slope_1_12 (M12-M1), logslope_1_12, butterfly_1_2_3
    (M1-2M2+M3), curvature_1_6_12 (M1-2M6+M12), prompt_vs_deferred (mean M1-3 - mean M4-12),
    sum_minus_win (next Summer - following Winter, T7c). The mirror pair win_minus_sum was dropped
    (D-032 F5): it has corr -0.99 with sum_minus_win and disjoint NaN patterns.
    """
    dates = pd.DatetimeIndex(wide.index)
    pad = 40
    w = wide.reindex(columns=months).to_numpy(dtype=float)
    w = np.hstack([w, np.full((len(dates), pad), np.nan)])
    rows = np.arange(len(dates))[:, None]
    c0 = months.get_indexer(front)
    if (c0 < 0).any():
        raise DataError("front month outside the month grid")
    m = w[rows, c0[:, None] + np.arange(12)[None, :]]  # M1..M12
    M = {k: m[:, k - 1] for k in range(1, 13)}
    out = pd.DataFrame(index=dates)
    out["m1_m2"] = M[1] - M[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = pd.Series((M[1] - M[2]) / np.where(M[2] > 0, M[2], np.nan), index=dates)
    out["m1_m2_dev"] = prior_gas_year_deviation(
        rel, front, gas_year_start_month=gas_year_start_month, min_years=seasonal_min_years)
    out["slope_1_6"] = M[6] - M[1]
    out["slope_1_12"] = M[12] - M[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["logslope_1_12"] = np.log(M[12]) - np.log(M[1])
    out["butterfly_1_2_3"] = M[1] - 2 * M[2] + M[3]
    out["curvature_1_6_12"] = M[1] - 2 * M[6] + M[12]
    out["prompt_vs_deferred"] = m[:, 0:3].mean(axis=1) - m[:, 3:12].mean(axis=1)

    hours = np.concatenate([month_hours(months, tz), np.full(pad, np.nan)])

    def strip(start: pd.DatetimeIndex) -> np.ndarray:
        p = months.get_indexer(start)
        val = np.full(len(dates), np.nan)
        ok = p >= 0
        cols = p[ok][:, None] + np.arange(6)[None, :]
        px = w[np.arange(len(dates))[ok][:, None], cols]
        h = hours[cols]
        val[ok] = (px * h).sum(axis=1) / h.sum(axis=1)  # NaN propagates if any month missing
        if season_wide is not None and len(season_wide.columns):
            sw = season_wide.reindex(index=dates)
            listed = np.array([
                sw.at[d, s] if s in sw.columns else np.nan for d, s in zip(dates, start)
            ], dtype=float)
            val = np.where(np.isnan(val), listed, val)
        return val

    next_sum = _next_month_on_or_after(dates, 4)
    foll_win = pd.DatetimeIndex(pd.to_datetime({"year": next_sum.year, "month": 10, "day": 1}))
    out["sum_minus_win"] = strip(next_sum) - strip(foll_win)
    return out


def build_curve_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: Mapping[str, Any]
) -> pd.DataFrame:
    """``curve`` family (see ``curve_features_from_wide``). Same roll params as ``price``."""
    idx = as_origin_index(origins)
    hub = str(params.get("hub", "TTF"))
    curves = visible_curves(bundle.curves, idx, params)
    prep = prepare_curve(curves, hub, allowed_grades=_allowed(params))
    if prep.empty:
        raise DataError(f"no {hub} monthly settlements in bundle.curves")
    wide, avail = _wide(prep)
    dates = pd.DatetimeIndex(wide.index)
    months, fm, _ = _schedule(dates, pd.DatetimeIndex(wide.columns), bundle.contracts, params)
    s_prep = prepare_curve(curves, hub, tenor_type="S", allowed_grades=_allowed(params))
    season_wide, s_avail = (_wide(s_prep) if len(s_prep) else (None, None))
    feats = curve_features_from_wide(
        wide, fm, months, season_wide=season_wide,
        gas_year_start_month=int(params.get("gas_year_start_month", 10)),
        seasonal_min_years=int(params.get("seasonal_min_years", 1)),
    )
    cols = list(feats.columns)
    av = avail.reindex(dates)
    grade = _grade(prep)
    if s_avail is not None:
        av = pd.concat([av, s_avail.reindex(dates)], axis=1).max(axis=1)
        grade = worst_grade([grade, _grade(s_prep)]).value
    feats["event_time"] = _event_time(dates)
    feats["available_time"] = pd.to_datetime(av.to_numpy(), utc=True)
    out = align_to_origins(feats.reset_index(drop=True), idx, cols,
                           max_staleness=_staleness(params))
    return _finish(out, CURVE, {c: grade for c in cols})


# ---------------------------------------------------------------------------------------------
# cross_commodity family
# ---------------------------------------------------------------------------------------------


def observation_by_date(
    observations: pd.DataFrame, series_id: str, *,
    allowed_grades: frozenset[str] | None = DEFAULT_ALLOWED_GRADES,
) -> tuple[pd.DataFrame, str]:
    """Final vintage of one observation series keyed by its local session date.

    Returns ``(frame[date, value, available_time], grade)``; empty frame if the series is absent.
    """
    df = observations.loc[observations["series_id"] == series_id]
    if df.empty:
        return pd.DataFrame(columns=["value", "available_time"]), PitGrade.A.value
    df = final_vintage(df, ["series_id", "event_time"], allowed_grades=allowed_grades)
    if df.empty:
        return pd.DataFrame(columns=["value", "available_time"]), PitGrade.A.value
    df["date"] = _local_naive(df["event_time"], df["source_timezone"]).dt.normalize()
    df = df.sort_values(["date", "event_time"]).drop_duplicates("date", keep="last")
    g = worst_grade(df["pit_grade"].unique().tolist()).value
    return df.set_index("date")[["value", "available_time"]], g


def _hub_at(curves: pd.DataFrame, hub: str, keys: pd.MultiIndex,
            allowed: frozenset[str] | None) -> tuple[pd.Series, pd.Series, str | None, str]:
    """Settle of ``hub`` at (date, delivery month) keys, its per-date availability, unit, grade."""
    prep = prepare_curve(curves, hub, allowed_grades=allowed)
    if prep.empty:
        nan = pd.Series(np.nan, index=keys)
        return nan, pd.Series(pd.NaT, index=keys, dtype="datetime64[ns, UTC]"), None, "A"
    unit = _single_unit(prep, hub)
    s = prep.set_index(["date", "month"])["value"]
    if s.index.duplicated().any():
        raise DataError(f"{hub}: duplicate (trade date, delivery month) rows")
    av = prep.groupby("date")["available_time"].max()
    val = s.reindex(keys).astype(float)
    avail = pd.Series(av.reindex(keys.get_level_values(0)).to_numpy(), index=keys)
    return val, pd.to_datetime(avail, utc=True), unit, _grade(prep)


def build_cross_commodity_features(
    bundle: DataBundle, origins: pd.DatetimeIndex, params: Mapping[str, Any]
) -> pd.DataFrame:
    """``cross_commodity`` family: session-paired spreads vs the TTF front month.

    Every cross price is taken for the **same delivery month** as the TTF front on the same local
    session date, and the pair is usable at the max publication time of its components.

    Columns:
        jkm_minus_ttf_usd_mmbtu: JKM - TTF (TTF converted with same-session EURUSD).
        ttf_minus_hh_cost_usd_mmbtu: TTF - (hh_slope * HH + hh_constant): an HH-indexed US LNG
            delivered-cost spread; slope/constant are config assumptions [CONFIRM].
        ttf_pct_brent: 100 * TTF(USD/MMBtu) / Brent(USD/bbl), oil-indexation "slope" in %.
        <hub>_minus_ttf_eur_mwh: inter-hub basis for each configured hub (NBP/THE/PEG...).

    Missing inputs give NaN columns (logged), never zeros.
    """
    idx = as_origin_index(origins)
    p = dict(params)
    allowed = _allowed(p)
    stale = _staleness(p)
    curves = visible_curves(bundle.curves, idx, p)
    fx_ids = [str(p["eurusd_series_id"])] + [
        str((c or {}).get("fx_series_id")) for c in (p.get("basis_hubs") or {}).values()
        if (c or {}).get("fx_series_id")]
    obs = visible_observations(bundle.observations, fx_ids, idx, p)
    ttf = front_return_series(curves, bundle.contracts, p)
    ttf_grade = str(ttf.attrs.get("pit_grade", "A"))
    base = ttf.set_index("date")
    dates = pd.DatetimeIndex(base.index)
    keys = pd.MultiIndex.from_arrays([dates, pd.DatetimeIndex(base["front_month"])],
                                     names=["date", "month"])
    ttf_eur = base["front_settle"].to_numpy(dtype=float)
    ttf_av = pd.to_datetime(base["available_time"], utc=True).to_numpy()

    fx_usd, g_usd = observation_by_date(obs, str(p["eurusd_series_id"]),
                                        allowed_grades=allowed)
    usd = fx_usd["value"].reindex(dates).to_numpy(dtype=float)
    usd_av = pd.to_datetime(fx_usd["available_time"].reindex(dates), utc=True).to_numpy()
    ttf_usd = eur_mwh_to_usd_mmbtu_vec(ttf_eur, usd)

    results: dict[str, tuple[np.ndarray, list[np.ndarray], list[str]]] = {}

    def need(unit: str | None, want: str, hub: str) -> None:
        if unit is not None and unit != want:
            raise UnitError(f"{hub} curve unit {unit!r}; cross_commodity expects {want}")

    jkm, jkm_av, ju, jg = _hub_at(curves, str(p.get("jkm_hub", "JKM")), keys, allowed)
    need(ju, "USD/MMBtu", "JKM")
    results["jkm_minus_ttf_usd_mmbtu"] = (jkm.to_numpy() - ttf_usd, [jkm_av.to_numpy(), ttf_av,
                                          usd_av], [jg, ttf_grade, g_usd])
    hh, hh_av, hu, hg = _hub_at(curves, str(p.get("hh_hub", "HH")), keys, allowed)
    need(hu, "USD/MMBtu", "HH")
    cost = float(p["hh_slope"]) * hh.to_numpy() + float(p["hh_constant_usd_mmbtu"])
    results["ttf_minus_hh_cost_usd_mmbtu"] = (ttf_usd - cost, [hh_av.to_numpy(), ttf_av, usd_av],
                                              [hg, ttf_grade, g_usd])
    br, br_av, bu, bg = _hub_at(curves, str(p.get("brent_hub", "Brent")), keys, allowed)
    need(bu, "USD/bbl", "Brent")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 100.0 * ttf_usd / np.where(br.to_numpy() > 0, br.to_numpy(), np.nan)
    results["ttf_pct_brent"] = (ratio, [br_av.to_numpy(), ttf_av, usd_av], [bg, ttf_grade, g_usd])

    for hub, cfg in (p.get("basis_hubs") or {}).items():
        val, av, unit, g = _hub_at(curves, str(hub), keys, allowed)
        avs, grades = [av.to_numpy(), ttf_av], [g, ttf_grade]
        fx = None
        if unit is not None and _PRICE_UNITS.get(unit, ("EUR",))[0] != "EUR":
            fx_id = (cfg or {}).get("fx_series_id")
            if not fx_id:
                raise UnitError(f"basis hub {hub} quoted in {unit} needs basis_hubs.{hub}."
                                "fx_series_id in configs/features.yaml")
            fxf, fg = observation_by_date(obs, str(fx_id), allowed_grades=allowed)
            fx = fxf["value"].reindex(dates).to_numpy(dtype=float)
            avs.append(pd.to_datetime(fxf["available_time"].reindex(dates), utc=True).to_numpy())
            grades.append(fg)
        eur = price_to_eur_mwh(val.to_numpy(), unit, fx) if unit else val.to_numpy()
        results[f"{str(hub).lower()}_minus_ttf_eur_mwh"] = (eur - ttf_eur, avs, grades)

    out = pd.DataFrame(index=idx)
    grades_out: dict[str, str] = {}
    et = _event_time(dates)
    for name, (val, avs, grades) in results.items():
        av = pd.DataFrame({i: pd.to_datetime(a, utc=True) for i, a in enumerate(avs)}).max(
            axis=1, skipna=False)
        fr = pd.DataFrame({"event_time": et, "available_time": av.to_numpy(), name: val})
        fr = fr.dropna(subset=[name, "available_time"])
        if fr.empty:
            log.info("cross_commodity: %s has no inputs in the bundle -> NaN", name)
        out[name] = align_to_origins(fr, idx, [name], max_staleness=stale)[name]
        grades_out[name] = worst_grade(grades).value
    return _finish(out, XC, grades_out)
