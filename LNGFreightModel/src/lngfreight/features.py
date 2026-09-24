"""Build the design matrix and the targets.

Three things happen here and the ORDER of them is the whole point:

  1. every series is shifted by its own publication lag, THEN
  2. windows are rolled over the shifted series, THEN
  3. targets are formed from unshifted freight prints.

Rolling first and shifting afterwards is the classic leak: a 21-day mean
computed on unshifted data and then shifted by one still contains the
observation from twenty days ago that the model could not have seen when the
window was formed. Doing it in this order makes that impossible.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from lngfreight.config import get

log = logging.getLogger(__name__)

TARGET_PREFIX = "y_h"
FRESH_PREFIX = "fresh_h"
SPOT = "atlantic_spot"
PACIFIC = "pacific_spot"

# Guards a divide-by-zero from turning into a silent inf in a ratio feature.
# Transit counts are genuinely zero on quiet days.
_EPS = 1e-9


# --------------------------------------------------------------------------
# staleness and lag
# --------------------------------------------------------------------------

def freshness(wide: pd.DataFrame, column: str) -> pd.Series:
    """True where the column has a genuine print on that business day."""
    return wide[column].notna()


def fill_bounded(series: pd.Series, limit: int) -> pd.Series:
    """Forward fill, but only across a bounded run of missing days."""
    return series.ffill(limit=limit)


def apply_lags(wide: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """Shift each column back by its publication lag plus the safety margin.

    The shift is in BUSINESS days on a business-day grid, applied to a
    calendar-day lag. That is deliberately conservative: a two-calendar-day
    lag observed on a Monday really points at the previous Thursday, and
    shifting two business days lands on Thursday rather than Saturday.

    The freight spot assessment is exempt when configured as same-day
    visible. It is the base of the target return and the number the
    random-walk benchmark is handed, so withholding it from the model alone
    would make the comparison unfair rather than careful.
    """
    if not get("features", "leakage", "apply_publication_lag"):
        raise ValueError(
            "apply_publication_lag is false. This pipeline has no unlagged "
            "mode; turning the leakage guard off is not a supported setting."
        )
    extra = int(get("features", "leakage", "extra_safety_lag_bd"))
    same_day_freight = bool(get("features", "leakage", "freight_spot_visible_same_day"))

    out = wide.copy()
    applied: dict[str, int] = {}
    for column in out.columns:
        if column in (SPOT, PACIFIC) and same_day_freight:
            shift = 0
        else:
            shift = int(lags.get(column, 0)) + extra
        if shift:
            out[column] = out[column].shift(shift)
        applied[column] = shift
    log.info("publication shifts applied (bd): %s",
             {k: v for k, v in sorted(applied.items()) if v})
    return out


# --------------------------------------------------------------------------
# family builders. Each returns a dict of column name -> Series.
# --------------------------------------------------------------------------

def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0.0, np.nan)


def family_arb(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    cfg = get("features", "families", "arb")
    zwin = int(cfg["zscore_window_bd"])
    out: dict[str, pd.Series] = {}
    for name, legs in cfg["levels"].items():
        left, right = legs["left"], legs["right"]
        if left not in lagged.columns or right not in lagged.columns:
            raise KeyError(f"arb feature {name} needs {left} and {right}")
        level = lagged[left].ffill() - lagged[right].ffill()
        out[f"arb__{name}"] = level
        out[f"arb__{name}__z{zwin}"] = _zscore(level, zwin)
        for w in cfg["change_windows_bd"]:
            out[f"arb__{name}__chg{int(w)}"] = level.diff(int(w))
    return out


def family_freight_own(lagged: pd.DataFrame, limit: int) -> dict[str, pd.Series]:
    cfg = get("features", "families", "freight_own")
    out: dict[str, pd.Series] = {}
    atl = fill_bounded(lagged[SPOT], limit)
    pac = fill_bounded(lagged[PACIFIC], limit)
    log_atl, log_pac = np.log(atl), np.log(pac)

    for w in cfg["returns_windows_bd"]:
        out[f"frt__ret{int(w)}"] = log_atl.diff(int(w))
        out[f"frt__pac_ret{int(w)}"] = log_pac.diff(int(w))
    for w in cfg["realised_vol_windows_bd"]:
        out[f"frt__vol{int(w)}"] = log_atl.diff().rolling(
            int(w), min_periods=int(w) // 2).std()
    for w in cfg["zscore_window_bd"]:
        out[f"frt__z{int(w)}"] = _zscore(log_atl, int(w))
    if cfg.get("basin_spread"):
        out["frt__basin_spread"] = atl - pac
        ratio = np.log((atl + _EPS) / (pac + _EPS))
        dz = int(cfg["derived_zscore_window_bd"])
        out["frt__basin_ratio"] = ratio
        out[f"frt__basin_ratio__z{dz}"] = _zscore(ratio, dz)
    return out


def family_on_water(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    cfg = get("features", "families", "on_water")
    smooth = int(cfg["smooth_window_bd"])
    zwin = int(cfg["zscore_window_bd"])
    out: dict[str, pd.Series] = {}
    smoothed: dict[str, pd.Series] = {}
    for name in cfg["series"]:
        if name not in lagged.columns:
            raise KeyError(f"on_water feature {name} is not in the panel")
        s = lagged[name].ffill().rolling(smooth, min_periods=1).mean()
        smoothed[name] = s
        out[f"ow__{name}"] = s
        out[f"ow__{name}__z{zwin}"] = _zscore(s, zwin)
        for w in cfg["change_windows_bd"]:
            out[f"ow__{name}__chg{int(w)}"] = s.pct_change(int(w))
    dz = int(cfg["derived_zscore_window_bd"])
    if cfg.get("ratio_30d_over_20d"):
        ratio = smoothed["afloat_30d_volume"] / (smoothed["afloat_20d_volume"] + _EPS)
        out["ow__float_storage_share"] = ratio
        out[f"ow__float_storage_share__z{dz}"] = _zscore(ratio, dz)
    if cfg.get("usa_share_of_total"):
        share = smoothed["afloat_20d_usa"] / (smoothed["afloat_20d_vessels"] + _EPS)
        out["ow__usa_share"] = share
        out[f"ow__usa_share__z{dz}"] = _zscore(share, dz)
    return out


def family_routing(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    cfg = get("features", "families", "routing")
    out: dict[str, pd.Series] = {}
    sums: dict[int, dict[str, pd.Series]] = {}
    for w in cfg["rolling_sum_windows_bd"]:
        w = int(w)
        sums[w] = {}
        for name in cfg["series"]:
            if name not in lagged.columns:
                raise KeyError(f"routing feature {name} is not in the panel")
            s = lagged[name].ffill().rolling(w, min_periods=w // 2).sum()
            sums[w][name] = s
            out[f"rt__{name}__sum{w}"] = s
    share_cfg = cfg["long_route_share"]
    for w, block in sums.items():
        num = sum(block[n] for n in share_cfg["numerator"])
        den = sum(block[n] for n in share_cfg["denominator"])
        share = num / (den + _EPS)
        out[f"rt__long_route_share__sum{w}"] = share
        for cw in cfg["change_windows_bd"]:
            out[f"rt__long_route_share__sum{w}__chg{int(cw)}"] = share.diff(int(cw))
    return out


def family_us_exports(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    cfg = get("features", "families", "us_exports")
    zwin = int(cfg["zscore_window_bd"])
    out: dict[str, pd.Series] = {}
    for name in cfg["series"]:
        if name not in lagged.columns:
            raise KeyError(f"us_exports feature {name} is not in the panel")
        s = lagged[name].ffill().rolling(
            int(cfg["smooth_window_bd"]), min_periods=1).mean()
        out[f"ux__{name}"] = s
        out[f"ux__{name}__z{zwin}"] = _zscore(s, zwin)
        for w in cfg["change_windows_bd"]:
            out[f"ux__{name}__chg{int(w)}"] = s.pct_change(int(w))
    return out


def _cape_share(lagged: pd.DataFrame, window: int) -> pd.Series:
    """Observed share of laden transits taking the Cape rather than a canal.

    Used to interpolate the Asian voyage distances between direct routing and
    the long way round. Built on a rolling sum because daily transit counts
    are small integers and frequently zero.
    """
    cfg = get("features", "families", "routing")
    share_cfg = cfg["long_route_share"]
    sums = {name: lagged[name].ffill().rolling(window, min_periods=window // 2).sum()
            for name in cfg["series"]}
    num = sum(sums[n] for n in share_cfg["numerator"])
    den = sum(sums[n] for n in share_cfg["denominator"])
    return (num / (den + _EPS)).clip(lower=0.0, upper=1.0)


def family_tonne_miles(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    """Ship-days demanded: volume weighted by how far each ton travels.

    The nearest thing to a supply/demand balance this Terminal supports. See
    features.yaml for why a direct fleet count is not available and why it
    matters less than it sounds over a 1-to-15 day horizon.
    """
    cfg = get("features", "families", "tonne_miles")
    legs = cfg["legs"]
    zwin = int(cfg["zscore_window_bd"])
    out: dict[str, pd.Series] = {}

    for name in legs:
        if name not in lagged.columns:
            raise KeyError(f"tonne_miles leg {name} is not in the panel")

    for window in cfg["rolling_sum_windows_bd"]:
        window = int(window)
        tons = {name: lagged[name].ffill().rolling(
            window, min_periods=window // 2).sum() for name in legs}
        total_tons = sum(tons.values())

        # Routing-adjusted distance for the legs that declare a long variant.
        cape = (_cape_share(lagged, window) if cfg.get("routing_adjusted")
                else None)
        tonne_miles = None
        for name, spec in legs.items():
            near = float(spec["distance_nm"])
            far = float(spec.get("distance_nm_long", spec["distance_nm"]))
            distance = near if cape is None else near + (far - near) * cape
            leg_tm = tons[name] * distance
            tonne_miles = leg_tm if tonne_miles is None else tonne_miles + leg_tm

        avg_haul = tonne_miles / (total_tons + _EPS)
        long_haul = ((tons["flow_amer_to_nasia"] + tons["flow_amer_to_sasia"])
                     / (total_tons + _EPS))

        out[f"tm__tonne_miles__sum{window}"] = tonne_miles
        out[f"tm__avg_haul_nm__sum{window}"] = avg_haul
        out[f"tm__long_haul_share__sum{window}"] = long_haul
        for label, series in (("tonne_miles", tonne_miles),
                              ("avg_haul_nm", avg_haul),
                              ("long_haul_share", long_haul)):
            out[f"tm__{label}__sum{window}__z{zwin}"] = _zscore(series, zwin)
            for change in cfg["change_windows_bd"]:
                out[f"tm__{label}__sum{window}__chg{int(change)}"] = (
                    series.pct_change(int(change)))
    return out


def family_utilisation(lagged: pd.DataFrame) -> dict[str, pd.Series]:
    """How much of the moving fleet is not available for the next cargo."""
    cfg = get("features", "families", "utilisation")
    zwin = int(cfg["zscore_window_bd"])
    out: dict[str, pd.Series] = {}

    for window in cfg["rolling_sum_windows_bd"]:
        window = int(window)
        for name in cfg["series"]:
            if name not in lagged.columns:
                raise KeyError(f"utilisation series {name} is not in the panel")
            s = lagged[name].ffill().rolling(window, min_periods=window // 2).sum()
            out[f"ut__{name}__sum{window}"] = s
            out[f"ut__{name}__sum{window}__z{zwin}"] = _zscore(s, zwin)
            for change in cfg["change_windows_bd"]:
                out[f"ut__{name}__sum{window}__chg{int(change)}"] = (
                    s.pct_change(int(change)))
        if cfg.get("afloat_over_exports"):
            exports = lagged["exports_global"].ffill().rolling(
                window, min_periods=window // 2).sum()
            afloat = lagged["afloat_20d_volume"].ffill()
            ratio = afloat / (exports + _EPS)
            out[f"ut__afloat_over_exports__sum{window}"] = ratio
            out[f"ut__afloat_over_exports__sum{window}__z{zwin}"] = _zscore(ratio, zwin)
            for change in cfg["change_windows_bd"]:
                out[f"ut__afloat_over_exports__sum{window}__chg{int(change)}"] = (
                    ratio.pct_change(int(change)))
    return out


def _roll_mask(index: pd.DatetimeIndex, days: int) -> pd.Series:
    """True on days near a month turn, where a generic future probably rolled.

    Returns on these days are discarded rather than used. Losing a genuine
    move costs a little signal; keeping a roll step teaches the model that
    freight jumps 20% on the first of the month.
    """
    day = pd.Series(index.day, index=index)
    month_end = pd.Series(index.days_in_month, index=index)
    return (day <= days) | (day > month_end - days)


def family_baltic(lagged: pd.DataFrame, limit: int) -> dict[str, pd.Series]:
    """The Baltic traded routes, and the assessment-to-traded gap.

    IKD1 shares a route with the target and has none of the target's serial
    correlation, so the gap between them measures how far the assessment
    still has to travel. See features.yaml for the measured basis-to-return
    correlations and for why raw returns of these contracts are not used.
    """
    cfg = get("features", "families", "baltic")
    zwin = int(cfg["zscore_window_bd"])
    mask = _roll_mask(lagged.index, int(cfg["roll_mask_days"]))
    out: dict[str, pd.Series] = {}

    levels: dict[str, pd.Series] = {}
    for name in cfg["series"]:
        if name not in lagged.columns:
            raise KeyError(f"baltic series {name} is not in the panel")
        level = lagged[name].ffill(limit=limit)
        levels[name] = level
        out[f"blt__{name}"] = level
        out[f"blt__{name}__z{zwin}"] = _zscore(np.log(level), zwin)
        log_level = np.log(level)
        for window in cfg["return_windows_bd"]:
            window = int(window)
            ret = log_level.diff(window)
            # A window ending on or spanning a roll is discarded. For a
            # multi-day window that means masking the whole window, not just
            # the roll day itself.
            spans_roll = mask.rolling(window + 1, min_periods=1).max().astype(bool)
            out[f"blt__{name}__ret{window}"] = ret.where(~spans_roll)

    if cfg.get("basis_to_spot"):
        same = cfg["same_route_as_target"]
        spot = fill_bounded(lagged[SPOT], limit)
        basis = np.log((levels[same] + _EPS) / (spot + _EPS))
        out["blt__basis_to_spot"] = basis
        out[f"blt__basis_to_spot__z{zwin}"] = _zscore(basis, zwin)
        for window in cfg["change_windows_bd"]:
            out[f"blt__basis_to_spot__chg{int(window)}"] = basis.diff(int(window))

    spread_cfg = cfg["east_west_spread"]
    east_west = np.log((levels[spread_cfg["numerator"]] + _EPS)
                       / (levels[spread_cfg["denominator"]] + _EPS))
    out["blt__east_west_spread"] = east_west
    out[f"blt__east_west_spread__z{zwin}"] = _zscore(east_west, zwin)
    for window in cfg["change_windows_bd"]:
        out[f"blt__east_west_spread__chg{int(window)}"] = east_west.diff(int(window))
    return out


def family_curve(lagged: pd.DataFrame, limit: int) -> dict[str, pd.Series]:
    """FFA curve shape. Also the source of the forward benchmark.

    Ratios are taken in logs so they sit on the same scale as the target,
    which makes the forward benchmark a direct comparison rather than a
    rescaled one.
    """
    cfg = get("features", "families", "curve")
    out: dict[str, pd.Series] = {}
    cols = {t: "ffa_" + t.replace("+", "").lower() for t in cfg["ffa_tenors"]}
    present = {t: c for t, c in cols.items() if c in lagged.columns}
    if not present:
        raise KeyError("no FFA tenors are present in the panel")
    spot = fill_bounded(lagged[SPOT], limit)
    for col in present.values():
        s = lagged[col].ffill()
        out[f"crv__{col}"] = s
        for w in cfg["change_windows_bd"]:
            out[f"crv__{col}__chg{int(w)}"] = np.log(
                (s + _EPS) / (s.shift(int(w)) + _EPS))
    if cfg.get("spot_vs_m1_ratio"):
        for near in ("ffa_m0", "ffa_m1"):
            if near in lagged.columns:
                out[f"crv__logfwd_{near[-2:]}_over_spot"] = np.log(
                    (lagged[near].ffill() + _EPS) / (spot + _EPS))
    if cfg.get("m1_vs_m3_slope") and {"ffa_m1", "ffa_m3"} <= set(lagged.columns):
        out["crv__m1_m3_slope"] = np.log(
            (lagged["ffa_m3"].ffill() + _EPS) / (lagged["ffa_m1"].ffill() + _EPS))
    return out


def family_seasonal(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    cfg = get("features", "families", "seasonal")
    out: dict[str, pd.Series] = {}
    doy = np.asarray(index.dayofyear, dtype=float)
    for k in range(1, int(cfg["fourier_order"]) + 1):
        out[f"sea__sin{k}"] = pd.Series(
            np.sin(2 * np.pi * k * doy / 365.25), index=index)
        out[f"sea__cos{k}"] = pd.Series(
            np.cos(2 * np.pi * k * doy / 365.25), index=index)
    if cfg.get("day_of_week"):
        dow = np.asarray(index.dayofweek, dtype=float)
        out["sea__dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 5.0), index=index)
        out["sea__dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 5.0), index=index)
    return out


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def build_targets(wide: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Forward log change of the freight assessment, per horizon.

    A target exists only where BOTH ends are genuine prints. During the
    weekly-only stretch of 2023 the 1-day target is therefore almost entirely
    absent while the 5-day target survives, which is the correct outcome: a
    forward-filled price differenced against itself is a zero return that
    never happened, and a model trained on those learns that freight does not
    move.
    """
    horizons = get("target", "target", "horizons_bd")
    if get("target", "target", "transform") != "log_change":
        raise ValueError("only the log_change target transform is implemented")
    require_fresh = bool(get("target", "staleness", "require_fresh_both_ends"))

    filled = fill_bounded(wide[SPOT], limit)
    fresh = freshness(wide, SPOT)
    log_price = np.log(filled)

    out = pd.DataFrame(index=wide.index)
    for h in horizons:
        h = int(h)
        y = log_price.shift(-h) - log_price
        if require_fresh:
            valid = fresh & fresh.shift(-h).fillna(False)
        else:
            valid = y.notna()
        out[f"{TARGET_PREFIX}{h}"] = y.where(valid)
        out[f"{FRESH_PREFIX}{h}"] = valid
    return out


def forward_benchmark(wide: pd.DataFrame, limit: int) -> pd.DataFrame:
    """What the FFA curve implies for the log change, per horizon.

    The M+0 contract settles against the average of the current month, so at
    short horizons it sits close to spot; M+1 is the market's view a month
    out. Neither is a clean h-day-ahead forward - no such instrument exists -
    so this is an approximation and is labelled as one. It is still the right
    benchmark, because it is the only forward-looking freight price the desk
    can actually transact.
    """
    cfg = get("model", "benchmarks", "ffa_forward")
    horizons = get("target", "target", "horizons_bd")
    short_col = "ffa_" + cfg["tenor_for_short_h"].replace("+", "").lower()
    long_col = "ffa_" + cfg["tenor_for_long_h"].replace("+", "").lower()
    spot = fill_bounded(wide[SPOT], limit)
    out = pd.DataFrame(index=wide.index)
    for h in horizons:
        h = int(h)
        # Inside a week, M+0 is the relevant contract; three weeks out the
        # curve's nearest tenor stops being the right comparison and M+1 is.
        col = short_col if h <= 5 else long_col
        if col not in wide.columns:
            raise KeyError(f"FFA benchmark needs {col}, absent from the panel")
        fwd = wide[col].ffill()
        out[f"{TARGET_PREFIX}{h}"] = np.log((fwd + _EPS) / (spot + _EPS))
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build(wide: pd.DataFrame, lags: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Returns features X, targets Y, and the FFA benchmark, all aligned."""
    limit = int(get("target", "staleness", "max_forward_fill_bd"))
    start = pd.Timestamp(get("target", "calendar", "start"))
    end_cfg = get("target", "calendar", "end")

    for required in (SPOT, PACIFIC):
        if required not in wide.columns:
            raise KeyError(f"the panel has no {required}; nothing can be built")

    lagged = apply_lags(wide, lags)

    columns: dict[str, pd.Series] = {}
    columns.update(family_arb(lagged))
    columns.update(family_freight_own(lagged, limit))
    columns.update(family_on_water(lagged))
    columns.update(family_routing(lagged))
    columns.update(family_us_exports(lagged))
    columns.update(family_tonne_miles(lagged))
    columns.update(family_utilisation(lagged))
    columns.update(family_baltic(lagged, limit))
    columns.update(family_curve(lagged, limit))
    columns.update(family_seasonal(wide.index))

    X = pd.DataFrame(columns, index=wide.index).replace([np.inf, -np.inf], np.nan)
    Y = build_targets(wide, limit)
    F = forward_benchmark(wide, limit)

    end = pd.Timestamp(end_cfg) if end_cfg else wide.index.max()
    window = (X.index >= start) & (X.index <= end)
    return {"X": X.loc[window], "Y": Y.loc[window], "forward": F.loc[window]}


def family_of(column: str) -> str:
    """Family prefix of a feature column, for grouped reporting."""
    return column.split("__", 1)[0]
