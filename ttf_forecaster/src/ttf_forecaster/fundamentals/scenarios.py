"""Gas-year calendar, PIT-safe seasonal normals, the net-storage-flow profile and scenarios.

Point-in-time rule (``docs/point_in_time_policy.md`` §5.3): every seasonal statistic here uses only
gas years **strictly before** a stated ``before_gas_year``. Callers pass the origin's gas year (or
an earlier one); nothing in this module looks at the clock.

Gas year ``Y`` runs 1 October ``Y`` to 30 September ``Y+1`` and is labelled by its start year.
Day-of-gas-year is 0 on 1 October.

Inputs are daily ``pd.Series`` indexed by naive local gas-day dates, as produced by
``data.fundamentals.to_daily`` from an as-of-resolved frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import DataError
from ttf_forecaster.settings import ConfigError, load_config

DAYS_PER_GAS_YEAR = 366  # day-of-gas-year index range 0..365 (leap years)
_CIRCLE = 365


class InsufficientHistory(DataError):
    """Fewer prior gas years than required for a seasonal statistic."""


# ---------------------------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------------------------


def gas_year(dates: pd.DatetimeIndex | pd.Timestamp) -> np.ndarray | int:
    """Gas year label (start year) of naive local dates."""
    if isinstance(dates, pd.Timestamp):
        return dates.year if dates.month >= 10 else dates.year - 1
    idx = pd.DatetimeIndex(dates)
    return np.where(idx.month >= 10, idx.year, idx.year - 1)


def day_of_gas_year(dates: pd.DatetimeIndex | pd.Timestamp) -> np.ndarray | int:
    """Days since 1 October of the date's gas year (0..365)."""
    if isinstance(dates, pd.Timestamp):
        start = pd.Timestamp(year=int(gas_year(dates)), month=10, day=1)
        return int((dates.normalize() - start).days)
    idx = pd.DatetimeIndex(dates).normalize()
    gy = np.asarray(gas_year(idx))
    starts = pd.to_datetime({"year": gy, "month": 10, "day": 1})
    return (idx - pd.DatetimeIndex(starts)).days.to_numpy()


def _circular_distance(a: np.ndarray, b: int) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, _CIRCLE - d)


# ---------------------------------------------------------------------------------------------
# Seasonal normals
# ---------------------------------------------------------------------------------------------


def seasonal_normal(
    series: pd.Series,
    target_days: pd.DatetimeIndex,
    before_gas_year: int,
    *,
    window_days: int = 3,
    min_years: int = 3,
) -> pd.DataFrame:
    """Same-day-of-gas-year mean and dispersion of ``series`` over prior gas years.

    For each target day: take, in each gas year ``y < before_gas_year``, the mean of ``series`` over
    days within ``±window_days`` (circular) of the target's day-of-gas-year; then average across
    years. ``std`` is the across-year standard deviation (ddof=1).

    Returns:
        Frame indexed by ``target_days`` with ``mean``, ``std``, ``n_years``. Rows with fewer than
        ``min_years`` contributing years are NaN (``n_years`` still reported).
    """
    s = series.dropna()
    gy = np.asarray(gas_year(s.index)) if len(s) else np.array([], dtype=int)
    hist = s.loc[gy < before_gas_year]
    out = pd.DataFrame(index=pd.DatetimeIndex(target_days), columns=["mean", "std", "n_years"],
                       dtype=float)
    if hist.empty:
        out["n_years"] = 0.0
        return out
    h_doy = np.asarray(day_of_gas_year(hist.index))
    h_gy = np.asarray(gas_year(hist.index))
    t_doy = np.asarray(day_of_gas_year(pd.DatetimeIndex(target_days)))
    vals = hist.to_numpy(dtype=float)
    for i, td in enumerate(t_doy):
        mask = _circular_distance(h_doy, int(td)) <= window_days
        if not mask.any():
            out.iloc[i] = [np.nan, np.nan, 0.0]
            continue
        per_year = pd.Series(vals[mask]).groupby(h_gy[mask]).mean()
        n = len(per_year)
        if n < min_years:
            out.iloc[i] = [np.nan, np.nan, float(n)]
        else:
            out.iloc[i] = [per_year.mean(), per_year.std(ddof=1), float(n)]
    return out


# ---------------------------------------------------------------------------------------------
# Net-flow profile
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NetFlowProfile:
    """Mean daily net storage flow by day-of-gas-year, in % of working gas volume per day.

    Attributes:
        pct_per_day: length-366 array indexed by day-of-gas-year (positive = injection).
        deviations: daily ``actual - profile`` (% WGV/day) over the estimation years, chronological,
            indexed by gas day.
        years: gas years used (all strictly before ``before_gas_year``).
        before_gas_year: exclusive upper bound on the estimation years.
    """

    pct_per_day: np.ndarray
    deviations: pd.Series
    years: tuple[int, ...]
    before_gas_year: int

    def season_deviations(self, season: str) -> np.ndarray:
        """Chronological deviations of ``season`` (``winter`` Oct–Mar, ``summer`` Apr–Sep)."""
        months = self.deviations.index.month
        mask = (months >= 10) | (months <= 3) if season == "winter" else (months >= 4) & (months <= 9)
        return self.deviations.loc[mask].to_numpy(dtype=float)


def net_flow_pct_of_wgv(net_flow_gwhd: pd.Series, wgv_twh: pd.Series) -> pd.Series:
    """Daily net flow (GWh/d) as % of the same day's working gas volume (TWh)."""
    wgv = wgv_twh.reindex(net_flow_gwhd.index)
    return net_flow_gwhd / (wgv * 1000.0) * 100.0


def estimate_net_flow_profile(
    net_flow_pct: pd.Series,
    before_gas_year: int,
    *,
    smoothing_window_days: int = 15,
    min_years: int = 3,
) -> NetFlowProfile:
    """Estimate the seasonal net-flow profile from gas years strictly before ``before_gas_year``.

    Raises:
        InsufficientHistory: fewer than ``min_years`` prior gas years with data.
    """
    s = net_flow_pct.dropna()
    if s.empty:
        raise InsufficientHistory("no net-flow history")
    gy = np.asarray(gas_year(s.index))
    hist = s.loc[gy < before_gas_year]
    years = tuple(sorted(set(np.asarray(gas_year(hist.index)).tolist()))) if len(hist) else ()
    if len(years) < min_years:
        raise InsufficientHistory(
            f"net-flow profile needs {min_years} gas years before {before_gas_year}; have {years}"
        )
    doy = np.asarray(day_of_gas_year(hist.index))
    raw = pd.Series(hist.to_numpy(dtype=float)).groupby(doy).mean()
    raw = raw.reindex(range(DAYS_PER_GAS_YEAR)).interpolate(limit_direction="both")
    # Day 365 exists only in leap gas years; smooth on a 365-day circle and copy day 364 to 365.
    base = raw.to_numpy()[:_CIRCLE]
    half = smoothing_window_days // 2
    padded = np.concatenate([base[-half:], base, base[:half]]) if half else base
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    smooth = np.convolve(padded, kernel, mode="valid") if half else base
    profile = np.append(smooth, smooth[-1])
    deviations = pd.Series(hist.to_numpy(dtype=float) - profile[doy], index=hist.index).sort_index()
    return NetFlowProfile(profile, deviations, years, before_gas_year)


# ---------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Adjustment applied to the profile flow (GWh/d, positive = injection).

    ``flow = withdrawal_multiplier * min(base, 0) + injection_multiplier * max(base, 0)
    + additive_gwhd``.
    """

    name: str
    withdrawal_multiplier: float = 1.0
    injection_multiplier: float = 1.0
    additive_gwhd: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.withdrawal_multiplier < 0 or self.injection_multiplier < 0:
            raise ConfigError(f"scenario {self.name}: multipliers must be >= 0")

    def apply(self, base_gwhd: np.ndarray) -> np.ndarray:
        """Scenario flow from a base (profile) flow array in GWh/d."""
        base = np.asarray(base_gwhd, dtype=float)
        return (
            self.withdrawal_multiplier * np.minimum(base, 0.0)
            + self.injection_multiplier * np.maximum(base, 0.0)
            + self.additive_gwhd
        )


def load_scenario_config(config_dir: str | None = None) -> dict[str, Any]:
    """The ``configs/scenarios.yaml`` mapping (validated for the keys this package needs)."""
    cfg = load_config("scenarios", config_dir)
    for key in ("scenarios", "season_ends", "profile", "bootstrap", "thresholds_pct", "ratchet"):
        if key not in cfg:
            raise ConfigError(f"scenarios.yaml: missing key {key!r}")
    return cfg


def scenarios_from_config(cfg: Mapping[str, Any]) -> dict[str, Scenario]:
    """Parse the ``scenarios`` block into ``Scenario`` objects."""
    out: dict[str, Scenario] = {}
    for name, spec in cfg["scenarios"].items():
        out[name] = Scenario(
            name=name,
            withdrawal_multiplier=float(spec.get("withdrawal_multiplier", 1.0)),
            injection_multiplier=float(spec.get("injection_multiplier", 1.0)),
            additive_gwhd=float(spec.get("additive_gwhd", 0.0)),
            description=str(spec.get("description", "")),
        )
    return out
