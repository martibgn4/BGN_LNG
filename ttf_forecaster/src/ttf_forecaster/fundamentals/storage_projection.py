"""Transparent scenario projection of storage from ``as_of`` to the end of the current season.

Method (parameters in ``configs/scenarios.yaml``):

1. Net-flow profile (% WGV/day by day-of-gas-year) from gas years strictly before the origin's gas
   year (``fundamentals.scenarios.estimate_net_flow_profile``).
2. Scenario flow = scenario adjustment of ``profile x current WGV`` (GWh/d).
3. Deterministic path: start from the last published inventory, add one day's flow at a time,
   enforcing the withdrawal / injection ratchet (if configured) and ``0 <= S <= WGV``.
4. Uncertainty: seeded block bootstrap of historical daily deviations (actual − profile) from the
   **same season** of the estimation years, added to the scenario flow; P10/P50/P90 of the end
   level and probabilities of ending below the configured thresholds.

Season ends: 31 March (winter, Oct–Mar) and 30 September (summer, Apr–Sep). Positive flow =
injection. Energies in TWh, rates in GWh/d.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.fundamentals.scenarios import (
    InsufficientHistory,
    NetFlowProfile,
    Scenario,
    day_of_gas_year,
    gas_year,
)
from ttf_forecaster.settings import ConfigError


def season_of(day: pd.Timestamp) -> str:
    """``winter`` (Oct–Mar) or ``summer`` (Apr–Sep) for a naive local date."""
    return "winter" if day.month >= 10 or day.month <= 3 else "summer"


def season_end(day: pd.Timestamp, season_ends: Mapping[str, str] | None = None) -> pd.Timestamp:
    """Last gas day of the season containing ``day`` (default 31 Mar / 30 Sep)."""
    ends = season_ends or {"winter_end_mmdd": "03-31", "summer_end_mmdd": "09-30"}
    if season_of(day) == "winter":
        mm, dd = (int(x) for x in str(ends["winter_end_mmdd"]).split("-"))
        year = day.year + 1 if day.month >= 10 else day.year
    else:
        mm, dd = (int(x) for x in str(ends["summer_end_mmdd"]).split("-"))
        year = day.year
    return pd.Timestamp(year=year, month=mm, day=dd)


@dataclass(frozen=True)
class Ratchet:
    """Fill-dependent flow limits (GWh/d), piecewise linear in fill %; ``None`` = unlimited."""

    withdrawal_fill_pct: np.ndarray | None = None
    withdrawal_max_gwhd: np.ndarray | None = None
    injection_fill_pct: np.ndarray | None = None
    injection_max_gwhd: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> Ratchet:
        if not cfg or not cfg.get("enabled", False):
            return cls()

        def _curve(key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
            block = cfg.get(key)
            if not block:
                return None, None
            x = np.asarray(block["fill_pct"], dtype=float)
            y = np.asarray(block["gwhd"], dtype=float)
            if x.shape != y.shape or np.any(np.diff(x) <= 0) or np.any(y < 0):
                raise ConfigError(f"ratchet.{key}: fill_pct must increase and gwhd be >= 0")
            return x, y

        wx, wy = _curve("withdrawal_max_gwhd")
        ix, iy = _curve("injection_max_gwhd")
        return cls(wx, wy, ix, iy)

    def limits(self, fill_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(max withdrawal, max injection) in GWh/d at the given fill levels."""
        inf = np.full(np.shape(fill_pct), np.inf)
        wd = inf if self.withdrawal_fill_pct is None else np.interp(
            fill_pct, self.withdrawal_fill_pct, self.withdrawal_max_gwhd)
        inj = inf if self.injection_fill_pct is None else np.interp(
            fill_pct, self.injection_fill_pct, self.injection_max_gwhd)
        return wd, inj


def simulate_paths(
    level0_twh: float,
    flows_gwhd: np.ndarray,
    wgv_twh: float,
    ratchet: Ratchet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate flows day by day under physical bounds.

    Args:
        level0_twh: starting inventory.
        flows_gwhd: ``(n_paths, horizon)`` desired daily net flows (positive = injection).
        wgv_twh: working gas volume (upper bound).
        ratchet: flow limits as a function of fill.

    Returns:
        ``(end_level_twh, min_level_twh, binding_days)`` per path; ``binding_days`` counts days on
        which a ratchet or the 0 / WGV bound reduced the desired flow.
    """
    flows = np.atleast_2d(np.asarray(flows_gwhd, dtype=float))
    n_paths, horizon = flows.shape
    level = np.full(n_paths, float(level0_twh))
    low = level.copy()
    binding = np.zeros(n_paths, dtype=int)
    for t in range(horizon):
        fill = level / wgv_twh * 100.0
        wd_max, inj_max = ratchet.limits(fill)
        desired = flows[:, t]
        f = np.clip(desired, -wd_max, inj_max)
        f = np.clip(f, -level * 1000.0, (wgv_twh - level) * 1000.0)
        binding += (np.abs(f - desired) > 1e-9).astype(int)
        level = level + f / 1000.0
        low = np.minimum(low, level)
    return level, low, binding


@dataclass(frozen=True)
class ProjectionResult:
    """Outcome of one projection. Levels in TWh, fills in % of WGV."""

    scenario: str
    season: str
    start_day: pd.Timestamp
    season_end_day: pd.Timestamp
    horizon_days: int
    start_level_twh: float
    wgv_twh: float
    end_level_twh: float
    end_fill_pct: float
    binding_days: int
    quantiles_fill_pct: dict[float, float] = field(default_factory=dict)
    prob_below_pct: dict[float, float] = field(default_factory=dict)
    n_paths: int = 0


def _bootstrap_deviations(
    pool: np.ndarray, horizon: int, n_paths: int, block_days: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_paths, horizon)`` block-bootstrap draws from a 1-d deviation pool (% WGV/day)."""
    if len(pool) == 0:
        raise InsufficientHistory("empty deviation pool for bootstrap")
    block = max(1, min(block_days, len(pool)))
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(pool) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    return pool[idx]


def project_storage(
    *,
    level0_twh: float,
    start_day: pd.Timestamp,
    wgv_twh: float,
    profile: NetFlowProfile,
    scenario: Scenario,
    origin_day: pd.Timestamp,
    scen_cfg: Mapping[str, Any],
    bootstrap: bool = True,
    seed_key: Sequence[int] = (),
) -> ProjectionResult:
    """Project storage from ``start_day`` (last published inventory day) to the end of the season
    containing ``origin_day``.

    Args:
        level0_twh: inventory on ``start_day`` (as published by the origin).
        start_day: naive local gas day of ``level0_twh``.
        wgv_twh: working gas volume used for bounds and to scale the % profile.
        profile: net-flow profile estimated from gas years before the origin's gas year.
        scenario: adjustment to the profile.
        origin_day: naive local date of the forecast origin (sets the season).
        scen_cfg: ``configs/scenarios.yaml`` mapping.
        bootstrap: also run the seeded bootstrap for quantiles / threshold probabilities.
        seed_key: extra integers mixed into the seed (e.g. the origin's ordinal) so different
            origins draw different but reproducible paths.

    Raises:
        InsufficientHistory: the profile's season has no deviations to bootstrap.
        ValueError: non-positive WGV or inventory outside [0, WGV].
    """
    if not np.isfinite(wgv_twh) or wgv_twh <= 0:
        raise ValueError(f"working gas volume must be > 0, got {wgv_twh}")
    if not (0.0 <= level0_twh <= wgv_twh * (1 + 1e-6)):
        raise ValueError(f"inventory {level0_twh} TWh outside [0, WGV={wgv_twh}]")
    if gas_year(origin_day) <= max(profile.years, default=-10**9):
        raise ValueError("profile must be estimated from gas years strictly before the origin's")
    season = season_of(origin_day)
    end_day = season_end(origin_day, scen_cfg.get("season_ends"))
    days = pd.date_range(start_day + pd.Timedelta(days=1), end_day, freq="D")
    horizon = len(days)
    ratchet = Ratchet.from_config(scen_cfg.get("ratchet"))
    base = profile.pct_per_day[np.asarray(day_of_gas_year(days), dtype=int)] / 100.0 * wgv_twh * 1000.0 \
        if horizon else np.zeros(0)
    flows = scenario.apply(base)
    end, _, binding = simulate_paths(level0_twh, flows[None, :], wgv_twh, ratchet)
    end_twh = float(end[0])
    result: dict[str, Any] = dict(
        scenario=scenario.name, season=season, start_day=start_day, season_end_day=end_day,
        horizon_days=horizon, start_level_twh=float(level0_twh), wgv_twh=float(wgv_twh),
        end_level_twh=end_twh, end_fill_pct=end_twh / wgv_twh * 100.0,
        binding_days=int(binding[0]),
    )
    if bootstrap:
        bcfg = scen_cfg["bootstrap"]
        n_paths = int(bcfg["n_paths"])
        rng = np.random.default_rng([int(bcfg["seed"]), *[int(k) for k in seed_key]])
        if horizon:
            pool = profile.season_deviations(season)
            dev = _bootstrap_deviations(pool, horizon, n_paths, int(bcfg["block_days"]), rng)
            paths = flows[None, :] + dev / 100.0 * wgv_twh * 1000.0
            ends, _, _ = simulate_paths(level0_twh, paths, wgv_twh, ratchet)
        else:
            ends = np.full(n_paths, float(level0_twh))
        fills = ends / wgv_twh * 100.0
        result["quantiles_fill_pct"] = {
            float(q): float(np.quantile(fills, q)) for q in bcfg["quantiles"]
        }
        result["prob_below_pct"] = {
            float(t): float(np.mean(fills < t)) for t in scen_cfg["thresholds_pct"][season]
        }
        result["n_paths"] = n_paths
    return ProjectionResult(**result)
