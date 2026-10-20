"""Transparent daily gas balance for one declared perimeter, in GWh/d, with an explicit residual.

Identity (configured in ``configs/sources/fundamentals.yaml`` ``balance``; positive = into storage):

    dS_implied = production + pipeline_imports + lng_sendout - demand - exports [- balancing]
    residual   = dS_observed - dS_implied

Rules:

* A component is the sum of its configured series. If any of those series is missing on a day the
  component is **NaN** (never 0), and so are ``dS_implied`` and ``residual``; ``residual_reason``
  names what was missing.
* Volume -> energy only through ``contracts.units.GasQuality`` with the configured GCV (and its
  stated source). Pure energy-rate units go through ``convert_energy_rate``. Anything else raises.
* Linepack / TSO balancing has no series; with ``treatment: absorbed_in_residual`` it is excluded
  from the identity and documented as living inside the residual.
* Input must be an **as-of-resolved** observation frame (``build_balance`` resolves for you).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.units import GasQuality, UnitError, convert_energy_rate
from ttf_forecaster.data.base import DataError
from ttf_forecaster.data.fundamentals import to_daily
from ttf_forecaster.features.point_in_time import resolve_as_of
from ttf_forecaster.settings import ConfigError, load_config

_RATE_UNITS = {"MWh/d", "GWh/d", "TWh/d", "MMBtu/d"}
_VOLUME_RATE_TO_MCM = {"mcm/d": 1.0, "bcm/d": 1000.0}


def load_balance_config(config_dir: str | None = None) -> dict[str, Any]:
    """The ``balance`` block of ``configs/sources/fundamentals.yaml``."""
    cfg = load_config("sources/fundamentals", config_dir).get("balance")
    if not cfg:
        raise ConfigError("sources/fundamentals.yaml: missing 'balance' block")
    for key in ("perimeter", "gas_quality", "components", "observed_storage_change",
                "residual_monitoring"):
        if key not in cfg:
            raise ConfigError(f"balance config missing key {key!r}")
    return cfg


def gas_quality_from_config(cfg: Mapping[str, Any]) -> GasQuality:
    """``GasQuality`` from the balance ``gas_quality`` block; the source must be stated."""
    gq = cfg["gas_quality"]
    if not gq.get("source"):
        raise ConfigError("balance.gas_quality.source must state where the GCV comes from")
    return GasQuality(float(gq["gcv_kwh_per_m3"]), str(gq["reference"]))


def to_gwhd(values: pd.Series, unit: str, quality: GasQuality) -> pd.Series:
    """Convert a daily rate series to GWh/d. Volumes need ``quality`` (GCV basis)."""
    if unit in _RATE_UNITS:
        factor = convert_energy_rate(1.0, unit, "GWh/d")
        return values * factor
    if unit in _VOLUME_RATE_TO_MCM:
        return values * _VOLUME_RATE_TO_MCM[unit] * quality.mcm_to_gwh(1.0)
    raise UnitError(f"cannot convert unit {unit!r} to GWh/d in the balance")


def _series_unit(resolved: pd.DataFrame, series_id: str) -> str | None:
    units = resolved.loc[resolved["series_id"] == series_id, "unit"].dropna().unique()
    if len(units) > 1:
        raise DataError(f"{series_id}: mixed units {sorted(units)} in one series")
    return str(units[0]) if len(units) else None


def _daily_gwhd(resolved: pd.DataFrame, sid: str, index: pd.DatetimeIndex,
                quality: GasQuality) -> pd.Series:
    unit = _series_unit(resolved, sid)
    if unit is None:
        return pd.Series(np.nan, index=index, name=sid)
    return to_gwhd(to_daily(resolved, sid).reindex(index), unit, quality).rename(sid)


@dataclass(frozen=True)
class BalanceResult:
    """Balance output.

    Attributes:
        table: daily frame (naive local gas day) with one column per component (signed GWh/d as
            it enters the identity), ``implied_storage_change_gwhd``,
            ``observed_storage_change_gwhd``, ``residual_gwhd``, ``residual_reason``.
        series: unsigned GWh/d of every input series (for audit).
        perimeter: the declared perimeter block.
        notes: statements the reader needs (e.g. balancing absorbed in residual, GCV used).
    """

    table: pd.DataFrame
    series: pd.DataFrame
    perimeter: Mapping[str, Any]
    notes: tuple[str, ...]


def compute_balance(
    resolved: pd.DataFrame,
    cfg: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> BalanceResult:
    """Daily balance over naive local gas days [start, end] from an as-of-resolved frame."""
    quality = gas_quality_from_config(cfg)
    index = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    table = pd.DataFrame(index=index)
    audit: dict[str, pd.Series] = {}
    reasons: list[pd.Series] = []
    notes = [f"GCV {quality.gcv_kwh_per_m3} kWh/m3 ({quality.reference}); "
             f"source: {cfg['gas_quality']['source']}"]

    def _component(name: str, sign: int, series_ids: list[str]) -> None:
        if sign not in (1, -1):
            raise ConfigError(f"balance component {name}: sign must be +1 or -1")
        if not series_ids:
            raise ConfigError(f"balance component {name}: no series configured")
        parts = []
        for sid in series_ids:
            s = _daily_gwhd(resolved, sid, index, quality)
            audit[sid] = s
            parts.append(s)
            reasons.append(pd.Series(np.where(s.isna(), f"{name}:{sid}", ""), index=index))
        total = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        total[pd.concat(parts, axis=1).isna().any(axis=1)] = np.nan
        table[name] = sign * total

    for name, comp in cfg["components"].items():
        _component(name, int(comp["sign"]), list(comp.get("series") or []))
    bal = cfg.get("balancing") or {"treatment": "absorbed_in_residual"}
    if bal.get("treatment") == "series":
        _component("balancing", -1, list(bal.get("series") or []))
    elif bal.get("treatment") == "absorbed_in_residual":
        notes.append("linepack / TSO balancing not observed: absorbed in the residual")
    else:
        raise ConfigError(f"balance.balancing.treatment {bal.get('treatment')!r} unknown")

    component_cols = list(table.columns)
    table["implied_storage_change_gwhd"] = table[component_cols].sum(
        axis=1, min_count=len(component_cols))
    table.loc[table[component_cols].isna().any(axis=1), "implied_storage_change_gwhd"] = np.nan

    obs_cfg = cfg["observed_storage_change"]
    method = obs_cfg["method"]
    if method == "injection_minus_withdrawal":
        inj = _daily_gwhd(resolved, obs_cfg["injection_series"], index, quality)
        wd = _daily_gwhd(resolved, obs_cfg["withdrawal_series"], index, quality)
        observed = inj - wd
        reasons.append(pd.Series(np.where(observed.isna(), "observed_storage_change", ""),
                                 index=index))
    elif method == "inventory_diff":
        sid = obs_cfg["inventory_series"]
        unit = _series_unit(resolved, sid)
        if unit not in (None, "TWh", "GWh"):
            raise UnitError(f"{sid}: inventory unit {unit!r} not supported")
        inv = to_daily(resolved, sid).reindex(
            pd.date_range(index[0] - pd.Timedelta(days=1), index[-1], freq="D"))
        scale = 1000.0 if unit == "TWh" else 1.0
        observed = (inv.diff() * scale).reindex(index)
        reasons.append(pd.Series(np.where(observed.isna(), "observed_storage_change", ""),
                                 index=index))
    else:
        raise ConfigError(f"observed_storage_change.method {method!r} unknown")
    table["observed_storage_change_gwhd"] = observed
    table["residual_gwhd"] = observed - table["implied_storage_change_gwhd"]

    joined = pd.concat(reasons, axis=1).apply(
        lambda r: ";".join(sorted({x for x in r if x})), axis=1) if reasons else ""
    table["residual_reason"] = np.where(table["residual_gwhd"].isna(),
                                        "missing: " + joined.astype(str), "")
    return BalanceResult(table, pd.DataFrame(audit, index=index), cfg["perimeter"], tuple(notes))


def build_balance(
    observations: pd.DataFrame,
    as_of: pd.Timestamp | str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Mapping[str, Any] | None = None,
) -> BalanceResult:
    """Resolve ``observations`` as of ``as_of`` (PIT policy §2) and compute the balance."""
    cfg = cfg or load_balance_config()
    resolved = resolve_as_of(observations, as_of)
    return compute_balance(resolved, cfg, start, end)


def monitor_residual(table: pd.DataFrame, monitoring: Mapping[str, Any]) -> pd.DataFrame:
    """Rolling residual diagnostics and alert flags.

    Columns: ``residual_roll_mean``, ``residual_roll_std`` (trailing ``window_days``, NaN-aware,
    ``min_periods``), ``flag_abs`` (|residual| > ``abs_band_gwhd``), ``flag_bias``
    (|rolling mean| > ``mean_band_gwhd``), ``flag_z`` (|residual − previous rolling mean| /
    previous rolling std > ``z_band``), ``flag_any``. NaN residual → all flags False and
    ``residual_missing`` True (missing is reported, not flagged as an outlier).
    """
    r = table["residual_gwhd"].astype(float)
    w = int(monitoring["window_days"])
    mp = int(monitoring.get("min_periods", max(2, w // 3)))
    mean = r.rolling(w, min_periods=mp).mean()
    std = r.rolling(w, min_periods=mp).std()
    prev_mean, prev_std = mean.shift(1), std.shift(1)
    z = (r - prev_mean).abs() / prev_std.where(prev_std > 0)
    out = pd.DataFrame(index=table.index)
    out["residual_gwhd"] = r
    out["residual_roll_mean"] = mean
    out["residual_roll_std"] = std
    out["flag_abs"] = (r.abs() > float(monitoring["abs_band_gwhd"])).fillna(False)
    out["flag_bias"] = (mean.abs() > float(monitoring["mean_band_gwhd"])).fillna(False)
    out["flag_z"] = (z > float(monitoring["z_band"])).fillna(False)
    out["flag_any"] = out[["flag_abs", "flag_bias", "flag_z"]].any(axis=1)
    out["residual_missing"] = r.isna()
    return out


def contributions(result: BalanceResult, freq: str = "MS") -> pd.DataFrame:
    """Period-mean signed GWh/d of each component, the implied and observed storage change, and the
    residual. A period containing a NaN day for a component reports NaN for that component (no
    silent averaging over gaps); ``n_days`` and ``n_complete_days`` show coverage."""
    t = result.table.drop(columns=["residual_reason"])
    grouped = t.resample(freq)
    out = grouped.mean()
    complete = grouped.count()
    sizes = grouped.size()
    out[complete.lt(sizes, axis=0)] = np.nan
    out["n_days"] = sizes
    out["n_complete_days"] = t.notna().all(axis=1).resample(freq).sum()
    return out
