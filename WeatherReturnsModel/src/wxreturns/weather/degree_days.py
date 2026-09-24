"""Degree days, and the demand weighting that turns a map into one number.

TWO THINGS THAT ARE EASY TO GET WRONG AND ARE HANDLED EXPLICITLY.

**The kink.** Degree days are computed per grid point and only then weighted.
``max(0, base - mean(T))`` is not ``mean(max(0, base - T))``, because the
function kinks at zero. Averaging temperature first understates degree days
whenever some points are above the base and others below - which is most of
autumn and spring, and therefore most trading days. `features.yaml` carries
``compute_before_weighting: true`` and this module honours it.

**The convention.** US gas uses a 65F base. European practice (Eurostat) uses
a 15.5C threshold with an 18C reference, and 65F is 18.33C, so the two are not
the same number dressed differently. Mixing them shifts EU degree days by
roughly a tenth in shoulder season. Each market names its own convention and
this module refuses to guess one.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wxreturns.config import get
from wxreturns.data.openmeteo import region_points

log = logging.getLogger(__name__)

MEAN_VARIABLE = "temperature_2m_mean"


def convention_for(region: str) -> str:
    """Which degree-day convention a region uses, from its market."""
    kind = get("regions", "regions", region, "kind")
    if kind != "gas":
        raise ValueError(
            f"region {region!r} is {kind!r}, not gas. Degree days are a gas "
            "demand construct; crop weather uses weather/crop.py instead."
        )
    return "us" if region.startswith("us") else "eu"


def _hdd_cdd(temperature: pd.Series, convention: str
             ) -> tuple[pd.Series, pd.Series]:
    """Point-level HDD and CDD under the named convention."""
    cfg = get("features", "degree_days", convention)
    base = float(cfg["base_temperature_c"])
    zero = 0.0

    if cfg["convention"] == "eurostat":
        # HDD accrues to the REFERENCE temperature, but only on days whose mean
        # is at or below the BASE. The two differ (15.5 vs 18.0), so the series
        # steps rather than rising smoothly from zero - that discontinuity is
        # the convention, not a bug.
        reference = float(cfg["reference_temperature_c"])
        hdd = pd.Series(np.where(temperature <= base, reference - temperature, zero),
                        index=temperature.index)
        cdd = pd.Series(np.where(temperature > reference, temperature - reference, zero),
                        index=temperature.index)
    else:
        hdd = (base - temperature).clip(lower=zero)
        cdd = (temperature - base).clip(lower=zero)

    return hdd, cdd


def point_degree_days(panel: pd.DataFrame, region: str) -> pd.DataFrame:
    """HDD and CDD per point per (issue_date, target_date).

    Expects the canonical weather panel. Only the daily-mean variable is used;
    max/min are carried for the ag features and for diagnostics.
    """
    if not get("features", "degree_days", "compute_before_weighting"):
        raise ValueError(
            "compute_before_weighting is false. Averaging temperature before "
            "taking degree days understates them wherever the region straddles "
            "the base temperature. This module does not implement that path."
        )

    means = panel[panel["variable"] == MEAN_VARIABLE].copy()
    if means.empty:
        raise ValueError(
            f"no {MEAN_VARIABLE!r} rows in the panel for {region!r}; degree "
            "days cannot be computed from max/min alone without asserting a "
            "diurnal shape")

    convention = convention_for(region)
    hdd, cdd = _hdd_cdd(means["value"], convention)
    means["hdd"] = hdd
    means["cdd"] = cdd
    means["convention"] = convention
    return means.drop(columns=["variable", "value"])


def weight_to_region(point_frame: pd.DataFrame, region: str) -> pd.DataFrame:
    """Collapse points to one demand-weighted series per (issue, target).

    Weights come from regions.yaml and are normalised. A point missing on a
    date is a data failure: filling it with the region mean would quietly
    reweight the composite, so it raises when configured to.
    """
    weights = {p["name"]: p["weight"] for p in region_points(region)}
    frame = point_frame.copy()
    frame["weight"] = frame["point"].map(weights)

    missing = frame[frame["weight"].isna()]["point"].unique()
    if len(missing):
        raise ValueError(
            f"points {sorted(missing)} are in the data but not in "
            f"regions.yaml for {region!r}; the composite would silently drop them")

    keys = ["issue_date", "target_date", "region"]
    expected = len(weights)

    def _weighted(group: pd.DataFrame) -> pd.Series:
        total = group["weight"].sum()
        return pd.Series({
            "hdd": float(np.dot(group["hdd"], group["weight"]) / total),
            "cdd": float(np.dot(group["cdd"], group["weight"]) / total),
            "points_present": len(group),
            "weight_covered": float(total),
        })

    out = (frame.groupby(keys, dropna=False, sort=False)
           .apply(_weighted, include_groups=False)
           .reset_index())

    if get("features", "hygiene", "raise_on_missing"):
        short = out[out["points_present"] < expected]
        if not short.empty:
            worst = short.iloc[0]
            raise ValueError(
                f"{region!r} has {len(short)} date(s) missing points - e.g. "
                f"target {worst['target_date']} has {int(worst['points_present'])} "
                f"of {expected}. A partial composite is a different series, not "
                "a slightly noisier one. Refetch the gap or drop the dates "
                "deliberately."
            )
    return out


def region_degree_days(panel: pd.DataFrame, region: str) -> pd.DataFrame:
    """Panel to demand-weighted regional HDD/CDD. The usual entry point."""
    return weight_to_region(point_degree_days(panel, region), region)
