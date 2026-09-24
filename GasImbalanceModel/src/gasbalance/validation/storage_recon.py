"""Storage reconciliation. SPEC 7 Phase 1 gate, SPEC 8 daily reconciliation.

The check is the storage identity from SPEC 5:

    S_t = S_(t-1) + injections_t - withdrawals_t

Reported inventory and reported flows are three independent series. Rebuilding
the inventory path from the flows and comparing it to the reported path is the
strongest integrity test available before any modelling starts: if it does not
close on ACTUALS, no forecast built on top of it can be trusted.

The residual is reported, never absorbed. SPEC 8: "never let it be absorbed
into the residual."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from gasbalance.units import Units

log = logging.getLogger(__name__)

INVENTORY = "gas_in_storage"
INJECTION = "injection"
WITHDRAWAL = "withdrawal"
CAPACITY = "working_gas_volume"


@dataclass
class StorageReconReport:
    """Per-country and aggregate closure of the storage identity."""

    daily: pd.DataFrame
    by_country: pd.DataFrame
    aggregate: pd.DataFrame
    alert_threshold_mcm_d: float

    @property
    def worst_country_bias(self) -> float:
        return float(self.by_country["mean_error_mcm_d"].abs().max())

    def summary_lines(self) -> list[str]:
        agg = self.aggregate
        return [
            f"days covered            {len(agg)}",
            f"countries               {len(self.by_country)}",
            f"aggregate mean error    {agg['error_mcm_d'].mean():+.2f} mcm/d",
            f"aggregate abs mean      {agg['error_mcm_d'].abs().mean():.2f} mcm/d",
            f"aggregate worst day     {agg['error_mcm_d'].abs().max():.2f} mcm/d",
            f"worst country bias      {self.worst_country_bias:.2f} mcm/d",
            f"alert threshold         {self.alert_threshold_mcm_d:.2f} mcm/d",
        ]


def reconcile_storage(
    obs: pd.DataFrame, units: Units, alert_threshold_mcm_d: float
) -> StorageReconReport:
    """Rebuild inventory from flows and measure the gap against reported stock.

    `obs` is the internal long format: inventory in TWh, flows in mcm/d.
    """
    wide = (
        obs.pivot_table(
            index=["obs_date", "country"], columns="series_id", values="value", aggfunc="last"
        )
        .reset_index()
        .sort_values(["country", "obs_date"])
    )
    for column in (INVENTORY, INJECTION, WITHDRAWAL):
        if column not in wide.columns:
            raise ValueError(f"storage reconciliation needs {column!r}; got {list(wide.columns)}")

    # Flows are mcm/d; inventory is TWh. Move flows onto TWh/d to compare.
    gwh_per_mcm = units.canonical_gcv
    twh_per_gwh = 1.0 / units.energy_to_gwh(1.0, "TWh")
    wide["net_injection_twh_d"] = (
        (wide[INJECTION] - wide[WITHDRAWAL]) * gwh_per_mcm * twh_per_gwh
    )

    wide["prev_inventory_twh"] = wide.groupby("country")[INVENTORY].shift(1)
    wide["implied_inventory_twh"] = wide["prev_inventory_twh"] + wide["net_injection_twh_d"]
    wide["error_twh"] = wide[INVENTORY] - wide["implied_inventory_twh"]
    # Express the error in the flow unit, so it is comparable with the 20 mcm/d
    # alert threshold SPEC 8 sets.
    wide["error_mcm_d"] = wide["error_twh"] / (gwh_per_mcm * twh_per_gwh)

    daily = wide.dropna(subset=["error_mcm_d"]).copy()

    by_country = (
        daily.groupby("country")
        .agg(
            days=("error_mcm_d", "size"),
            mean_error_mcm_d=("error_mcm_d", "mean"),
            abs_mean_mcm_d=("error_mcm_d", lambda s: s.abs().mean()),
            worst_mcm_d=("error_mcm_d", lambda s: s.abs().max()),
        )
        .reset_index()
        .sort_values("abs_mean_mcm_d", ascending=False)
    )

    aggregate = (
        daily.groupby("obs_date", as_index=False)
        .agg(
            error_mcm_d=("error_mcm_d", "sum"),
            inventory_twh=(INVENTORY, "sum"),
        )
        .sort_values("obs_date")
    )

    breaching = by_country[by_country["abs_mean_mcm_d"].abs() > alert_threshold_mcm_d]
    if not breaching.empty:
        log.warning(
            "storage identity fails beyond %.1f mcm/d for: %s",
            alert_threshold_mcm_d,
            breaching["country"].tolist(),
        )

    return StorageReconReport(
        daily=daily,
        by_country=by_country,
        aggregate=aggregate,
        alert_threshold_mcm_d=alert_threshold_mcm_d,
    )
