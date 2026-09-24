"""Phase 1 reconciliation. SPEC 7 acceptance gate, SPEC 8.

Three things are checked, and each one reports rather than repairs:

1. Internal netting. Every internal interconnection point is reported twice,
   by the TSO on each side. Signed, the pair must cancel. What survives is
   TSO disagreement, which is a data-quality number, not a gas flow.

2. Point coverage. A perimeter point that returns no data on a gas day is a
   hole in the balance. SPEC 7 asks for "no unexplained gaps"; a literal
   reading is unachievable against ENTSOG, so gaps are classified into
   expected (the registry says the point publishes nothing) and unexplained
   (the registry says it does, and it did not).

3. Tolerance breaches, against the thresholds in perimeter.yaml.

Nothing here fills, interpolates or zeroes a missing value. SPEC 11.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from gasbalance.perimeter import BALANCE_ROLES, ROLE_INTERNAL, Perimeter

log = logging.getLogger(__name__)


@dataclass
class ReconciliationReport:
    """What Phase 1 can and cannot prove about a date range."""

    days: int
    internal_net: pd.DataFrame
    internal_breaches: pd.DataFrame
    perimeter_flows: pd.DataFrame
    coverage: pd.DataFrame
    orphans: pd.DataFrame
    unexplained_missing: pd.DataFrame
    expected_missing: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.internal_breaches.empty and self.unexplained_missing.empty

    def summary_lines(self) -> list[str]:
        worst = (
            self.internal_net["internal_net_mcm_d"].abs().max()
            if not self.internal_net.empty
            else 0.0
        )
        return [
            f"days covered              {self.days}",
            f"internal net, worst day   {worst:.4f} mcm/d",
            f"internal breaches         {len(self.internal_breaches)}",
            f"one-sided internal points {len(self.orphans)} point-days "
            f"({self.orphans['gross_mcm_d'].sum():.0f} mcm/d gross, excluded from net)",
            f"points expected to report {len(self.coverage)}",
            f"unexplained missing       {len(self.unexplained_missing)}",
            f"expected missing          {len(self.expected_missing)}",
            f"VERDICT                   {'PASS' if self.passed else 'FAIL'}",
        ]


def reconcile(
    flows: pd.DataFrame, points: pd.DataFrame, perimeter: Perimeter
) -> ReconciliationReport:
    """Run the Phase 1 checks over an already-fetched set of daily flows."""
    signed = perimeter.signed_flows(flows, points)
    pairs = perimeter.internal_pairs(signed)
    internal_net = perimeter.internal_net_matched(pairs)
    orphans = perimeter.internal_orphans(pairs)
    breaches = perimeter.internal_net_breaches(internal_net)
    perim = perimeter.perimeter_flows(signed)

    days = sorted(flows["obs_date"].unique())
    wanted_roles = set(BALANCE_ROLES) | {ROLE_INTERNAL}
    expected = points[points["ring_role"].isin(wanted_roles)].copy()

    grid = expected.assign(key=1).merge(
        pd.DataFrame({"obs_date": days, "key": 1}), on="key"
    ).drop(columns="key")
    reported = signed[["obs_date", "point_key", "operator_key", "direction_key"]].assign(
        reported=True
    )
    coverage = grid.merge(
        reported, on=["obs_date", "point_key", "operator_key", "direction_key"], how="left"
    )
    coverage["reported"] = coverage["reported"].fillna(False).astype(bool)

    missing = coverage[~coverage["reported"]].copy()
    # `has_data` is ENTSOG's own statement about whether a point publishes at
    # all. False means the absence is expected and is not a model gap.
    publishes = missing["has_data"].fillna(False).astype(bool)
    unexplained = missing[publishes]
    expected_missing = missing[~publishes]

    notes: list[str] = []
    if not unexplained.empty:
        by_point = unexplained.groupby("point_key").size().sort_values(ascending=False)
        notes.append(
            f"{len(unexplained)} point-days missing from points ENTSOG says do publish; "
            f"worst offenders: {by_point.head(5).to_dict()}"
        )
        log.warning(notes[-1])
    if not orphans.empty:
        notes.append(
            f"{len(orphans)} internal point-days had only one TSO reporting; their "
            f"{orphans['gross_mcm_d'].sum():.0f} mcm/d gross is a coverage gap, not a "
            "netting failure, and is excluded from the net above"
        )
        log.warning(notes[-1])
    if not breaches.empty:
        notes.append(
            f"{len(breaches)} day(s) where internal flows failed to net to zero "
            f"beyond {perimeter.tolerances.internal_net_abs_mcm} mcm/d"
        )
        log.warning(notes[-1])

    return ReconciliationReport(
        days=len(days),
        internal_net=internal_net,
        internal_breaches=breaches,
        perimeter_flows=perim,
        coverage=coverage,
        orphans=orphans,
        unexplained_missing=unexplained,
        expected_missing=expected_missing,
        notes=notes,
    )
