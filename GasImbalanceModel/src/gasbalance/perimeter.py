"""Perimeter definition and netting. SPEC 7 Phase 1.

The rule this module enforces: only flows crossing the ring count.

Norwegian gas landing at Easington and moving on through the IUK to Zeebrugge
is one import (NO -> UK, crossing) and one internal transfer (UK -> BE), and
the internal transfer must never appear in the balance. With the NW Europe ring
that falls out automatically - both IUK and BBL classify as internal, because
both of their countries are inside.

Classification is per (point, operator, direction), not per point, because the
two sides of an interconnection point are reported by two different TSOs and
can disagree. Netting those two sides against each other is the test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from gasbalance.config import load

log = logging.getLogger(__name__)

ROLE_CROSSING = "crossing"
ROLE_INTERNAL = "internal"
ROLE_STORAGE = "storage"
ROLE_LNG = "lng_entry"
ROLE_PRODUCTION = "production"
ROLE_OUTSIDE = "outside"
ROLE_INTERNAL_NODE = "internal_node"

#: Roles whose flows enter the balance identity as perimeter terms.
BALANCE_ROLES = frozenset({ROLE_CROSSING, ROLE_LNG, ROLE_PRODUCTION})


class PerimeterError(RuntimeError):
    """Raised when a point cannot be classified. Never guessed around."""


@dataclass(frozen=True)
class Tolerances:
    point_daily_abs_mcm: float
    point_daily_rel: float
    internal_net_abs_mcm: float
    perimeter_daily_rel: float
    alert_systematic_gap_mcm_per_d: float


class Perimeter:
    """The ring, loaded from perimeter.yaml."""

    def __init__(self, config_name: str = "perimeter.yaml") -> None:
        cfg = load(config_name)
        self.name: str = cfg["name"]
        self.inside: set[str] = set(cfg["inside"])
        self.outside: set[str] = set(cfg["outside"])
        self.exclusions: set[str] = {e["point_key"] for e in (cfg.get("exclusions") or [])}
        self._types: dict[str, list[str]] = cfg["perimeter_point_types"]
        self._internal_types: set[str] = set(cfg["internal_point_types"])
        self.tolerances = Tolerances(**cfg["tolerances"])

        self._validate_country_lists(cfg)

        overlap = self.inside & self.outside
        if overlap:
            raise PerimeterError(f"countries in both inside and outside: {sorted(overlap)}")

    @staticmethod
    def _validate_country_lists(cfg: dict[str, Any]) -> None:
        """Guard against YAML 1.1 coercing bare country codes.

        This is not hypothetical. An unquoted `NO` in a YAML list parses as the
        boolean False, not the string "NO", so Norway - the largest single
        supply dial in the whole model - silently drops out of the perimeter
        and every Norwegian import gets misclassified. `ON`, `OFF`, `YES` and
        `N` coerce the same way. Country codes must be quoted, and anything
        that is not a two-letter string fails loudly here.
        """
        for key in ("inside", "outside"):
            for entry in cfg[key]:
                if not isinstance(entry, str) or len(entry) != 2 or not entry.isalpha():
                    raise PerimeterError(
                        f"perimeter.yaml {key} contains {entry!r} ({type(entry).__name__}), "
                        "not a two-letter country code. Quote every code: YAML reads "
                        "a bare NO as boolean False."
                    )

    # -- classification -----------------------------------------------------

    def known_country(self, country: str | None) -> bool:
        return country in self.inside or country in self.outside

    def _role_for(self, point_type: str, tso_country: str, adjacent_country: str | None) -> str:
        if tso_country not in self.inside:
            return ROLE_OUTSIDE
        if point_type in self._types["crossing"]:
            if adjacent_country in self.inside:
                return ROLE_INTERNAL
            return ROLE_CROSSING
        if point_type in self._types["lng_entry"]:
            return ROLE_LNG
        if point_type in self._types["production"]:
            return ROLE_PRODUCTION
        if point_type in self._types["storage"]:
            return ROLE_STORAGE
        if point_type in self._internal_types:
            return ROLE_INTERNAL_NODE
        raise PerimeterError(
            f"point type {point_type!r} is in neither perimeter_point_types nor "
            "internal_point_types in perimeter.yaml. Classify it deliberately - "
            "an unclassified type would silently drop out of the balance."
        )

    def classify(self, registry: pd.DataFrame) -> pd.DataFrame:
        """Attach a ring_role to every ENTSOG point-direction.

        Raises on any country key that perimeter.yaml does not mention, so that
        a new ENTSOG country cannot silently be treated as outside the ring.
        """
        frame = registry.copy()
        seen = set(frame["tSOCountry"].dropna().unique()) | set(
            frame["adjacentCountry"].dropna().unique()
        )
        unknown = {c for c in seen if c and not self.known_country(c)}
        if unknown:
            raise PerimeterError(
                f"ENTSOG reports countries not listed in perimeter.yaml: "
                f"{sorted(unknown)}. Add each to inside or outside explicitly."
            )

        frame["ring_role"] = [
            self._role_for(pt, tso, adj)
            for pt, tso, adj in zip(
                frame["pointType"],
                frame["tSOCountry"],
                frame["adjacentCountry"],
                strict=True,
            )
        ]
        excluded = frame["pointKey"].isin(self.exclusions)
        if excluded.any():
            log.warning("excluding %d point-directions by config", int(excluded.sum()))
            frame.loc[excluded, "ring_role"] = ROLE_OUTSIDE
        return frame

    def to_points_table(self, classified: pd.DataFrame) -> pd.DataFrame:
        """Reshape the classified registry into the `points` DB table layout."""
        gcv_min = pd.to_numeric(classified.get("tpTsoGCVMin"), errors="coerce")
        gcv_max = pd.to_numeric(classified.get("tpTsoGCVMax"), errors="coerce")
        return pd.DataFrame(
            {
                "point_key": classified["pointKey"],
                "point_label": classified["pointLabel"],
                "operator_key": classified["operatorKey"],
                "operator_label": classified["operatorLabel"],
                "direction_key": classified["directionKey"],
                "point_type": classified["pointType"],
                "cross_border_type": classified["crossBorderPointType"],
                "tso_country": classified["tSOCountry"],
                "adjacent_country": classified["adjacentCountry"],
                "ring_role": classified["ring_role"],
                "has_data": classified["hasData"].astype("boolean"),
                "is_double_reporting": classified["isDoubleReporting"].astype("boolean"),
                "is_pipe_in_pipe": classified["isPipeInPipe"].astype("boolean"),
                "gcv_min": gcv_min,
                "gcv_max": gcv_max,
                "gcv_unit": classified.get("tpTsoGCVUnit"),
            }
        )

    # -- netting ------------------------------------------------------------

    def signed_flows(self, flows: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
        """Join flows to roles and sign them: entry positive, exit negative."""
        keys = ["point_key", "operator_key", "direction_key"]
        merged = flows.merge(
            points[[*keys, "ring_role", "tso_country", "adjacent_country"]],
            on=keys,
            how="left",
            validate="many_to_one",
        )
        unmatched = merged["ring_role"].isna()
        if unmatched.any():
            missing = sorted(merged.loc[unmatched, "point_key"].unique())[:10]
            raise PerimeterError(
                f"{int(unmatched.sum())} flow rows have no entry in the point "
                f"registry (e.g. {missing}). Re-run `gasbalance resolve-points`; "
                "do not drop them."
            )
        sign = merged["direction_key"].map({"entry": 1.0, "exit": -1.0})
        if sign.isna().any():
            bad = sorted(merged.loc[sign.isna(), "direction_key"].unique())
            raise PerimeterError(f"unknown direction keys {bad}")
        merged["signed_mcm_d"] = merged["value"] * sign
        return merged

    def internal_net(self, signed: pd.DataFrame) -> pd.DataFrame:
        """Net internal flows per day. SPEC 7: this must sum to zero.

        Both sides of an internal interconnection point are reported, one as an
        exit by the upstream TSO and one as an entry by the downstream TSO. Sum
        them with sign and the pair cancels. What is left is the disagreement
        between the two TSOs, which is a data-quality measure, not a gas flow.
        """
        internal = signed[signed["ring_role"] == ROLE_INTERNAL]
        return (
            internal.groupby("obs_date", as_index=False)["signed_mcm_d"]
            .sum()
            .rename(columns={"signed_mcm_d": "internal_net_mcm_d"})
        )

    def perimeter_flows(self, signed: pd.DataFrame) -> pd.DataFrame:
        """Daily net flow across the ring, split by role."""
        crossing = signed[signed["ring_role"].isin(BALANCE_ROLES)]
        return (
            crossing.groupby(["obs_date", "ring_role"], as_index=False)["signed_mcm_d"]
            .sum()
            .rename(columns={"signed_mcm_d": "net_mcm_d"})
        )

    def internal_pairs(self, signed: pd.DataFrame) -> pd.DataFrame:
        """Split internal points into matched pairs and orphan sides.

        This split matters. An internal interconnection point only cancels if
        BOTH reporting TSOs published for that gas day. When one side is
        missing, the surviving side looks exactly like a netting failure, and
        reporting it as one would blame the model for a hole in the data.

        Returns one row per (day, point) with the number of reporting operators
        and the signed sum, so matched and orphan cases can be judged apart.
        """
        internal = signed[signed["ring_role"] == ROLE_INTERNAL]
        grouped = internal.groupby(["obs_date", "point_key"], as_index=False).agg(
            operators=("operator_key", "nunique"),
            net_mcm_d=("signed_mcm_d", "sum"),
            gross_mcm_d=("signed_mcm_d", lambda x: x.abs().sum()),
        )
        grouped["matched"] = grouped["operators"] > 1
        return grouped

    def internal_net_matched(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Daily net across only those internal points that reported both sides."""
        matched = pairs[pairs["matched"]]
        return (
            matched.groupby("obs_date", as_index=False)["net_mcm_d"]
            .sum()
            .rename(columns={"net_mcm_d": "internal_net_mcm_d"})
        )

    def internal_orphans(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Internal points where only one TSO reported. A data gap, not a flow."""
        return pairs[~pairs["matched"]].copy()

    def internal_net_breaches(self, internal_net: pd.DataFrame) -> pd.DataFrame:
        """Days where internal flows failed to net to zero within tolerance."""
        tol = self.tolerances.internal_net_abs_mcm
        breaches = internal_net[internal_net["internal_net_mcm_d"].abs() > tol].copy()
        breaches["tolerance_mcm_d"] = tol
        return breaches


def summarise_roles(classified: pd.DataFrame) -> dict[str, Any]:
    """Counts by role, for the Phase 1 report."""
    counts = classified["ring_role"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
