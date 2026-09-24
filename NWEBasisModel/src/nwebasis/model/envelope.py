"""The no-arbitrage envelope: how wide the discount can economically get.

Two bounds, both economic rather than statistical:

  **Cost floor.** Selling a cargo DES NWE instead of selling the same energy at
  TTF costs a regas slot, a tariff, boil-off, port dues and working capital.
  That stack is the natural width of the discount when regas capacity is freely
  available - the basis has no reason to sit below it, because at that point
  the buyer is being paid more than the cost of the alternative.

  **Diversion cap.** The discount cannot narrow past the point where the
  marginal US cargo prefers Asia. That level is the east-west arb, which Spark
  publishes directly as ``neaMinusNwe.ttfBasis``.

WHAT THE ENVELOPE IS FOR. Not as a hard clamp on the forecast - the basis broke
the cost floor decisively in November 2022, reaching -10.52 when the binding
constraint stopped being cost and became capacity. It is a REGIME CLASSIFIER.
Inside the envelope, cost anchors the basis and a point forecast is meaningful.
Outside it, the basis is the shadow price of a scarce slot, mean reversion is
to a different level, and the model should widen its interval rather than
pretend to precision it does not have.

This module raises rather than defaulting when the cost stack is unset. A
guessed tariff would move the anchor by exactly its own error, silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nwebasis.config import get


class RegasCostUnset(RuntimeError):
    """The regas cost stack has not been filled in, so no cost floor exists."""


@dataclass(frozen=True)
class CostStack:
    """The all-in cost of converting a DES NWE cargo into hub gas."""

    capacity_weighted_tariff: float
    adders: dict[str, float]

    @property
    def total(self) -> float:
        return self.capacity_weighted_tariff + sum(self.adders.values())

    @property
    def implied_basis_floor(self) -> float:
        """As a basis: cost is a discount, so the floor is negative."""
        return -self.total


def _missing_entries() -> list[str]:
    """Every unset number in regas.yaml, named so they can be filled in one pass."""
    missing = []
    for key, terminal in get("regas", "terminals").items():
        if terminal.get("tariff") is None:
            missing.append(f"terminals.{key}.tariff")
        if terminal.get("weight") is None:
            missing.append(f"terminals.{key}.weight")
    for key, adder in get("regas", "adders").items():
        if adder.get("value") is None:
            missing.append(f"adders.{key}")
    return missing


def cost_stack() -> CostStack:
    """Build the cost floor, or explain precisely what is missing."""
    missing = _missing_entries()
    if missing:
        raise RegasCostUnset(
            "regas.yaml is incomplete, so no cost anchor can be built. "
            f"{len(missing)} entries unset:\n  " + "\n  ".join(missing) +
            "\n\nThe model falls back to anchor mode 'empirical', which "
            "estimates the anchor from the basis history instead. That works, "
            "but it cannot extrapolate to a cost structure it has never seen - "
            "a tariff change would not move the forecast."
        )
    terminals = get("regas", "terminals").values()
    weights = np.array([t["weight"] for t in terminals], dtype=float)
    tariffs = np.array([t["tariff"] for t in terminals], dtype=float)
    weighted = float((weights * tariffs).sum() / weights.sum())
    adders = {key: float(adder["value"]) for key, adder in get("regas", "adders").items()}
    return CostStack(capacity_weighted_tariff=weighted, adders=adders)


def diversion_cap(arb: pd.Series) -> pd.Series:
    """Narrowest justified basis, from the published east-west arb.

    ``neaMinusNwe`` is positive when Asia's netback beats Europe's. At that
    point Europe has to richen by at least the gap to hold the cargo, so the
    arb maps to the cap on how narrow the NWE basis can sit.
    """
    return arb.astype(float)


def classify(free_slots: pd.Series) -> pd.Series:
    """Label each observation ``scarce`` or ``ample`` on regas availability.

    Scarcity is defined on the slot distribution and NOT on the basis itself.
    Defining it on the basis would be circular: the model would then "predict"
    a regime that had been read off the thing being predicted.
    """
    percentile = float(get("model", "regimes", "scarcity_percentile"))
    threshold = free_slots.quantile(percentile)
    labels = pd.Series(np.where(free_slots <= threshold, "scarce", "ample"),
                       index=free_slots.index, name="regime")

    # Not an error - just a sample too short to say anything separate about the
    # rare regime. Flagged so a caller does not read a two-regime table that is
    # really one regime plus noise.
    minimum = int(get("model", "regimes", "min_days_in_regime"))
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < minimum:
        labels.attrs["warning"] = (
            f"a regime has fewer than {minimum} days; treat the split as "
            "descriptive, not as a fitted state"
        )
    return labels
