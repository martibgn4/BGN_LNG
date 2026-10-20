"""Energy and volume units.

Definitional constants (exact by definition) live here. Gas-quality-dependent factors (volume to
energy) are **not** constants: they depend on the calorific value of the gas and the reference
conditions of the source, so they are passed in explicitly or read from configuration.
See GasImbalanceModel PHASE1_NOTES for the 10.55 vs 11 TWh/bcm inconsistency that motivates this.
"""

from __future__ import annotations

from dataclasses import dataclass

# Exact: 1 MMBtu = 1,055,055.85262 kJ (IT Btu) -> 1 MWh = 3.6e6 kJ / 1,055,055.85262 kJ
MMBTU_PER_MWH: float = 3.6e6 / 1_055_055.85262
MWH_PER_MMBTU: float = 1.0 / MMBTU_PER_MWH

# Pure energy-scale prefixes, in MWh.
_ENERGY_IN_MWH: dict[str, float] = {
    "MWh": 1.0,
    "GWh": 1e3,
    "TWh": 1e6,
    "kWh": 1e-3,
    "MMBtu": MWH_PER_MMBTU,
    "therm": MWH_PER_MMBTU / 10.0,  # 1 therm = 100,000 Btu = 0.1 MMBtu
}

# Energy-rate units -> (energy unit, per-day).
_RATE_UNITS: dict[str, str] = {"MWh/d": "MWh", "GWh/d": "GWh", "TWh/d": "TWh", "MMBtu/d": "MMBtu"}


class UnitError(ValueError):
    """Unknown unit or an incompatible conversion (e.g. energy to volume without a GCV)."""


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between pure energy units (MWh, GWh, TWh, kWh, MMBtu, therm)."""
    try:
        return value * _ENERGY_IN_MWH[from_unit] / _ENERGY_IN_MWH[to_unit]
    except KeyError as exc:
        raise UnitError(f"Unknown energy unit in {from_unit!r} -> {to_unit!r}") from exc


def convert_energy_rate(value: float, from_unit: str, to_unit: str) -> float:
    """Convert per-day energy rates (MWh/d, GWh/d, TWh/d, MMBtu/d)."""
    try:
        return convert_energy(value, _RATE_UNITS[from_unit], _RATE_UNITS[to_unit])
    except KeyError as exc:
        raise UnitError(f"Unknown rate unit in {from_unit!r} -> {to_unit!r}") from exc


def eur_mwh_to_usd_mmbtu(price_eur_mwh: float, eurusd: float) -> float:
    """EUR/MWh -> USD/MMBtu. ``eurusd`` is USD per EUR at the **same** session (PIT policy §4)."""
    return price_eur_mwh * eurusd / MMBTU_PER_MWH


def usd_mmbtu_to_eur_mwh(price_usd_mmbtu: float, eurusd: float) -> float:
    """USD/MMBtu -> EUR/MWh."""
    return price_usd_mmbtu * MMBTU_PER_MWH / eurusd


@dataclass(frozen=True)
class GasQuality:
    """Volume-to-energy basis. ``gcv_kwh_per_m3`` must be stated by the source or configuration.

    Attributes:
        gcv_kwh_per_m3: gross calorific value, kWh per cubic metre at ``reference``.
        reference: reference conditions label, e.g. ``"25/0 C"`` or ``"15/15 C"``.
    """

    gcv_kwh_per_m3: float
    reference: str

    def mcm_to_gwh(self, mcm: float) -> float:
        """Million cubic metres -> GWh (GCV basis)."""
        return mcm * 1e6 * self.gcv_kwh_per_m3 / 1e6

    def gwh_to_mcm(self, gwh: float) -> float:
        """GWh (GCV basis) -> million cubic metres."""
        return gwh * 1e6 / (self.gcv_kwh_per_m3 * 1e6)
