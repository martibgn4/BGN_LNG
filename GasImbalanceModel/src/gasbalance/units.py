"""Unit conversion layer. SPEC 5.

Two rules this module exists to enforce:

1. Convert at the point of ingestion. A raw-unit value must not propagate past
   a fetcher, so every fetcher calls into here and nothing else converts.
2. Every series declares its GCV/NCV basis. A series that does not declare one
   raises - it is never defaulted, because the GCV/NCV gap is ~10% and a silent
   default would put a 10% error straight into the balance.

No numeric constant appears in this file. Everything is read from
config/conversions.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gasbalance.config import load

Basis = Literal["GCV", "NCV"]

CANONICAL_BASIS: Basis = "GCV"
"""European gas volumetrics are quoted GCV (SPEC 5), so GCV is what we store."""


class UnitError(ValueError):
    """Base for every conversion failure. Fail loudly, never default."""


class MissingBasisError(UnitError):
    """A series did not declare GCV or NCV. SPEC 5 requires an explicit basis."""


class UnknownUnitError(UnitError):
    """A raw unit is not in the conversion table."""


class UnknownSeriesError(UnitError):
    """A (source, series_id) pair is not declared in conversions.yaml."""


@dataclass(frozen=True)
class SeriesSpec:
    """How one published series arrives: its raw unit and its energy basis."""

    source: str
    series_id: str
    raw_unit: str
    basis: Basis


class Units:
    """Reads conversions.yaml once and answers every conversion question."""

    def __init__(self, config_name: str = "conversions.yaml") -> None:
        self._cfg = load(config_name)
        self._energy = self._cfg["energy_scale_to_gwh"]
        self._volume = self._cfg["volume_scale_to_mcm"]

    # -- calorific values ---------------------------------------------------

    @property
    def canonical_gcv(self) -> float:
        """kWh per m3, GCV. The one number bcm<->TWh and mcm<->GWh both use."""
        return float(self._cfg["canonical_gcv_kwh_per_m3"]["value"])

    @property
    def gcv_to_ncv_ratio(self) -> float:
        """GCV / NCV for pipeline-spec gas."""
        return float(self._cfg["gcv_to_ncv_ratio"]["value"])

    def gcv_for_source(self, source: str | None) -> float:
        """Per-origin calorific value. SPEC 5: a Norwegian mcm is not a Qatari mcm."""
        if source is None:
            return self.canonical_gcv
        table = self._cfg["source_gcv_kwh_per_m3"]
        if source not in table:
            raise UnknownSeriesError(
                f"no calorific value declared for source {source!r}; "
                "add it to conversions.yaml rather than defaulting"
            )
        entry = table[source]
        if entry.get("basis") != CANONICAL_BASIS:
            raise MissingBasisError(
                f"source {source!r} calorific value must be declared on "
                f"{CANONICAL_BASIS}, got {entry.get('basis')!r}"
            )
        return float(entry["value"])

    # -- series declarations ------------------------------------------------

    def series_spec(self, source: str, series_id: str) -> SeriesSpec:
        """Look up how a published series arrives. Absent means a hard error."""
        sources = self._cfg["source_units"]
        if source not in sources:
            raise UnknownSeriesError(f"source {source!r} not declared in conversions.yaml")
        if series_id not in sources[source]:
            raise UnknownSeriesError(
                f"series {source}.{series_id} not declared in conversions.yaml; "
                "declare its raw_unit and basis before ingesting it"
            )
        entry = sources[source][series_id]
        basis = entry.get("basis")
        if basis not in ("GCV", "NCV"):
            raise MissingBasisError(
                f"series {source}.{series_id} has basis {basis!r}; "
                "SPEC 5 requires an explicit GCV or NCV declaration"
            )
        raw_unit = entry.get("raw_unit")
        if not raw_unit:
            raise UnknownUnitError(f"series {source}.{series_id} has no raw_unit")
        return SeriesSpec(source, series_id, raw_unit, basis)

    def all_series_specs(self) -> list[SeriesSpec]:
        """Every declared series, so a test can assert none lacks a basis."""
        out: list[SeriesSpec] = []
        for source, series in self._cfg["source_units"].items():
            for series_id in series:
                out.append(self.series_spec(source, series_id))
        return out

    # -- basis crossover ----------------------------------------------------

    def to_canonical_basis(self, energy: float, basis: Basis) -> float:
        """Move an energy quantity onto GCV. SPEC 5 calls this a ~10% gap."""
        if basis == CANONICAL_BASIS:
            return energy
        if basis == "NCV":
            return energy * self.gcv_to_ncv_ratio
        raise MissingBasisError(f"unknown basis {basis!r}")

    def efficiency_to_canonical_basis(self, efficiency: float, basis: Basis) -> float:
        """Move a thermal efficiency onto GCV.

        Efficiencies move the opposite way to energies: the same plant scores
        lower on GCV because GCV counts more input energy for the same gas.
        Power-sector efficiencies are normally quoted NCV (SPEC 2b), and
        dividing an NCV efficiency into a GCV energy density is exactly the
        ~10% error SPEC 5 warns about.
        """
        if basis == CANONICAL_BASIS:
            return efficiency
        if basis == "NCV":
            return efficiency / self.gcv_to_ncv_ratio
        raise MissingBasisError(f"unknown basis {basis!r}")

    # -- scalar conversions -------------------------------------------------

    def energy_to_gwh(self, value: float, unit: str) -> float:
        if unit not in self._energy:
            raise UnknownUnitError(f"energy unit {unit!r} not in conversions.yaml")
        return value * float(self._energy[unit])

    def gwh_to_energy(self, value_gwh: float, unit: str) -> float:
        if unit not in self._energy:
            raise UnknownUnitError(f"energy unit {unit!r} not in conversions.yaml")
        return value_gwh / float(self._energy[unit])

    def volume_to_mcm(self, value: float, unit: str) -> float:
        if unit not in self._volume:
            raise UnknownUnitError(f"volume unit {unit!r} not in conversions.yaml")
        return value * float(self._volume[unit])

    def mcm_to_volume(self, value_mcm: float, unit: str) -> float:
        if unit not in self._volume:
            raise UnknownUnitError(f"volume unit {unit!r} not in conversions.yaml")
        return value_mcm / float(self._volume[unit])

    # -- energy <-> volume --------------------------------------------------

    def gwh_to_mcm(self, value_gwh: float, source: str | None = None) -> float:
        """1 mcm at CV c kWh/m3 carries exactly c GWh, so mcm = GWh / c."""
        return value_gwh / self.gcv_for_source(source)

    def mcm_to_gwh(self, value_mcm: float, source: str | None = None) -> float:
        return value_mcm * self.gcv_for_source(source)

    # -- the two functions fetchers actually call ---------------------------

    def to_mcm_per_day(
        self, value: float, source: str, series_id: str, origin: str | None = None
    ) -> float:
        """Convert one published daily value to the internal flow unit, mcm/d.

        `origin` names the physical gas source used for the calorific value,
        and is deliberately distinct from `source`, which names the publisher.
        """
        spec = self.series_spec(source, series_id)
        unit = spec.raw_unit.split("/")[0]
        if unit in self._volume:
            return self.volume_to_mcm(value, unit)
        gwh = self.to_canonical_basis(self.energy_to_gwh(value, unit), spec.basis)
        return self.gwh_to_mcm(gwh, origin)

    def to_twh(
        self, value: float, source: str, series_id: str, origin: str | None = None
    ) -> float:
        """Convert one published stock value to the internal energy unit, TWh."""
        spec = self.series_spec(source, series_id)
        unit = spec.raw_unit.split("/")[0]
        if unit in self._volume:
            mcm = self.volume_to_mcm(value, unit)
            return self.gwh_to_energy(self.mcm_to_gwh(mcm, origin), "TWh")
        gwh = self.to_canonical_basis(self.energy_to_gwh(value, unit), spec.basis)
        return self.gwh_to_energy(gwh, "TWh")


def default_units() -> Units:
    return Units()
