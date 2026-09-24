"""Unit conversion tests. SPEC 5 and SPEC 8 (golden-file tests on conversions)."""

from __future__ import annotations

import pytest

from gasbalance.units import (
    CANONICAL_BASIS,
    MissingBasisError,
    UnknownSeriesError,
    UnknownUnitError,
    Units,
)


@pytest.fixture(scope="module")
def units() -> Units:
    return Units()


# -- golden values ----------------------------------------------------------


def test_one_bcm_is_the_spec_reference_in_twh(units: Units) -> None:
    """SPEC 5: 1 bcm ~= 10.55 TWh (GCV)."""
    twh = units.gwh_to_energy(units.mcm_to_gwh(units.volume_to_mcm(1.0, "bcm")), "TWh")
    assert twh == pytest.approx(10.55, abs=1e-9)


def test_bcm_and_mcm_conversions_are_mutually_consistent(units: Units) -> None:
    """The failure this guards against is SPEC 5 quoting two different CVs.

    SPEC 5 gives both "1 bcm ~= 10.55 TWh" and "1 mcm/d ~= 11 GWh/d". 1 bcm is
    exactly 1000 mcm, so those imply 10.55 and 11.00 kWh/m3 respectively - a
    4.3% gap. On a 1500 mcm/d balance that is ~65 mcm/d of pure unit error,
    three times the 20 mcm/d gap SPEC 8 says must raise an alert.

    This test asserts the model uses ONE calorific value everywhere. It does
    not assert which one - that is a trader decision recorded in
    conversions.yaml.
    """
    gwh_per_mcm = units.mcm_to_gwh(1.0)
    twh_per_bcm = units.gwh_to_energy(units.mcm_to_gwh(units.volume_to_mcm(1.0, "bcm")), "TWh")
    assert gwh_per_mcm == pytest.approx(twh_per_bcm, rel=1e-12)


@pytest.mark.parametrize("volume_mcm", [0.0, 1.0, 137.5, 1500.0])
def test_volume_energy_roundtrip(units: Units, volume_mcm: float) -> None:
    back = units.gwh_to_mcm(units.mcm_to_gwh(volume_mcm))
    assert back == pytest.approx(volume_mcm, rel=1e-12)


@pytest.mark.parametrize("unit", ["kWh", "MWh", "GWh", "TWh"])
def test_energy_scale_roundtrip(units: Units, unit: str) -> None:
    assert units.gwh_to_energy(units.energy_to_gwh(3.0, unit), unit) == pytest.approx(3.0)


# -- GCV / NCV crossover ----------------------------------------------------


def test_ncv_energy_is_scaled_up_to_gcv(units: Units) -> None:
    """GCV counts more energy for the same gas, so GCV > NCV."""
    gcv = units.to_canonical_basis(100.0, "NCV")
    assert gcv > 100.0
    assert gcv == pytest.approx(100.0 * units.gcv_to_ncv_ratio)


def test_gcv_energy_is_untouched(units: Units) -> None:
    assert units.to_canonical_basis(100.0, CANONICAL_BASIS) == 100.0


def test_efficiency_moves_the_opposite_way_to_energy(units: Units) -> None:
    """A plant scores LOWER on GCV than on NCV.

    This is the trap in SPEC 2b: its worked example divides a 50% efficiency
    (an NCV figure in normal usage) into a GCV energy density. Doing that
    understates gas burn by the full ~10% GCV/NCV gap.
    """
    eff_gcv = units.efficiency_to_canonical_basis(0.50, "NCV")
    assert eff_gcv < 0.50
    assert eff_gcv == pytest.approx(0.50 / units.gcv_to_ncv_ratio)
    assert units.gcv_to_ncv_ratio == pytest.approx(1.1, abs=0.05)  # SPEC 5: ~10%


def test_basis_gap_is_material(units: Units) -> None:
    """Confirm the crossover is worth enforcing rather than approximating."""
    assert abs(units.to_canonical_basis(1.0, "NCV") - 1.0) > 0.05


# -- fail loudly, never default ---------------------------------------------


def test_every_declared_series_has_a_basis(units: Units) -> None:
    """SPEC 5: "Add a test that fails if a series lacks a declared basis"."""
    specs = units.all_series_specs()
    assert specs, "conversions.yaml declares no series at all"
    for spec in specs:
        assert spec.basis in ("GCV", "NCV"), f"{spec.source}.{spec.series_id} has no basis"
        assert spec.raw_unit, f"{spec.source}.{spec.series_id} has no raw unit"


def test_undeclared_series_raises_rather_than_defaulting(units: Units) -> None:
    with pytest.raises(UnknownSeriesError):
        units.series_spec("entsog", "a_series_nobody_declared")
    with pytest.raises(UnknownSeriesError):
        units.series_spec("some_new_vendor", "physical_flow")


def test_unknown_unit_raises(units: Units) -> None:
    with pytest.raises(UnknownUnitError):
        units.energy_to_gwh(1.0, "therms")
    with pytest.raises(UnknownUnitError):
        units.volume_to_mcm(1.0, "cubic_feet")


def test_unknown_basis_raises(units: Units) -> None:
    with pytest.raises(MissingBasisError):
        units.to_canonical_basis(1.0, "HHV")  # type: ignore[arg-type]


def test_unknown_origin_raises_rather_than_using_default_cv(units: Units) -> None:
    with pytest.raises(UnknownSeriesError):
        units.gcv_for_source("mars")


# -- per-source calorific values --------------------------------------------


def test_source_calorific_values_differ(units: Units) -> None:
    """SPEC 5: a Norwegian mcm is not an Algerian mcm is not a Qatari mcm."""
    norway = units.gcv_for_source("norway")
    qatar = units.gcv_for_source("qatar_lng")
    lgas = units.gcv_for_source("nl_lgas")
    assert qatar > norway > lgas
    assert (qatar - lgas) / lgas > 0.10  # L-gas vs rich LNG is a big gap


def test_origin_changes_the_volume_for_the_same_energy(units: Units) -> None:
    rich = units.gwh_to_mcm(1000.0, "qatar_lng")
    lean = units.gwh_to_mcm(1000.0, "nl_lgas")
    assert lean > rich  # lower CV means more cubic metres for the same energy


# -- end-to-end fetcher-facing conversions ----------------------------------


def test_entsog_kwh_per_day_becomes_mcm_per_day(units: Units) -> None:
    """One day at 1 TWh/d in ENTSOG units."""
    mcm = units.to_mcm_per_day(1e9, "entsog", "physical_flow")
    assert mcm == pytest.approx(1e9 * 1e-6 / units.canonical_gcv)


def test_agsi_stock_becomes_twh(units: Units) -> None:
    assert units.to_twh(42.0, "agsi", "gas_in_storage") == pytest.approx(42.0)


def test_agsi_flow_becomes_mcm_per_day(units: Units) -> None:
    """AGSI publishes injection/withdrawal in GWh/d."""
    mcm = units.to_mcm_per_day(1055.0, "agsi", "withdrawal")
    assert mcm == pytest.approx(100.0, rel=1e-9)
