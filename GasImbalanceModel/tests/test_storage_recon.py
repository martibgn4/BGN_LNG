"""Storage identity tests. SPEC 5 storage equation, SPEC 8 reconciliation.

Fixtures are hand-built and deliberately exact, so the arithmetic of the
identity is pinned independently of any vendor's data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from gasbalance.units import Units
from gasbalance.validation.storage_recon import reconcile_storage

ALERT = 20.0


@pytest.fixture(scope="module")
def units() -> Units:
    return Units()


def build(country: str, inventory_twh: list[float], inj_mcm: list[float], wdr_mcm: list[float]):
    """Long-format observations in internal units: TWh stock, mcm/d flows."""
    rows = []
    start = date(2025, 1, 1)
    for i, (stock, inj, wdr) in enumerate(zip(inventory_twh, inj_mcm, wdr_mcm, strict=True)):
        day = start + timedelta(days=i)
        for series_id, value in (
            ("gas_in_storage", stock),
            ("injection", inj),
            ("withdrawal", wdr),
            ("working_gas_volume", 100.0),
        ):
            rows.append(
                {"obs_date": day, "country": country, "series_id": series_id, "value": value}
            )
    return pd.DataFrame(rows)


def twh_per_mcm(units: Units) -> float:
    return units.canonical_gcv / units.energy_to_gwh(1.0, "TWh")


def test_a_perfectly_consistent_country_closes_exactly(units: Units) -> None:
    """S_t = S_(t-1) + inj - wdr, built to hold by construction."""
    step = twh_per_mcm(units)  # TWh moved by 1 mcm
    inventory = [50.0, 50.0 + 10 * step, 50.0 + 10 * step - 4 * step]
    obs = build("DE", inventory, [0.0, 10.0, 0.0], [0.0, 0.0, 4.0])

    report = reconcile_storage(obs, units, ALERT)
    assert report.daily["error_mcm_d"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert report.by_country["abs_mean_mcm_d"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_a_missing_withdrawal_shows_up_at_its_true_size(units: Units) -> None:
    """A 30 mcm/d withdrawal omitted from the flows must surface as 30 mcm/d."""
    step = twh_per_mcm(units)
    inventory = [50.0, 50.0 - 30 * step]
    obs = build("DE", inventory, [0.0, 0.0], [0.0, 0.0])  # withdrawal not reported

    report = reconcile_storage(obs, units, ALERT)
    error = report.daily["error_mcm_d"].iloc[0]
    assert error == pytest.approx(-30.0, rel=1e-9)
    assert abs(error) > ALERT  # loud enough to breach the SPEC 8 threshold


def test_error_is_reported_not_absorbed(units: Units) -> None:
    """SPEC 8: a gap must never be silently folded into another term."""
    step = twh_per_mcm(units)
    obs = build("DE", [50.0, 50.0 + 5 * step], [0.0, 0.0], [0.0, 0.0])
    report = reconcile_storage(obs, units, ALERT)

    assert "error_mcm_d" in report.daily.columns
    assert report.daily["error_mcm_d"].abs().sum() > 0
    # the reported inventory is untouched by the reconciliation
    assert report.daily["gas_in_storage"].iloc[0] == pytest.approx(50.0 + 5 * step)


def test_opposite_signed_adjacent_days_cancel_in_the_mean(units: Units) -> None:
    """A one-day alignment shift biases nothing, which is why bias is the test.

    This is the observed real-world pattern: a flow booked against the wrong
    gas day produces a large positive error and an equal negative one next to
    it. Mean error stays ~0 while worst-day error is large, so the mean is the
    honest measure of whether the data is broken.
    """
    step = twh_per_mcm(units)
    inventory = [50.0, 50.0 + 40 * step, 50.0]
    obs = build("DE", inventory, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    report = reconcile_storage(obs, units, ALERT)
    assert report.daily["error_mcm_d"].abs().max() > ALERT
    assert report.by_country["mean_error_mcm_d"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_countries_are_reconciled_independently(units: Units) -> None:
    step = twh_per_mcm(units)
    good = build("BE", [10.0, 10.0], [0.0, 0.0], [0.0, 0.0])
    bad = build("DE", [50.0, 50.0 - 25 * step], [0.0, 0.0], [0.0, 0.0])

    report = reconcile_storage(pd.concat([good, bad], ignore_index=True), units, ALERT)
    by_country = report.by_country.set_index("country")
    assert by_country.loc["BE", "abs_mean_mcm_d"] == pytest.approx(0.0, abs=1e-9)
    assert by_country.loc["DE", "abs_mean_mcm_d"] == pytest.approx(25.0, rel=1e-9)


def test_missing_series_raises_rather_than_assuming_zero(units: Units) -> None:
    obs = build("DE", [50.0, 50.0], [0.0, 0.0], [0.0, 0.0])
    obs = obs[obs["series_id"] != "withdrawal"]
    with pytest.raises(ValueError, match="withdrawal"):
        reconcile_storage(obs, units, ALERT)
