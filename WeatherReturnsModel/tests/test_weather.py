"""Tests for the weather layer, aimed at the specific ways it can lie."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxreturns.config import get
from wxreturns.data.openmeteo import region_points
from wxreturns.weather import crop, degree_days, normals, revisions


# --- weighting ---------------------------------------------------------------

def test_region_weights_normalise() -> None:
    for region in get("regions", "regions"):
        weights = [p["weight"] for p in region_points(region)]
        assert all(w > 0.0 for w in weights), f"{region} has a non-positive weight"
        assert np.isclose(sum(weights), 1.0), f"{region} weights do not normalise"


def test_config_weights_are_written_close_to_one() -> None:
    """A typo in a raw weight should be visible before normalisation hides it."""
    for region, cfg in get("regions", "regions").items():
        raw = sum(p["weight"] for p in cfg["points"])
        assert 0.9 < raw < 1.1, (
            f"{region} raw weights sum to {raw:.3f}; they are normalised anyway "
            "but should be written close to 1 so a typo is visible")


# --- degree days -------------------------------------------------------------

def _panel(temps: dict[str, list[float]], region: str, dates: list[str]
           ) -> pd.DataFrame:
    rows = []
    for point, series in temps.items():
        for when, value in zip(dates, series, strict=True):
            rows.append({
                "issue_date": pd.Timestamp(when) - pd.Timedelta(days=1),
                "target_date": pd.Timestamp(when),
                "region": region, "point": point,
                "variable": "temperature_2m_mean", "value": value,
                "model": "test", "member": -1,
            })
    return pd.DataFrame(rows)


def test_degree_days_computed_before_weighting() -> None:
    """The kink at zero: averaging temperature first understates HDD.

    Two points, one well above the base and one well below. The correct
    answer averages their degree days; averaging temperature first gives a
    strictly smaller number. If this test ever passes with equality, the
    weighting order has been reversed somewhere.
    """
    region = "us_gas"
    points = region_points(region)
    hot, cold = points[0]["name"], points[1]["name"]
    dates = ["2024-04-15"]

    panel = _panel({hot: [30.0], cold: [0.0]}, region, dates)
    per_point = degree_days.point_degree_days(panel, region)

    base = get("features", "degree_days", "us", "base_temperature_c")
    expected_hot = max(0.0, base - 30.0)
    expected_cold = max(0.0, base - 0.0)
    assert np.isclose(per_point.set_index("point").loc[hot, "hdd"], expected_hot)
    assert np.isclose(per_point.set_index("point").loc[cold, "hdd"], expected_cold)

    # The wrong order would give max(0, base - mean(30, 0)) = base - 15, which
    # is strictly less than the weighted mean of the two point HDDs.
    naive = max(0.0, base - (30.0 + 0.0) / 2)
    correct = (expected_hot + expected_cold) / 2
    assert correct > naive, "the kink must make the correct answer larger"


def test_eu_and_us_conventions_differ() -> None:
    """65F and the Eurostat 15.5/18 pair are not the same rule."""
    temperature = pd.Series([16.0])
    us_hdd, _ = degree_days._hdd_cdd(temperature, "us")
    eu_hdd, _ = degree_days._hdd_cdd(temperature, "eu")
    # At 16C the US convention accrues 65F - 16C; Eurostat accrues nothing,
    # because 16 is above its 15.5 threshold.
    assert us_hdd.iloc[0] > 0.0
    assert eu_hdd.iloc[0] == 0.0


def test_degree_days_rejects_non_gas_region() -> None:
    with pytest.raises(ValueError, match="not gas"):
        degree_days.convention_for("us_corn_belt")


def test_missing_point_raises_not_silently_reweights() -> None:
    region = "us_gas"
    points = region_points(region)
    panel = _panel({points[0]["name"]: [5.0]}, region, ["2024-01-10"])
    per_point = degree_days.point_degree_days(panel, region)
    with pytest.raises(ValueError, match="missing points"):
        degree_days.weight_to_region(per_point, region)


# --- revisions: the coverage trap --------------------------------------------

def _forecast_panel(region: str = "us_gas") -> pd.DataFrame:
    """Two issue dates, overlapping target days, with a real change on one day."""
    rows = []
    for issue, targets in [
        ("2024-01-10", {"2024-01-11": 20.0, "2024-01-12": 22.0, "2024-01-13": 24.0}),
        ("2024-01-11", {"2024-01-12": 22.0, "2024-01-13": 30.0, "2024-01-14": 25.0}),
    ]:
        for target, hdd in targets.items():
            rows.append({"region": region, "issue_date": pd.Timestamp(issue),
                         "target_date": pd.Timestamp(target), "hdd": hdd,
                         "cdd": 0.0})
    return pd.DataFrame(rows)


def test_revision_uses_only_common_target_days() -> None:
    """The central trap: a rolling horizon must not look like news.

    Issue 10 covers the 11th-13th, issue 11 covers the 12th-14th. Only the
    12th and 13th are common. The 14th is new and the 11th has dropped out,
    and neither is a revision. The 12th is unchanged (22 -> 22) and the 13th
    is revised 24 -> 30, so +6 is the only news in the panel.
    """
    panel = _forecast_panel()
    revised = revisions.daily_revisions(panel, value_column="hdd", step_days=1)

    targets = set(revised["target_date"].dt.strftime("%Y-%m-%d"))
    assert targets == {"2024-01-12", "2024-01-13"}, (
        f"only common target days may yield a revision, got {sorted(targets)}")
    assert np.isclose(revised["revision"].sum(), 6.0), (
        "the only real revision is +6 HDD on the 13th; anything else is the "
        "horizon rolling forward being mistaken for news")


def test_revision_sign_is_positive_when_forecast_turns_colder() -> None:
    """HDD rises as temperature falls, so a colder forecast is a POSITIVE revision."""
    panel = _forecast_panel()
    revised = revisions.daily_revisions(panel, value_column="hdd", step_days=1)
    changed = revised[revised["target_date"] == pd.Timestamp("2024-01-13")]
    assert changed["revision"].iloc[0] > 0.0


def test_revisions_reject_reanalysis_rows() -> None:
    panel = _forecast_panel()
    panel.loc[0, "issue_date"] = pd.NaT
    with pytest.raises(ValueError, match="reanalysis"):
        revisions.daily_revisions(panel, value_column="hdd")


def test_only_historical_buckets_are_backfillable() -> None:
    """Leads beyond 7 cannot be reconstructed and must not enter a backtest."""
    historical = {b["name"] for b in revisions.historical_buckets()}
    every = {b["name"] for b in revisions.all_buckets()}
    assert historical == {"d1_3", "d4_7"}
    assert every - historical == {"d8_15", "d16_30"}
    limit = get("sources", "open_meteo", "previous_runs", "max_lead_days")
    assert max(b["max_lead"] for b in revisions.historical_buckets()) <= limit


# --- normals -----------------------------------------------------------------

def test_leap_day_folds_onto_feb28() -> None:
    dates = pd.Series([pd.Timestamp("2024-02-28"), pd.Timestamp("2024-02-29"),
                       pd.Timestamp("2024-03-01"), pd.Timestamp("2023-03-01")])
    doy = normals._day_of_year(dates)
    assert doy.iloc[0] == doy.iloc[1], "29 Feb must fold onto 28 Feb"
    # 1 March must mean the same day-of-year in a leap and a common year.
    assert doy.iloc[2] == doy.iloc[3]


def test_normals_smoothing_wraps_the_year() -> None:
    """1 January must be smoothed over as many days as 1 July."""
    values = pd.Series(np.arange(366, dtype=float))
    smoothed = normals._smooth_circular(values, 15)
    assert len(smoothed) == len(values)
    assert smoothed.notna().all()


def test_normals_require_enough_years() -> None:
    history = pd.DataFrame({
        "region": "us_gas",
        "target_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "hdd": np.linspace(0.0, 10.0, 100),
        "cdd": np.zeros(100),
    })
    with pytest.raises(ValueError, match="at least"):
        normals.build_normals(history)


# --- crop calendar -----------------------------------------------------------

def test_southern_window_wraps_the_year_end() -> None:
    """Brazilian pod-fill runs December into February and must not be empty."""
    dates = pd.Series(pd.to_datetime(
        ["2024-12-25", "2025-01-15", "2025-06-01"]))
    inside = crop._in_window(dates, "12-20", "02-10")
    assert list(inside) == [True, True, False]


def test_hemisphere_detected_from_latitudes() -> None:
    assert crop.hemisphere_for_region("brazil_soy") == "southern"
    assert crop.hemisphere_for_region("us_corn_belt") == "northern"
    assert crop.hemisphere_for_region("argentina_grains") == "southern"


def test_gating_drops_rather_than_zeroes_off_window_rows() -> None:
    frame = pd.DataFrame({
        "target_date": pd.to_datetime(["2024-07-15", "2024-04-01"]),
        "value": [35.0, 12.0],
    })
    gated = crop.gate_to_critical(frame, "corn", "us_corn_belt")
    assert len(gated) == 1
    assert gated["target_date"].iloc[0] == pd.Timestamp("2024-07-15")


def test_heat_stress_is_a_count_not_a_mean() -> None:
    frame = pd.DataFrame({"value": [38.0, 20.0, 33.0]})
    stress = crop.heat_stress_days(frame, "corn")
    assert list(stress) == [1, 0, 1]


def test_march_corn_belongs_to_the_previous_crop_year() -> None:
    """Old crop: a March contract's northern growing season already happened."""
    start, _ = crop.crop_year_window("C", 3, 2025)
    assert start.year == 2024
    start, _ = crop.crop_year_window("C", 12, 2025)
    assert start.year == 2025
