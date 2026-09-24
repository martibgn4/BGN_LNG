"""BNEF LDZ loader tests. SPEC 2a, SPEC 3 (is_estimated), SPEC 11.

The behaviours worth pinning are the ones that would silently corrupt the fit:
loading a modelled column as if it were an observation, loading German
nominations as if they were actuals, or double-counting the file's own
aggregate rows.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gasbalance.data.base import SourceUnavailable
from gasbalance.data.bnef_ldz import VALUE_COLUMNS, BnefLdzFetcher


@pytest.fixture(scope="module")
def fetcher() -> BnefLdzFetcher:
    return BnefLdzFetcher()


@pytest.fixture(scope="module")
def loaded(fetcher: BnefLdzFetcher) -> pd.DataFrame:
    if not fetcher.is_available():
        pytest.skip("BNEF workbook not present on this machine")
    return fetcher.fetch(date(2022, 10, 1), date(2025, 9, 30)).frame


# -- the three columns are not the same kind of thing ------------------------


def test_only_actuals_are_unflagged_observations() -> None:
    """SPEC 3: anything not observed must carry is_estimated."""
    assert VALUE_COLUMNS["actual"] == ("ldz_demand", False)
    assert VALUE_COLUMNS["expected"][1] is True
    assert VALUE_COLUMNS["benchmark"][1] is True


def test_modelled_columns_are_flagged_in_the_output(loaded: pd.DataFrame) -> None:
    """BNEF's regression must never be mistaken for data.

    Fitting our model against BNEF's model would teach it to reproduce BNEF
    rather than reality, and the backtest would look good while being circular.
    """
    flags = loaded.groupby("series_id")["is_estimated"].agg(lambda s: bool(s.iloc[0]))
    assert flags["ldz_demand"] is False or flags["ldz_demand"] == False  # noqa: E712
    assert flags["ldz_demand_bnef_expected"]
    assert flags["ldz_demand_normal_2016_2020"]


def test_fitting_target_is_only_the_unflagged_series(loaded: pd.DataFrame) -> None:
    target = loaded[~loaded["is_estimated"]]
    assert set(target["series_id"].unique()) == {"ldz_demand"}


# -- region mapping ----------------------------------------------------------


def test_german_actuals_come_from_implied_flow_not_nominations(
    fetcher: BnefLdzFetcher,
) -> None:
    """BNEF states nominations are "not reflective of actual demand"."""
    mapping = fetcher.region_map()
    assert mapping["Germany implied flow"] == "DE"
    assert mapping["Germany flow nominations"] is None


def test_file_aggregates_are_excluded(fetcher: BnefLdzFetcher) -> None:
    """Loading 'Europe Perimeter' alongside its members would double-count."""
    mapping = fetcher.region_map()
    assert mapping["Europe Perimeter"] is None
    assert mapping["Europe Perimeter excluding Germany"] is None


def test_countries_outside_the_ring_are_excluded(fetcher: BnefLdzFetcher) -> None:
    mapping = fetcher.region_map()
    assert mapping["Italy"] is None
    assert mapping["Spain"] is None


def test_unmapped_region_raises_rather_than_dropping(fetcher: BnefLdzFetcher) -> None:
    """A new region in next week's export must be classified deliberately."""
    raw = pd.DataFrame(
        {
            "region": ["Atlantis"],
            "date": [pd.Timestamp("2025-01-15")],
            "actual": [1.0],
            "benchmark": [1.0],
            "expected": [1.0],
            "vs_benchmark_pct": [0.0],
            "vs_expected_pct": [0.0],
        }
    )
    with pytest.raises(SourceUnavailable, match="not mapped"):
        fetcher._normalise(raw, date(2025, 1, 1), date(2025, 12, 31), fetcher.now())


# -- loaded data -------------------------------------------------------------


def test_loaded_countries_are_inside_the_perimeter(loaded: pd.DataFrame) -> None:
    from gasbalance.perimeter import Perimeter

    assert set(loaded["country"].unique()) <= Perimeter().inside


def test_values_are_in_the_internal_flow_unit(loaded: pd.DataFrame) -> None:
    assert set(loaded["unit"].unique()) == {"mcm/d"}


def test_ldz_demand_is_strongly_seasonal(loaded: pd.DataFrame) -> None:
    """LDZ is space heating. If it is not ~4-6x higher in January than July,
    the wrong column or the wrong regions have been loaded."""
    actual = loaded[loaded["series_id"] == "ldz_demand"].copy()
    actual["month"] = pd.to_datetime(actual["obs_date"]).dt.month
    daily = actual.groupby(["obs_date", "month"], as_index=False)["value"].sum()
    january = daily.loc[daily["month"] == 1, "value"].mean()
    july = daily.loc[daily["month"] == 7, "value"].mean()
    assert january / july > 3.0
    assert january / july < 8.0


def test_no_value_was_interpolated(loaded: pd.DataFrame) -> None:
    """SPEC 11: gaps are dropped and reported, never filled."""
    assert loaded["value"].notna().all()


def test_missing_workbook_raises(tmp_path) -> None:
    fetcher = BnefLdzFetcher(workbook=tmp_path / "nope.xlsx", cache_dir=tmp_path)
    assert not fetcher.is_available()
    with pytest.raises(SourceUnavailable, match="does not generate demand data"):
        fetcher.fetch(date(2025, 1, 1), date(2025, 1, 2))
