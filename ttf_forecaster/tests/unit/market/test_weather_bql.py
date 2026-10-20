"""BQL weather via Excel (D-042): lead mapping pinned to recorded probes, query shapes, parsing, the xlwings driver
(mocked), the pull orchestration with a fake runner, and the adapter's refusal of ticker-less BQL series.

The recorded values below are the 2026-10-19 probe results for DE Country, as_of 2026-10-01, run 0z, ECMWF
ENSEMBLE_MEAN, HDD_18C (see docs/bloomberg_discovery.md section 7). No Bloomberg call is made here.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import bql_excel, weather_bql as wb
from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    Backend,
    BackendUnavailable,
    DataError,
    IdentifierUnknown,
    validate_frame,
)
from ttf_forecaster.data.bloomberg import BloombergConfig, LiveBloombergAdapter
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.pipelines import bql_weather as pipe
from ttf_forecaster.settings import CONFIG_DIR, load_config

AS_OF = date(2026, 10, 1)
# Recorded: range offset(1..15) values for DE HDD_18C, as_of 2026-10-01 (row i = valid day AS_OF + i)
RANGE_HDD = [None, 1.93, 2.59, 2.98, 3.11, 2.36, 2.54, 4.95, 7.31, 6.43, 6.40, 6.67, 6.90, 6.61, 6.41]
# Recorded single-offset requests: offset k -> (valid date, value)
SINGLE = {2: (date(2026, 10, 2), 1.93), 5: (date(2026, 10, 5), 3.11), 9: (date(2026, 10, 9), 7.31)}
# Recorded actuals: range 2025-09-01..05 HDD_18C; single day 2025-09-05 = 1.84
ACT_RANGE = [0.57, 0.22, 0.05, 1.49, 1.84]
ACT_START = date(2025, 9, 1)
REF = "weather.de.regional_t2m_obs_degc"
BQL_DE = "weather.de.t2m_mean_obs_degc"


def _block(values_by_loc: dict[str, list[float | None]], fields: tuple[str, ...] = wb.FORECAST_FIELDS) -> pd.DataFrame:
    """A result frame shaped like the Excel driver output (showids=t): ID column, one column per field."""
    rows: list[dict[str, Any]] = []
    for loc, hdd in values_by_loc.items():
        for i, v in enumerate(hdd):
            row: dict[str, Any] = {"ID": loc}
            for f in fields:
                row[f"#w().{f}"] = v if f == "HDD_18C" else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _all_locs(hdd: list[float | None]) -> dict[str, list[float | None]]:
    return {loc: list(hdd) for loc in wb.LOCATIONS}


def _bundle(values_by_loc: dict[str, list[float | None]], models: int) -> pd.DataFrame:
    """A bundled forecast result: columns ``#m<j>().<FIELD>`` per model, HDD carries the recorded values."""
    rows: list[dict[str, Any]] = []
    for loc, hdd in values_by_loc.items():
        for i, v in enumerate(hdd):
            row: dict[str, Any] = {"ID": loc}
            for j in range(models):
                for f in wb.FORECAST_FIELDS:
                    row[f"#m{j}().{f}"] = v if f == "HDD_18C" else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- lead mapping (pinned)

class TestLeadMapping:
    def test_range_row_i_is_valid_day_as_of_plus_i(self) -> None:
        df = _bundle(_all_locs(RANGE_HDD), models=1)
        fc = wb.forecast_frame(df, AS_OF, "ECMWF", pd.Timestamp("2026-10-19", tz="UTC"),
                               models=[("ENSEMBLE_MEAN", "mean")])
        de = fc[(fc.series_id == "weather.de.hdd18_degc_day") & (fc.model == "ECMWF_ENSEMBLE_MEAN")]
        assert list(de.lead_hours) == [24.0 * i for i in range(15)]
        assert de.valid_time.iloc[0] == pd.Timestamp("2026-10-01", tz="UTC")
        assert de.valid_time.iloc[14] == pd.Timestamp("2026-10-15", tz="UTC")
        assert de.value.isna().iloc[0] and de.value.iloc[1] == pytest.approx(1.93)

    @pytest.mark.parametrize("k", sorted(SINGLE))
    def test_single_offset_k_equals_range_row_k_minus_1(self, k: int) -> None:
        """Offset k is the valid day AS_OF + k - 1 (recorded dates) and carries range row k - 1's value."""
        valid, value = SINGLE[k]
        assert valid == AS_OF + pd.Timedelta(days=k - 1).to_pytimedelta()
        assert RANGE_HDD[k - 1] == pytest.approx(value) or (value is None and RANGE_HDD[k - 1] is None)

    def test_dates_audit_rejects_a_shifted_mapping(self) -> None:
        class Fake:
            def __call__(self, queries, timeout_s=0):
                shifted = [AS_OF + pd.Timedelta(days=i + 1).to_pytimedelta() for i in range(15)]
                return {"audit": pd.DataFrame({"ID": ["DE Country"] * 15, "DATES": shifted})}
        with pytest.raises(DataError, match="lead mapping audit failed"):
            pipe.audit_dates(Fake(), AS_OF)

    def test_dates_audit_accepts_the_proved_mapping(self) -> None:
        dates = [AS_OF + pd.Timedelta(days=i).to_pytimedelta() for i in range(15)]

        class Fake:
            def __call__(self, queries, timeout_s=0):
                return {"audit": pd.DataFrame({"ID": ["DE Country"] * 15, "DATES": dates})}
        assert pipe.audit_dates(Fake(), AS_OF)[0].startswith("2026-10-01 dates ok")

    def test_actual_rows_are_stamped_with_the_observed_day(self) -> None:
        """Recorded 2026-10-19: the single-day request for 2025-09-05 returned 1.84, which is row 4 of the range
        from 2025-09-01. BQL labels the value of a requested day with the next day, so row 4 is the observed
        day 2025-09-06. The stored event time must be that day, not the requested one (D-045)."""
        df = pd.DataFrame({"ID": ["DE Country"] * 5, "#w().HDD_18C": ACT_RANGE,
                           "#w().CDD_18C": [0.0] * 5, "#w().TEMPERATURE": [10.0] * 5,
                           "#w().WIND_SPEED": [1.0] * 5})
        labels = wb.actual_event_dates(dates_frame(ACT_START + timedelta(days=1), 5), ACT_START,
                                       ACT_START + timedelta(days=4))
        obs = wb.observation_frame(df, labels, pd.Timestamp("2026-10-19", tz="UTC"))
        hdd = obs[obs.series_id == "weather.de.hdd18_obs_degc_day"].set_index("event_time").value
        assert hdd[pd.Timestamp("2025-09-06", tz="UTC")] == pytest.approx(1.84)
        assert pd.Timestamp("2025-09-05", tz="UTC") not in hdd.index or hdd[pd.Timestamp("2025-09-05", tz="UTC")] != 1.84
        assert hdd[pd.Timestamp("2025-09-02", tz="UTC")] == pytest.approx(0.57)
        assert obs.pit_grade.unique().tolist() == ["B"]
        assert (obs.publication_time == obs.event_time + pd.Timedelta(days=1, hours=12)).all()


class TestActualDates:
    """The observed-day labels, per location: the next day of the requested start, increasing, within the period (D-045)."""

    def test_labels_starting_the_day_after_the_request_are_accepted(self) -> None:
        labels = wb.actual_event_dates(dates_frame(date(2026, 10, 2), 6), date(2026, 10, 1), date(2026, 10, 6))
        assert labels["DE Country"] == [date(2026, 10, 2) + timedelta(days=i) for i in range(6)]

    def test_labels_on_the_requested_start_are_accepted_for_gb_style_locations(self) -> None:
        labels = wb.actual_event_dates(dates_frame(date(2026, 10, 1), 6, "GB Country"), date(2026, 10, 1), date(2026, 10, 6))
        assert labels["GB Country"][0] == date(2026, 10, 1)

    def test_labels_before_the_requested_start_are_refused(self) -> None:
        with pytest.raises(DataError, match="expected"):
            wb.actual_event_dates(dates_frame(date(2026, 9, 30), 6), date(2026, 10, 1), date(2026, 10, 6))

    def test_a_missing_day_is_kept_and_reported(self) -> None:
        gap = pd.DataFrame({"ID": ["DE Country"] * 3, "DATES": ["2026-10-02", "2026-10-03", "2026-10-05"]})
        labels = wb.actual_event_dates(gap, date(2026, 10, 1), date(2026, 10, 6))["DE Country"]
        assert wb.missing_days(labels) == [date(2026, 10, 4)]

    def test_labels_past_the_period_end_are_refused(self) -> None:
        with pytest.raises(DataError, match="past"):
            wb.actual_event_dates(dates_frame(date(2026, 10, 2), 9), date(2026, 10, 1), date(2026, 10, 6))

    def test_each_location_has_its_own_labels(self) -> None:
        both = pd.concat([dates_frame(date(2026, 10, 2), 5, "DE Country"),
                          dates_frame(date(2026, 10, 2), 4, "GB Country")], ignore_index=True)
        labels = wb.actual_event_dates(both, date(2026, 10, 1), date(2026, 10, 6))
        assert len(labels["DE Country"]) == 5 and len(labels["GB Country"]) == 4

    def test_a_count_mismatch_between_values_and_labels_is_refused(self) -> None:
        df = pd.DataFrame({"ID": ["DE Country"] * 5, "#w().HDD_18C": ACT_RANGE, "#w().CDD_18C": [0.0] * 5,
                           "#w().TEMPERATURE": [10.0] * 5, "#w().WIND_SPEED": [1.0] * 5})
        labels = {"DE Country": [date(2025, 9, 2) + timedelta(days=i) for i in range(4)]}
        with pytest.raises(ValueError, match="value rows"):
            wb.observation_frame(df, labels, pd.Timestamp("2026-10-19", tz="UTC"))

    def test_the_dates_query_asks_for_dates_only_for_every_location(self) -> None:
        q = wb.actual_dates_query(date(2026, 10, 1), date(2026, 10, 6))
        assert "period_reference_date" in q and "TEMPERATURE" not in q and "ae='A'" in q and "DE Country" in q


def dates_frame(first: date, n: int, loc: str = "DE Country") -> pd.DataFrame:
    return pd.DataFrame({"ID": [loc] * n,
                         "DATES": [(first + timedelta(days=i)).isoformat() for i in range(n)]})


class TestAlignmentCheck:
    """Permanent regression (D-045): BQL actuals must correlate best with the WSI regional series at lag 0."""

    @staticmethod
    def _series(days: list[date], values: np.ndarray, sid: str, tz_local: bool) -> pd.DataFrame:
        if tz_local:  # WSI: Amsterdam midnight, stored as 23:00 UTC the day before
            ev = [pd.Timestamp(d, tz="Europe/Amsterdam").tz_convert("UTC") for d in days]
        else:  # BQL: UTC midnight
            ev = [pd.Timestamp(d, tz="UTC") for d in days]
        return pd.DataFrame({"series_id": sid, "event_time": ev, "value": values})

    def _fixture(self, bql_shift_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(7)
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(120)]
        truth = 5 + 10 * np.sin(np.arange(120) / 9.0) + rng.normal(0, 0.5, 120)
        # the BQL value stored on day D equals the true observation of day D + bql_shift_days
        bql_vals = truth
        bql_days = [d - timedelta(days=bql_shift_days) for d in days]
        bql = self._series(bql_days, bql_vals, BQL_DE, tz_local=False)
        wsi = self._series(days, truth, REF, tz_local=True)
        return bql, wsi

    def test_aligned_actuals_pass_with_best_lag_zero(self) -> None:
        bql, wsi = self._fixture(bql_shift_days=0)
        corr = pipe.check_actuals_alignment(bql, wsi, pairs=[(BQL_DE, REF)])
        assert max(corr[BQL_DE], key=corr[BQL_DE].get) == 0

    def test_a_one_day_early_store_raises(self) -> None:
        """The defect of 2026-10-19: BQL stored one day early. The check must refuse it."""
        bql, wsi = self._fixture(bql_shift_days=1)
        with pytest.raises(DataError, match="lag"):
            pipe.check_actuals_alignment(bql, wsi, pairs=[(BQL_DE, REF)])

    def test_a_missing_reference_raises(self) -> None:
        bql, _ = self._fixture(bql_shift_days=0)
        with pytest.raises(DataError, match="needs"):
            pipe.check_actuals_alignment(bql, bql.iloc[0:0], pairs=[(BQL_DE, REF)])


# ---------------------------------------------------------------- query shapes

class TestQueries:
    def test_forecast_query_bundles_every_model_for_one_source(self) -> None:
        q = wb.forecast_query(AS_OF, "ECMWF")
        assert "as_of_date='2026-10-01'" in q and "est_source='ECMWF'" in q
        for mt in ("ENSEMBLE_MEAN", "DETERMINISTIC", "ENSEMBLE_MIN", "ENSEMBLE_MAX", "ENSEMBLE_MEDIAN"):
            assert f"model_type=['{mt}']" in q
        assert q.count("offset=range(1,16)") == 5 and "run='0z'" in q
        assert "weight_type='POPULATION'" in q
        for j in range(5):
            for f in wb.FORECAST_FIELDS:
                assert f"#m{j}().{f}" in q
        assert "'NORTHWEST EUROPE REGION'" in q and "'DE Country'" in q
        assert "est_source='GFS'" in wb.forecast_query(AS_OF, "GFS") and "model_type=['DETERMINISTIC']" in wb.forecast_query(AS_OF, "GFS")

    def test_actual_query_uses_literal_dates_and_no_solar(self) -> None:
        q = wb.actual_query(date(2025, 1, 1), date(2025, 12, 31))
        assert "period_reference=range('2025-01-01','2025-12-31')" in q
        assert "SOLAR_RADIATION" not in q  # rejected by Excel for actuals (2026-10-19)
        assert "as_of_date" not in q

    def test_normals_query_has_no_solar(self) -> None:
        q = wb.normals_query(date(2026, 1, 1), "NORMALS_10Y")
        assert "est_source='NORMALS_10Y'" in q and "SOLAR_RADIATION" not in q

    def test_series_ids_follow_the_convention(self) -> None:
        assert wb.series_id("de", "HDD_18C", observed=False) == "weather.de.hdd18_degc_day"
        assert wb.series_id("de", "HDD_18C", observed=True) == "weather.de.hdd18_obs_degc_day"
        assert wb.series_id("nwe", "TEMPERATURE", observed=False, normal="NORMALS_10Y") == \
            "weather.nwe.t2m_mean_normals_10y_degc"

    def test_split_rejects_a_location_with_the_wrong_row_count(self) -> None:
        df = _block({"DE Country": RANGE_HDD[:10]})
        with pytest.raises(ValueError, match="rows per location, expected 15"):
            wb._split_by_id(df, 15)


# ---------------------------------------------------------------- frames pass the schemas

def test_forecast_frame_is_schema_valid_and_point_in_time() -> None:
    df = _bundle(_all_locs(RANGE_HDD), models=5)
    fc = wb.forecast_frame(df, AS_OF, "ECMWF", pd.Timestamp("2026-10-19", tz="UTC"))
    assert set(fc.member) == {"mean", "det", "min", "max", "median"}
    assert len(fc) == 12 * 5 * 5 * 15
    validate_frame(fc, FORECAST_SCHEMA)
    assert (fc.issue_time == pd.Timestamp("2026-10-01", tz="UTC")).all()
    assert (fc.publication_time == pd.Timestamp("2026-10-01 10:00", tz="UTC")).all()
    assert (fc.publication_time >= fc.issue_time).all()
    assert set(fc.pit_grade) == {"A"} and set(fc.entitlement) == {"licensed_bloomberg"}


def test_normal_frame_is_grade_c() -> None:
    dates = [AS_OF + pd.Timedelta(days=i).to_pytimedelta() for i in range(16)]
    df = _block(_all_locs([4.0] * 16), fields=wb.OBSERVED_FIELDS)
    nf = wb.normal_frame(df, dates, "NORMALS_10Y", pd.Timestamp("2026-10-19", tz="UTC"))
    validate_frame(nf, OBSERVATION_SCHEMA)
    assert set(nf.pit_grade) == {"C"}


# ---------------------------------------------------------------- the xlwings driver, with a fake Excel

# ---------------------------------------------------------------- orchestration with a fake runner

def _fake_runner_factory(calls: list[dict[str, str]]):
    def runner(queries: dict[str, str], timeout_s: int = 0):
        calls.append(queries)
        out: dict[str, Any] = {}
        for name, q in queries.items():
            if "ae='A'" in q:
                n = 5
                out[name] = pd.DataFrame({"ID": ["DE Country"] * n, "#w().HDD_18C": ACT_RANGE,
                                          "#w().CDD_18C": [0.0] * n, "#w().TEMPERATURE": [10.0] * n,
                                          "#w().WIND_SPEED": [1.0] * n})
            elif "period_reference_date" in q:
                out[name] = pd.DataFrame({"ID": ["DE Country"] * 15,
                                          "DATES": [AS_OF + pd.Timedelta(days=i).to_pytimedelta() for i in range(15)]})
            else:
                models = 2 if "est_source='GFS'" in q else 5
                out[name] = _bundle(_all_locs(RANGE_HDD + ([None] if models == 99 else [])), models=models)
        return out
    return runner


def test_pull_forecasts_stores_schema_valid_rows(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    calls: list[dict[str, str]] = []
    summary = pipe.pull_forecasts(AS_OF, AS_OF, store, _fake_runner_factory(calls), chunk_days=5)
    assert summary.failures == []
    assert summary.audits and summary.audits[0].startswith("2026-10-01 dates ok")
    assert sum(len(c) for c in calls) == len(wb.FORECAST_BUNDLES) + 1  # one as_of: ECMWF + GFS bundles, + audit
    assert summary.rows["forecast ECMWF_ENSEMBLE_MEAN"] == 12 * 5 * 15
    assert summary.rows["forecast GFS_DETERMINISTIC"] == 12 * 5 * 15


def test_pull_forecasts_records_a_failed_query_and_continues(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)

    def runner(queries, timeout_s=0):
        out = {}
        for name, q in queries.items():
            if "period_reference_date" in q:
                out[name] = pd.DataFrame({"ID": ["DE Country"] * 15,
                                          "DATES": [AS_OF + pd.Timedelta(days=i).to_pytimedelta() for i in range(15)]})
            elif "est_source='GFS'" in q:
                out[name] = "#N/A Invalid Parameter"
            else:
                out[name] = _bundle(_all_locs(RANGE_HDD), models=5)
        return out
    summary = pipe.pull_forecasts(AS_OF, AS_OF, store, runner, chunk_days=5)
    assert any("GFS" in f or "Invalid" in f for f in summary.failures)
    assert "forecast ECMWF_ENSEMBLE_MEAN" in summary.rows


def test_pull_actuals_checks_the_day_count(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    calls: list[dict[str, str]] = []
    def runner(queries, timeout_s=0):
        calls.append(queries)
        # a short block (5 days) for a whole year must be refused, not stored
        return {name: pd.DataFrame({"ID": ["DE Country"] * 5, "#w().HDD_18C": ACT_RANGE,
                                    "#w().CDD_18C": [0.0] * 5, "#w().TEMPERATURE": [10.0] * 5,
                                    "#w().WIND_SPEED": [1.0] * 5}) for name in queries}
    summary = pipe.pull_actuals([2025], store, runner)
    assert summary.failures and "act_2025" in summary.failures[0]
    assert "actual" not in summary.rows


# ---------------------------------------------------------------- the Bloomberg adapter never serves BQL series

def test_live_adapter_refuses_bql_weather_series(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    import shutil
    shutil.copytree(CONFIG_DIR, cfg_dir, ignore=shutil.ignore_patterns("local_*"))
    (cfg_dir / "local_data_sources.yaml").write_text("live_calls_authorised: true\n", encoding="utf-8")
    load_config.cache_clear()

    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"Bloomberg client called: {name}")
    adapter = LiveBloombergAdapter(config_dir=str(cfg_dir), client=_Exploding())
    with pytest.raises(IdentifierUnknown, match="weather.de.hdd18_degc_day"):
        adapter.get_historical_series(["weather.de.hdd18_degc_day"], date(2026, 1, 1), date(2026, 1, 31))
    cfg = BloombergConfig.load(str(cfg_dir))
    assert cfg.series["weather.de.hdd18_degc_day"].is_known is False
    assert cfg.series["weather.de.hdd18_degc_day"].status == "known"  # status known, no ticker: not served
