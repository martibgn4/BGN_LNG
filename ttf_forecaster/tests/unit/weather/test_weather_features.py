"""features/weather.py on SYNTHETIC data: conventions, revisions, PIT and the zero-revision trap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import PointInTimeViolation
from ttf_forecaster.data.weather import SYNTHETIC_EQUAL_WEIGHTS
from ttf_forecaster.features.weather import (
    build_weather_features,
    cooling_degree_days,
    heating_degree_days,
    noleap_doy,
    seasonal_normal_table,
)
from ttf_forecaster.settings import ConfigError

from .conftest import bundle, origins

FC_WINDOW = ("2023-01-09", "2023-01-20")


def _build(fc: pd.DataFrame, obs: pd.DataFrame, params: dict[str, object],
           o: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    return build_weather_features(bundle(obs, fc), o if o is not None else origins(*FC_WINDOW),
                                  params)


class TestDegreeDays:
    def test_eurostat_hdd_cdd(self) -> None:
        t = np.array([10.0, 15.0, 15.1, np.nan, 25.0, 23.9])
        hdd = heating_degree_days(t, 18.0, 15.0)
        np.testing.assert_allclose(hdd[:3], [8.0, 3.0, 0.0])
        assert np.isnan(hdd[3])
        cdd = cooling_degree_days(t, 21.0, 24.0)
        np.testing.assert_allclose(cdd[[0, 4, 5]], [0.0, 4.0, 0.0])

    def test_noleap_doy_maps_feb29(self) -> None:
        d = pd.DatetimeIndex(["2024-02-28", "2024-02-29", "2024-03-01", "2023-12-31"])
        assert list(noleap_doy(d)) == [59, 59, 60, 365]

    def test_normals_use_only_prior_gas_years(self) -> None:
        days = pd.date_range("2015-10-01", "2023-09-30", freq="D", tz="UTC")
        d = pd.DataFrame({"day": days, "hdd": 1.0})
        d.loc[d["day"] >= pd.Timestamp("2022-10-01", tz="UTC"), "hdd"] = 99.0
        tab = seasonal_normal_table(d, ["hdd"], 2022, 30, 3, 7, 10)
        assert tab is not None and np.allclose(tab["hdd"], 1.0)
        assert seasonal_normal_table(d, ["hdd"], 2017, 30, 3, 7, 10) is None  # only 2 years


class TestBuilder:
    def test_columns_grades_and_labels(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        f = _build(*syn_points, syn_params)
        assert all(c.startswith("weather__") for c in f.columns)
        assert f.index.name == "origin_time"
        assert f.attrs["weights_label"] == SYNTHETIC_EQUAL_WEIGHTS
        g = f.attrs["pit_grade"]
        assert g["weather__hdd_rev_d1_3"] == "A"
        assert g["weather__obs_hdd_anom_7d"] == "B"
        assert g["weather__hdd_fc_anom_d1_6"] == "B"  # normal from grade-B observations
        assert "weather__hdd_rev_d1_3" in f.attrs["heating_season_only"]
        assert f[["weather__hdd_rev_d1_3", "weather__hdd_fc_d1_6",
                  "weather__hdd_ens_spread_d1_7"]].notna().all().all()
        assert (f["weather__hdd_ens_spread_d1_7"] > 0).all()

    def test_config_region_without_weights_raises(self, syn_points) -> None:  # type: ignore[no-untyped-def]
        from ttf_forecaster.data.weather import WeightsUnavailable

        with pytest.raises(WeightsUnavailable):
            _build(*syn_points, {})

    def test_colder_new_run_gives_positive_revision(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-17", "2023-01-17")
        base = _build(fc, obs, syn_params, o)
        colder = fc.copy()
        run = colder["issue_time"] == pd.Timestamp("2023-01-17", tz="UTC")
        colder.loc[run, "value"] -= 3.0  # all points, all leads of the newest run
        new = _build(colder, obs, syn_params, o)
        delta = new["weather__hdd_rev_d1_3"].iloc[0] - base["weather__hdd_rev_d1_3"].iloc[0]
        # 3 days x 3 degC, provided every temperature stays below the 15 C threshold
        assert delta == pytest.approx(9.0)

    def test_no_newer_run_is_nan_not_zero(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-17", "2023-01-17")
        missing = fc.loc[fc["issue_time"] != pd.Timestamp("2023-01-17", tz="UTC")]
        f = _build(missing, obs, syn_params, o)
        assert np.isnan(f["weather__hdd_rev_d1_3"].iloc[0])
        assert f.attrs["flags"]["weather__hdd_rev_d1_3"][o[0].tz_convert("UTC").isoformat()] \
            == "no_newer_run"

    def test_missing_previous_state_is_nan(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-17", "2023-01-17")
        only_new = fc.loc[fc["issue_time"] >= pd.Timestamp("2023-01-17", tz="UTC")]
        f = _build(only_new, obs, syn_params, o)
        assert np.isnan(f["weather__hdd_rev_d1_5"].iloc[0])
        assert np.isfinite(f["weather__hdd_fc_d1_6"].iloc[0])

    def test_leakage_canary(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-16", "2023-01-16")
        cut = o[0].tz_convert("UTC")
        fc2, obs2 = fc.copy(), obs.copy()
        fc2.loc[fc2["publication_time"] > cut, "value"] += 50.0
        obs2.loc[obs2["publication_time"] > cut, "value"] -= 50.0
        a, b = _build(fc, obs, syn_params, o), _build(fc2, obs2, syn_params, o)
        pd.testing.assert_frame_equal(a, b)

    def test_current_gas_year_obs_do_not_move_normals(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-16", "2023-01-16")
        obs2 = obs.copy()
        obs2.loc[obs2["event_time"] >= pd.Timestamp("2022-10-01", tz="UTC"), "value"] -= 5.0
        a, b = _build(fc, obs, syn_params, o), _build(fc, obs2, syn_params, o)
        assert a["weather__hdd_fc_anom_d1_6"].iloc[0] == b["weather__hdd_fc_anom_d1_6"].iloc[0]
        assert a["weather__obs_hdd_anom_7d"].iloc[0] != b["weather__obs_hdd_anom_7d"].iloc[0]

    def test_observed_features_respect_lag(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        fc, obs = syn_points
        o = origins("2023-01-16", "2023-01-16")
        obs2 = obs.copy()
        recent = obs2["event_time"] >= pd.Timestamp("2023-01-12", tz="UTC")  # D-4: not yet public
        obs2.loc[recent, "value"] -= 20.0
        a, b = _build(fc, obs, syn_params, o), _build(fc, obs2, syn_params, o)
        assert a["weather__obs_hdd_anom_7d"].iloc[0] == b["weather__obs_hdd_anom_7d"].iloc[0]

    def test_revision_window_beyond_archive_refused(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        p = dict(syn_params, revision={"windows": ["d1_7"]})
        with pytest.raises(ConfigError, match="max lead"):
            _build(*syn_points, p)
        p = dict(syn_params, level_windows=["d8_14"])
        with pytest.raises(ConfigError, match="live-accumulated"):
            _build(*syn_points, p)

    def test_naive_origins_rejected(self, syn_points, syn_params) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(PointInTimeViolation):
            _build(*syn_points, syn_params, pd.DatetimeIndex(["2023-01-16 19:00"]))

    def test_preaggregated_synthetic_hdd(self) -> None:
        cfg = synthetic.SyntheticConfig(start="2022-11-01", end="2023-01-31")
        fc = synthetic.weather_forecast_runs(cfg)
        from ttf_forecaster.data.base import OBSERVATION_SCHEMA, empty_frame

        f = build_weather_features(
            bundle(empty_frame(OBSERVATION_SCHEMA), fc), origins("2023-01-09", "2023-01-13"),
            {"region": "eu_preaggregated", "deterministic": {"model": "SYNTH_ENS",
                                                             "member": "mean"},
             "ensemble_windows": []},
        )
        assert f["weather__hdd_rev_d1_3"].notna().all()
        assert f["weather__cdd_rev_d1_3"].isna().all()
        assert f["weather__hdd_fc_anom_d1_6"].isna().all()  # no observed HDD -> no normal
