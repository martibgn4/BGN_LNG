"""data/weather.py: backends, weights policy and the Open-Meteo silent-failure guards."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    Backend,
    BackendUnavailable,
    IdentifierUnknown,
    SchemaError,
    WeatherAdapter,
    validate_frame,
)
from ttf_forecaster.data.weather import (
    SYNTHETIC_EQUAL_WEIGHTS,
    LeadUnavailable,
    LiveWeatherAdapter,
    LocalFileWeatherAdapter,
    MockWeatherAdapter,
    SyntheticWeatherConfig,
    WeightsUnavailable,
    check_ensemble_depth,
    hourly_to_daily,
    load_weather_config,
    make_weather_adapter,
    previous_runs_to_forecasts,
    region_from_config,
    validate_lead_coverage,
)

INGEST = pd.Timestamp("2024-01-10", tz="UTC")


def _hourly(days: int = 3, leads: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    df = pd.DataFrame({"time": t, "temperature_2m": 1.0})
    for n in leads:
        df[f"temperature_2m_previous_day{n}"] = 1.0 + n
    return df


class TestConfigAndWeights:
    def test_config_weights_are_requires_user_input_and_refused(self) -> None:
        region = region_from_config(load_weather_config(), "eu_ldz")
        assert all(p.weight is None for p in region.points)
        with pytest.raises(WeightsUnavailable, match="BNEF"):
            region.weights()

    def test_equal_weights_only_when_explicit_and_labelled(self) -> None:
        region = region_from_config(load_weather_config(), "eu_ldz")
        w, label = region.weights(allow_synthetic_equal=True)
        assert label == SYNTHETIC_EQUAL_WEIGHTS
        assert pytest.approx(sum(w.values())) == 1.0

    def test_licence_not_confirmed(self) -> None:
        assert load_weather_config()["licence"]["commercial_use_confirmed"] is False


class TestLeadGuards:
    def test_previous_runs_mapping_conservative(self) -> None:
        out = previous_runs_to_forecasts(_hourly(), series_id="weather.aa.t2m_mean_degc",
                                         geography="AA", model="m", ingestion_time=INGEST,
                                         leads=[1, 2])
        validate_frame(out, FORECAST_SCHEMA)
        row = out.loc[(out["lead_hours"] == 48.0)
                      & (out["valid_time"] == pd.Timestamp("2024-01-03", tz="UTC"))].iloc[0]
        assert row["issue_time"] == pd.Timestamp("2024-01-01", tz="UTC")
        # end of issue day + 8 h availability lag
        assert row["publication_time"] == pd.Timestamp("2024-01-02 08:00", tz="UTC")
        assert bool(row["publication_time_estimated"])
        assert row["value"] == 3.0

    def test_lead_beyond_seven_refused(self) -> None:
        with pytest.raises(LeadUnavailable, match="zero revisions"):
            previous_runs_to_forecasts(_hourly(leads=(8,)), series_id="s", geography="X",
                                       model="m", ingestion_time=INGEST, leads=[8])

    def test_all_null_lead_column_is_an_error(self) -> None:
        h = _hourly(leads=(1, 7))
        h["temperature_2m_previous_day7"] = np.nan
        with pytest.raises(LeadUnavailable, match="entirely null"):
            previous_runs_to_forecasts(h, series_id="s", geography="X", model="m",
                                       ingestion_time=INGEST, leads=[1, 7])

    def test_incomplete_day_is_nan_not_partial_mean(self) -> None:
        h = _hourly(leads=(1,))
        h.loc[5, "temperature_2m_previous_day1"] = np.nan
        out = previous_runs_to_forecasts(h, series_id="s", geography="X", model="m",
                                         ingestion_time=INGEST, leads=[1])
        bad = out.loc[out["valid_time"] == pd.Timestamp("2024-01-01", tz="UTC")]
        assert bad["value"].isna().all()
        assert (bad["quality_flag"] == "missing_input").all()

    def test_naive_hourly_rejected(self) -> None:
        h = _hourly()
        h["time"] = h["time"].dt.tz_localize(None)
        with pytest.raises(SchemaError, match="tz-aware"):
            hourly_to_daily(h, "temperature_2m", 24)

    def test_validate_lead_coverage(self, syn_points: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        fc = syn_points[0]
        validate_lead_coverage(fc.loc[fc["model"] == "SYNTH_DET"], 7)
        with pytest.raises(LeadUnavailable, match="lead > 7"):
            validate_lead_coverage(fc.loc[fc["model"] == "SYNTH_ENS"], 7)  # ens reaches 10 d
        dead = fc.loc[fc["model"] == "SYNTH_DET"].copy()
        dead.loc[dead["lead_hours"] == 168.0, "value"] = np.nan
        with pytest.raises(LeadUnavailable, match="all-null"):
            validate_lead_coverage(dead, 7)

    def test_ensemble_null_padding_detected(self,
                                            syn_points: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        ens = syn_points[0].loc[syn_points[0]["model"] == "SYNTH_ENS"]
        check_ensemble_depth(ens, 10, 10, 0.8)
        with pytest.raises(LeadUnavailable, match="null padding"):
            check_ensemble_depth(ens, 30, 34, 0.8)  # model claims 34 but data has 10 days
        with pytest.raises(LeadUnavailable, match="measured"):
            check_ensemble_depth(ens, 35, 10, 0.8)


class TestBackends:
    def test_factory_mock_satisfies_protocol(self) -> None:
        a = make_weather_adapter("mock", cfg=SyntheticWeatherConfig(
            obs_start="2021-10-01", start="2022-01-01", end="2022-01-31"))
        assert isinstance(a, WeatherAdapter) and a.backend is Backend.MOCK
        fc = a.get_forecast_runs(["weather.aa.t2m_mean_degc", "weather.eu.hdd"],
                                 date(2022, 1, 3), date(2022, 1, 4))
        assert set(fc["source"]) == {"SYNTHETIC"}
        assert fc["issue_time"].dt.date.between(date(2022, 1, 3), date(2022, 1, 4)).all()
        with pytest.raises(IdentifierUnknown):
            a.get_observed(["weather.zz.t2m_mean_obs_degc"], date(2022, 1, 1), date(2022, 1, 2))

    def test_live_refuses_without_authorisation(self) -> None:
        with pytest.raises(BackendUnavailable, match="not authorised"):
            make_weather_adapter("live")

    def test_live_refuses_without_licence_even_if_authorised(self) -> None:
        with pytest.raises(BackendUnavailable, match="NON-COMMERCIAL"):
            LiveWeatherAdapter(live_calls_authorised=True)

    def test_local_file_roundtrip_and_naive_csv(self, tmp_path,  # type: ignore[no-untyped-def]
                                                syn_points: tuple[pd.DataFrame, pd.DataFrame]
                                                ) -> None:
        fc = syn_points[0].loc[syn_points[0]["model"] == "SYNTH_DET"].head(70)
        (tmp_path / "weather").mkdir()
        fc.to_parquet(tmp_path / "weather" / "forecasts_a.parquet")
        a = LocalFileWeatherAdapter(extracts_dir=tmp_path)
        got = a.get_forecast_runs(["weather.aa.t2m_mean_degc"], date(2022, 1, 1),
                                  date(2024, 1, 1))
        assert len(got) == len(fc.loc[fc["series_id"] == "weather.aa.t2m_mean_degc"])
        obs = syn_points[1].head(5).copy()
        obs["event_time"] = obs["event_time"].dt.strftime("%Y-%m-%d %H:%M")  # no offset
        obs.drop(columns=["publication_time"]).to_csv(tmp_path / "weather" / "observed.csv",
                                                      index=False)
        with pytest.raises(SchemaError, match="offset"):
            a.get_observed([obs["series_id"].iloc[0]], date(2010, 1, 1), date(2030, 1, 1))
