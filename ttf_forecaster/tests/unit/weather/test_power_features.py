"""features/power.py on SYNTHETIC frames: economics by hand, PIT, missing-parameter behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.features.power import (
    build_power_features,
    clean_dark_spread,
    clean_spark_spread,
    coal_eur_per_mwh_th,
    prior_years_normal,
    switching_gas_price,
)

from .conftest import bundle, make_fc, make_obs, origins

EFF = {"gas_plant_efficiency": 0.50, "coal_plant_efficiency": 0.40}  # SYNTHETIC test values
O = origins("2024-01-15", "2024-01-19")


class TestEconomics:
    def test_hand_computed(self) -> None:
        coal = coal_eur_per_mwh_th(120.0, 1.2, 6.978)
        assert coal == pytest.approx(120.0 / 1.2 / 6.978)
        assert clean_spark_spread(100.0, 30.0, 80.0, 0.5, 0.202) == pytest.approx(
            100 - 60 - 80 * 0.404)
        assert clean_dark_spread(100.0, 14.0, 80.0, 0.4, 0.3406) == pytest.approx(
            100 - 35 - 80 * 0.8515)

    def test_switching_price_equalises_spreads(self) -> None:
        coal, eua = 14.0, 70.0
        g = switching_gas_price(coal, eua, 0.5, 0.4, 0.202, 0.3406)
        assert clean_spark_spread(90, g, eua, 0.5, 0.202) == pytest.approx(
            clean_dark_spread(90, coal, eua, 0.4, 0.3406))

    def test_prior_years_normal_cutoff_and_min_years(self) -> None:
        days = pd.date_range("2018-01-01", "2024-03-01", freq="D", tz="UTC")
        hist = pd.Series(10.0, index=days)
        hist.loc[hist.index >= pd.Timestamp("2023-10-01", tz="UTC")] = 99.0
        tgt = pd.date_range("2024-01-16", "2024-01-16", tz="UTC")
        cutoff = pd.Timestamp("2023-10-01", tz="UTC")
        assert prior_years_normal(hist, tgt, cutoff, 5, 3, 7) == pytest.approx(10.0)
        assert np.isnan(prior_years_normal(hist, tgt, cutoff, 2, 3, 7))
        cap = pd.Series([50.0, 100.0], index=pd.DatetimeIndex(["2018-01-01", "2023-01-01"],
                                                               tz="UTC"))
        # capacity doubled since every prior-year reference -> normal scaled up
        assert prior_years_normal(hist, tgt, cutoff, 1, 1, 7, cap, 100.0) == pytest.approx(10.0)
        assert prior_years_normal(hist, tgt, cutoff, 5, 1, 7, cap, 100.0) > 10.0


def _prices(days: pd.DatetimeIndex) -> pd.DataFrame:
    lag = pd.Timedelta(days=1, hours=18)  # next-day 18:00 UTC, i.e. not same-session
    return pd.concat([
        make_obs("price.power_de.base_m1_eur_mwh", days, 100.0, lag, "EUR/MWh", "A"),
        make_obs("price.ttf.m1_eur_mwh", days, 30.0, lag, "EUR/MWh", "A"),
        make_obs("price.coal_api2.m1_usd_t", days, 120.0, lag, "USD/t", "A"),
        make_obs("price.eua.front_eur_t", days, 70.0, lag, "EUR/t", "A"),
        make_obs("fx.eurusd.spot_usd_per_eur", days, 1.2, lag, "USD/EUR", "A"),
    ], ignore_index=True)


@pytest.fixture(scope="module")
def power_bundle():  # type: ignore[no-untyped-def]
    """Raw MW components; the GW ids the features read are derived (data/derived.py, D-034)."""
    hist_days = pd.date_range("2018-01-01", "2024-01-20", freq="D")
    fc_days = pd.date_range("2024-01-01", "2024-01-22", freq="D")
    def fc_vintages(sid: str, value: float) -> pd.DataFrame:
        """Two vintages per delivery day: an earlier run (revision 0, 2 days out) and the
        day-ahead value (revision 1, published the evening before)."""
        early = make_obs(sid, fc_days, value, pd.Timedelta(days=-2, hours=6), "MW")
        late = make_obs(sid, fc_days, value, pd.Timedelta(hours=-9), "MW")
        late["revision"] = 1
        return pd.concat([early, late], ignore_index=True)

    obs = pd.concat([
        fc_vintages("power.de.wind_onshore_fc_mw", 20000.0),
        fc_vintages("power.de.wind_offshore_fc_mw", 10000.0),
        fc_vintages("power.de.solar_fc_tennet_mw", 2000.0),
        fc_vintages("power.de.solar_fc_50hertz_mw", 1000.0),
        fc_vintages("power.de.solar_fc_amprion_mw", 1000.0),
        make_obs("power.de.wind_total_gen_mw", hist_days, 20000.0, unit="MW"),
        make_obs("power.de.solar_gen_mw", hist_days, 5000.0, unit="MW"),
        make_obs("power.fr.outage_nuclear_unavail_mw", hist_days, 13000.0, unit="MW"),  # avail 50 GW
        *[make_obs(sid, hist_days[::7], 200000.0, pd.Timedelta(days=7), "MWh")
          for sid in ("power.no.hydro_reservoir_mwh", "power.se.hydro_reservoir_mwh",
                      "power.ch.hydro_reservoir_mwh", "power.at.hydro_reservoir_mwh",
                      "power.fr.hydro_reservoir_mwh")],
        _prices(hist_days[-40:]),
    ], ignore_index=True)
    issue = pd.date_range("2024-01-01", "2024-01-20", freq="D")
    fc = pd.concat([
        make_fc("power.de.load_fc_gw", issue, range(1, 3), lambda i, ld: 60.0 + i.day),
        make_fc("power.fr.outage_nuclear_unavail_mw", issue, range(1, 30),
                lambda i, ld: 15000.0 if i.day < 12 else 18000.0, unit="MW"),  # avail 48 / 45 GW
    ], ignore_index=True)
    return bundle(obs, fc)


class TestBuilder:
    def test_features_with_synthetic_efficiencies(self, power_bundle) -> None:  # type: ignore[no-untyped-def]
        f = build_power_features(power_bundle, O, {"economics": EFF})
        assert all(c.startswith("power__") for c in f.columns)
        assert np.allclose(f["power__de_wind_fc_anom_gw"], 10.0)  # 30 vs 20 normal
        assert np.allclose(f["power__de_solar_fc_anom_gw"], -1.0)
        assert np.allclose(f["power__de_residual_load_rev_gw"], 1.0)  # load +1 GW per day
        assert np.allclose(f["power__fr_nuclear_avail_anom_gw"], -5.0)
        assert f["power__hydro_reservoir_anom_pct_alp_nord"].abs().max() < 1e-9
        coal = 120 / 1.2 / 6.978
        # gas emission factor on a GCV basis since D-034 (0.202 NCV x 0.901 = 0.182)
        sw = switching_gas_price(coal, 70.0, 0.5, 0.4, 0.182, 0.3406)
        assert np.allclose(f["power__gas_minus_switch_eur_mwh"], 30.0 - sw)
        assert f.attrs["economics_status"]["gas_plant_efficiency"] == "params_override"
        assert f.attrs["pit_grade"]["power__de_wind_fc_anom_gw"] == "B"
        assert f.attrs["pit_grade"]["power__clean_spark_eur_mwh"] == "A"

    def test_nuclear_revision_vs_one_week_ago(self, power_bundle) -> None:  # type: ignore[no-untyped-def]
        o = origins("2024-01-15", "2024-01-15")
        f = build_power_features(power_bundle, o, {"economics": EFF})
        assert f["power__fr_nuclear_avail_rev_gw"].iloc[0] == pytest.approx(-3.0)

    def test_missing_efficiencies_give_nan_and_flag(self, power_bundle, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # D-034 set assumed efficiencies in configs/; test the null path on a temp config copy.
        import shutil

        import yaml

        from ttf_forecaster.settings import CONFIG_DIR, load_config

        cfg_dir = tmp_path / "configs"
        shutil.copytree(CONFIG_DIR, cfg_dir, ignore=shutil.ignore_patterns("local*"))
        pfile = cfg_dir / "sources" / "power.yaml"
        data = yaml.safe_load(pfile.read_text(encoding="utf-8"))
        for k in ("gas_plant_efficiency", "coal_plant_efficiency"):
            data["economics"][k]["value"] = None
            data["economics"][k]["status"] = "requires_user_input"
        pfile.write_text(yaml.safe_dump(data), encoding="utf-8")
        load_config.cache_clear()
        f = build_power_features(power_bundle, O, {"config_dir": str(cfg_dir)})
        assert f["power__clean_spark_eur_mwh"].isna().all()
        reason = next(iter(f.attrs["flags"]["power__clean_spark_eur_mwh"].values()))
        assert reason.startswith("requires_user_input:gas_plant_efficiency")
        assert f["power__de_wind_fc_gw"].notna().all()  # other features unaffected

    def test_leakage_canary(self, power_bundle) -> None:  # type: ignore[no-untyped-def]
        o = origins("2024-01-16", "2024-01-16")
        cut = o[0].tz_convert("UTC")
        b2 = bundle(power_bundle.observations.copy(), power_bundle.forecasts.copy())
        b2.observations.loc[b2.observations["publication_time"] > cut, "value"] *= 3.0
        b2.forecasts.loc[b2.forecasts["publication_time"] > cut, "value"] *= 3.0
        pd.testing.assert_frame_equal(build_power_features(power_bundle, o, {"economics": EFF}),
                                      build_power_features(b2, o, {"economics": EFF}))

    def test_same_session_prices_not_used(self, power_bundle) -> None:  # type: ignore[no-untyped-def]
        o = origins("2024-01-16", "2024-01-16")
        obs = power_bundle.observations.copy()
        same_day = (obs["series_id"] == "price.ttf.m1_eur_mwh") & (
            obs["event_time"] == pd.Timestamp("2024-01-16", tz="UTC"))
        obs.loc[same_day, "value"] = 500.0  # published 2024-01-17 18:00 UTC -> invisible
        a = build_power_features(power_bundle, o, {"economics": EFF})
        b = build_power_features(bundle(obs, power_bundle.forecasts), o, {"economics": EFF})
        pd.testing.assert_frame_equal(a, b)
