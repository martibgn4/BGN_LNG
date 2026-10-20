"""Data-quality checks. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.validation import (
    DataQualityError,
    QualityParams,
    assess_quality,
    check_coverage,
    check_revision_monotonicity,
    check_series_id_convention,
    check_staleness,
    check_unit_consistency,
    coverage_summary,
    find_duplicate_keys,
    flag_outliers_robust_z,
    unit_suffix_matches,
    unit_token,
)

from .helpers import daily_obs, obs_frame

OBS_KEY = ["series_id", "event_time", "revision", "publication_time"]


class TestParams:
    def test_from_config(self) -> None:
        p = QualityParams.from_config({"outlier_z_threshold": 4.0, "hard_checks": ["unit_inconsistency"]})
        assert p.outlier_z_threshold == 4.0 and p.hard_checks == ("unit_inconsistency",)
        assert QualityParams.from_config(None) == QualityParams()

    @pytest.mark.parametrize("cfg", [{"nope": 1}, {"outlier_transform": "log"},
                                     {"hard_checks": ["bogus"]},
                                     {"outlier_window": 5, "outlier_min_periods": 10}])
    def test_bad_config(self, cfg) -> None:
        with pytest.raises(ValueError):
            QualityParams.from_config(cfg)


class TestDuplicates:
    def test_exact_vs_conflicting(self) -> None:
        df = obs_frame([
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
            ("a.eu.x_twh", "2024-01-02", 1.0, "2024-01-03", 0),
            ("a.eu.x_twh", "2024-01-02", 2.0, "2024-01-03", 0),
            ("a.eu.x_twh", "2024-01-03", 1.0, "2024-01-04", 0),
        ])
        d = find_duplicate_keys(df, OBS_KEY, ("value", "unit"))
        assert len(d) == 2
        assert d.sort_values("event_time")["conflicting"].tolist() == [False, True]

    def test_nan_equals_nan(self) -> None:
        df = obs_frame([("a.eu.x_twh", "2024-01-01", np.nan, "2024-01-02", 0)] * 2)
        assert not find_duplicate_keys(df, OBS_KEY)["conflicting"].any()

    def test_empty(self) -> None:
        assert find_duplicate_keys(obs_frame([]), OBS_KEY).empty


class TestRevisions:
    def test_monotone_ok_and_gap_ok(self) -> None:
        df = obs_frame([
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
            ("a.eu.x_twh", "2024-01-01", 1.1, "2024-01-05", 2),  # gap 0 -> 2 is allowed
        ])
        assert check_revision_monotonicity(df, ["series_id", "event_time"]).empty

    def test_revision_published_before_predecessor(self) -> None:
        df = obs_frame([
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-05", 0),
            ("a.eu.x_twh", "2024-01-01", 1.1, "2024-01-03", 1),
        ])
        bad = check_revision_monotonicity(df, ["series_id", "event_time"])
        assert len(bad) == 1 and bad["revision"].iloc[0] == 1 and bad["prev_revision"].iloc[0] == 0

    def test_same_publication_time_is_violation(self) -> None:
        df = obs_frame([
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-05", 0),
            ("a.eu.x_twh", "2024-01-01", 1.1, "2024-01-05", 1),
        ])
        assert len(check_revision_monotonicity(df, ["series_id", "event_time"])) == 1

    def test_synthetic_storage_is_monotone(self, syn_storage) -> None:
        assert check_revision_monotonicity(syn_storage, ["series_id", "event_time"]).empty


class TestUnits:
    def test_mixed_units(self) -> None:
        df = pd.concat([daily_obs("a.eu.x_twh", "2024-01-01", 3),
                        daily_obs("a.eu.x_twh", "2024-01-04", 2, unit="GWh")])
        u = check_unit_consistency(df, ["series_id"], {"a.eu.x_twh": "TWh"})
        assert not u["consistent"].iloc[0] and u["units"].iloc[0] == ("GWh", "TWh")
        assert u["matches_expected"].iloc[0] is False

    def test_expected_match(self) -> None:
        u = check_unit_consistency(daily_obs("a.eu.x_twh", "2024-01-01", 3), ["series_id"],
                                   {"a.eu.x_twh": "TWh"})
        assert u["consistent"].iloc[0] and u["matches_expected"].iloc[0] is True


class TestStaleness:
    def test_fresh_and_stale(self) -> None:
        df = pd.concat([daily_obs("a.eu.x_twh", "2024-01-01", 10),
                        daily_obs("b.eu.y_twh", "2024-01-01", 3)])
        s = check_staleness(df, pd.Timestamp("2024-01-11 12:00", tz="UTC"), "D",
                            grace=pd.Timedelta(days=1)).set_index("series")
        assert not s.loc["a.eu.x_twh", "is_stale"] and s.loc["b.eu.y_twh", "is_stale"]

    def test_unpublished_rows_do_not_refresh(self) -> None:
        df = daily_obs("a.eu.x_twh", "2024-01-01", 10, lag_days=30)
        s = check_staleness(df, pd.Timestamp("2024-01-11", tz="UTC"), "D")
        assert s["is_stale"].iloc[0] and pd.isna(s["last_event_time"].iloc[0])

    def test_business_day_frequency_over_weekend(self) -> None:
        # Friday 2024-01-12 last event; Monday morning is within 3 business days
        df = obs_frame([("a.eu.x_twh", "2024-01-12", 1.0, "2024-01-12 19:00", 0)])
        s = check_staleness(df, pd.Timestamp("2024-01-15 08:00", tz="UTC"), "B")
        assert not s["is_stale"].iloc[0]

    def test_expected_but_absent_series(self) -> None:
        df = daily_obs("a.eu.x_twh", "2024-01-01", 3)
        s = check_staleness(df, pd.Timestamp("2024-01-04", tz="UTC"),
                            {"a.eu.x_twh": "D", "missing.eu.z_twh": "D"}).set_index("series")
        assert s.loc["missing.eu.z_twh", "is_stale"]

    def test_naive_as_of(self) -> None:
        with pytest.raises(ValueError):
            check_staleness(daily_obs("a.eu.x_twh", "2024-01-01", 3), "2024-01-04", "D")


class TestOutliers:
    def test_spike_flagged_not_dropped(self) -> None:
        df = daily_obs("a.eu.x_twh", "2024-01-01", 80, seed=3)
        df.loc[50, "value"] += 100.0
        df["quality_flag"] = "ok"
        out = flag_outliers_robust_z(df, set_quality_flag=True)
        assert len(out) == len(df)
        assert out.loc[50, "outlier_suspect"]
        assert out.loc[50, "quality_flag"] == "outlier_suspect"
        assert out["outlier_suspect"].sum() <= 3  # the spike and its reversal
        assert df["quality_flag"].eq("ok").all()  # input untouched

    def test_trailing_only(self) -> None:
        """Changing a future value must not change any earlier z-score (no hindsight)."""
        df = daily_obs("a.eu.x_twh", "2024-01-01", 80, seed=4)
        base = flag_outliers_robust_z(df)
        pert = df.copy()
        pert.loc[70:, "value"] *= 50
        after = flag_outliers_robust_z(pert)
        pd.testing.assert_series_equal(base.loc[:69, "robust_z"], after.loc[:69, "robust_z"])

    def test_min_periods(self) -> None:
        out = flag_outliers_robust_z(daily_obs("a.eu.x_twh", "2024-01-01", 15))
        assert out["robust_z"].isna().all() and not out["outlier_suspect"].any()

    def test_requires_single_vintage(self, syn_storage) -> None:
        with pytest.raises(ValueError, match="unique"):
            flag_outliers_robust_z(syn_storage)

    def test_empty(self) -> None:
        out = flag_outliers_robust_z(obs_frame([]))
        assert "robust_z" in out and out.empty

    def test_synthetic_flag_untouched(self) -> None:
        df = daily_obs("a.eu.x_twh", "2024-01-01", 80, seed=3)
        df.loc[50, "value"] += 100.0
        out = flag_outliers_robust_z(df, set_quality_flag=True)
        assert (out["quality_flag"] == "synthetic").all()


class TestCoverage:
    def test_gaps_local_dates(self) -> None:
        # Event times are local midnight Amsterdam (23:00 UTC previous day): coverage must use
        # local dates, not UTC dates.
        df = obs_frame([("a.eu.x_twh", d, 1.0, "2024-02-01", 0)
                        for d in ["2024-01-01", "2024-01-02", "2024-01-04"]],
                       tz="Europe/Amsterdam")
        gaps = check_coverage(df, pd.date_range("2024-01-01", "2024-01-05"))
        assert gaps["missing_date"].tolist() == [pd.Timestamp("2024-01-03")]
        gaps_all = check_coverage(df, pd.date_range("2024-01-01", "2024-01-05"),
                                  within_series_span=False)
        assert len(gaps_all) == 2

    def test_summary_and_absent_series(self) -> None:
        df = daily_obs("a.eu.x_twh", "2024-01-01", 4)
        cov, gaps = coverage_summary(df, pd.date_range("2024-01-01", "2024-01-04"),
                                     series_ids=["a.eu.x_twh", "b.eu.y_twh"])
        c = cov.set_index("series")
        assert c.loc["a.eu.x_twh", "coverage_ratio"] == 1.0
        assert c.loc["b.eu.y_twh", "n_missing"] == 4 and c.loc["b.eu.y_twh", "coverage_ratio"] == 0


class TestNaming:
    def test_convention(self) -> None:
        r = check_series_id_convention({"storage.eu.inventory_twh": "TWh",
                                        "price.ttf.da_eur_mwh": "EUR/MWh",
                                        "weather.eu.hdd": "degC_day",
                                        "BadTicker Index": "TWh"}).set_index("series_id")
        assert r.loc["storage.eu.inventory_twh", "unit_suffix_ok"]
        assert r.loc["price.ttf.da_eur_mwh", "unit_suffix_ok"]
        assert r.loc["weather.eu.hdd", "pattern_ok"] and not r.loc["weather.eu.hdd", "unit_suffix_ok"]
        assert not r.loc["BadTicker Index", "pattern_ok"]

    @pytest.mark.parametrize(("sid", "u", "ok"), [
        ("storage.eu.injection_gwhd", "GWh/d", True), ("storage.eu.injection_gwh_d", "GWh/d", True),
        ("fx.eurusd.spot_usd_per_eur", "USD/EUR", True), ("storage.eu.full_pct", "%", True),
        ("storage.eu.inventory_twh", "GWh", False)])
    def test_unit_suffix_variants(self, sid: str, u: str, ok: bool) -> None:
        assert unit_suffix_matches(sid, u) is ok

    @pytest.mark.parametrize(("u", "t"), [("EUR/MWh", "eur_mwh"), ("GWh/d", "gwh_d"), ("%", "pct"),
                                          ("USD/MMBtu", "usd_mmbtu")])
    def test_unit_token(self, u: str, t: str) -> None:
        assert unit_token(u) == t


class TestAssessQuality:
    def test_clean_synthetic_storage(self, syn_storage) -> None:
        cal = pd.date_range("2022-01-03", "2023-12-29", freq="D")
        res = assess_quality(syn_storage, "observation", as_of=pd.Timestamp("2024-01-02", tz="UTC"),
                             expected_frequency="D", staleness_grace=pd.Timedelta(days=2),
                             calendar=cal, expected_units={"storage.eu.inventory_twh": "TWh"})
        assert res.ok, res.hard_failures
        row = res.report.set_index("series_id").loc["storage.eu.inventory_twh"]
        assert row["max_revision"] == 1 and row["unit_consistent"]
        assert row["coverage_ratio"] == 1.0 and not row["is_stale"]
        assert row["is_synthetic"] and row["pit_grades"] == "B"

    def test_hard_failures_reported_and_raised(self) -> None:
        df = pd.concat([
            obs_frame([("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-05", 0),
                       ("a.eu.x_twh", "2024-01-01", 1.1, "2024-01-03", 1),
                       ("a.eu.x_twh", "2024-01-02", 1.0, "2024-01-05", 0),
                       ("a.eu.x_twh", "2024-01-02", 9.0, "2024-01-05", 0)]),
            obs_frame([("a.eu.x_twh", "2024-01-03", 1.0, "2024-01-05", 0)], unit="GWh"),
        ], ignore_index=True)
        res = assess_quality(df, "observation")
        assert not res.ok and len(res.hard_failures) == 3
        rep = res.report.set_index("series_id").loc["a.eu.x_twh"]
        assert rep["n_conflicting_keys"] == 1 and rep["n_revision_violations"] == 1
        with pytest.raises(DataQualityError, match="conflicting"):
            res.raise_if_failed()
        soft = assess_quality(df, "observation", params=QualityParams(hard_checks=()))
        assert soft.ok

    def test_forecast_and_curve_kinds(self, syn_weather, syn_curve) -> None:
        f = assess_quality(syn_weather, "forecast")
        assert f.ok and "weather.eu.hdd" in f.report["series_id"].tolist()
        assert any("naming convention" in w for w in f.warnings)  # weather.eu.hdd lacks unit
        c = assess_quality(syn_curve, "curve")
        assert c.ok and c.report["n_rows"].sum() == len(syn_curve)

    def test_empty_frame(self) -> None:
        res = assess_quality(obs_frame([]), "observation")
        assert res.ok and res.warnings

    def test_planted_world_also_clean(self) -> None:
        cfg = synthetic.SyntheticConfig(start="2023-01-02", end="2023-06-30", world="planted")
        assert assess_quality(synthetic.storage_observations(cfg), "observation").ok
