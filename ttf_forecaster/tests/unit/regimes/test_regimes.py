"""Tests for models/regimes.py. All data is SYNTHETIC (ttf_forecaster.data.synthetic)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    FORECAST_SCHEMA,
    DataBundle,
    PointInTimeViolation,
    empty_frame,
)
from ttf_forecaster.models import regimes as R

# Shorter windows so ~3 synthetic years give labels and Markov fits quickly.
FAST = {
    "inputs": {
        "price_percentile": {"window_trading_days": 500, "min_periods": 120},
        "realised_vol": {"percentile_window_trading_days": 500, "percentile_min_periods": 120},
        "storage": {"min_prior_gas_years": 1},
        "m1m2": {"min_prior_gas_years": 1},
    },
    "markov": {"min_train_obs": 150, "refit_every": 63, "max_iter": 50},
}


def _cfg(extra: dict | None = None) -> dict:
    return R.load_regime_config(R._deep_merge(FAST, extra or {}))


@pytest.fixture(scope="module")
def bundle() -> DataBundle:
    cfg = synthetic.SyntheticConfig(start="2021-01-04", end="2023-09-29")
    st = synthetic.storage_observations(cfg)
    cv = synthetic.ttf_curve(cfg, st)
    obs = pd.concat([st, _wgv_rows(st)], ignore_index=True)
    return DataBundle(obs, empty_frame(FORECAST_SCHEMA), cv, empty_frame(CONTRACT_METADATA_SCHEMA))


def _wgv_rows(storage: pd.DataFrame, value: float = 1100.0) -> pd.DataFrame:
    """SYNTHETIC constant working gas volume published alongside the first storage release."""
    w = storage.loc[storage["revision"] == 0].copy()
    w["series_id"] = "storage.eu.working_gas_volume_twh"
    w["value"] = value
    return w.reset_index(drop=True)


@pytest.fixture(scope="module")
def full_series(bundle: DataBundle) -> pd.DataFrame:
    from ttf_forecaster.settings import load_config

    load_config.cache_clear()
    return R.regime_label_series(bundle, _cfg())


def _truncate(b: DataBundle, t: pd.Timestamp) -> DataBundle:
    return DataBundle(
        b.observations.loc[b.observations["publication_time"] <= t].reset_index(drop=True),
        b.forecasts,
        b.curves.loc[b.curves["publication_time"] <= t].reset_index(drop=True),
        b.contracts,
    )


def _inputs(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 18:00", periods=len(rows), freq="D", tz="UTC")
    df = pd.DataFrame(
        rows, index=idx,
        columns=["storage_z", "price_percentile", "vol_percentile", "m1m2_dev"],
    )
    df["jkm_ttf"] = np.nan
    return df


# ---------------------------------------------------------------------------------------------
# Config and rule scoring
# ---------------------------------------------------------------------------------------------


class TestConfig:
    def test_loads_and_validates(self) -> None:
        cfg = R.load_regime_config()
        assert R._labels(cfg) == R.DEFAULT_LABELS
        assert cfg["evaluation_only"]["crisis_window"]["start"] == "2021-07-01"

    def test_unordered_thresholds_rejected(self) -> None:
        bad = {"rules": {"score_thresholds": {"tight_at_or_above": 6}}}
        with pytest.raises(R.RegimeConfigError, match="oversupplied < tight < crisis"):
            R.load_regime_config(bad)


class TestRules:
    def test_all_four_labels_reachable(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 1, "score_band": 0}}})
        inp = _inputs([
            (1.5, 0.10, 0.5, -0.05),   # -3 -> oversupplied
            (0.0, 0.50, 0.5, 0.00),    # 0 -> normal
            (-1.2, 0.85, 0.85, 0.0),   # +3 -> tight
            (-2.5, 0.97, 0.97, 0.05),  # +7 -> crisis
        ])
        out = R.classify_regimes(inp, cfg)
        assert list(out["label"]) == ["oversupplied", "normal", "tight", "crisis"]
        assert list(out["score"]) == [-3.0, 0.0, 3.0, 5.0]  # market stress capped at +2

    def test_min_indicators_gives_nan_not_guess(self) -> None:
        cfg = _cfg()
        inp = _inputs([(np.nan, 0.97, np.nan, np.nan)])
        out = R.classify_regimes(inp, cfg)
        assert np.isnan(out["score"].iloc[0]) and out["label"].iloc[0] is None

    def test_jkm_ignored_unless_enabled(self) -> None:
        inp = _inputs([(0.0, 0.5, 0.5, 0.0)])
        inp["jkm_ttf"] = -0.5
        assert "jkm_ttf" not in R.indicator_scores(inp, _cfg()).columns
        on = R.indicator_scores(inp, _cfg({"inputs": {"jkm_ttf": {"enabled": True}}}))
        assert on["jkm_ttf"].iloc[0] == 1.0


class TestHysteresis:
    def _score(self, values: list[float]) -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
        return pd.Series(values, index=idx, dtype=float)

    def test_flicker_suppressed_by_min_duration(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 3, "score_band": 0}}})
        s = self._score([0, 2, 0, 2, 0, 2, 2, 2, 2])
        codes = R.apply_hysteresis(s, cfg).tolist()
        # isolated tight days never published; switch on the 3rd consecutive day, not earlier
        assert codes == [1, 1, 1, 1, 1, 1, 1, 2, 2]

    def test_band_makes_exit_stickier_than_entry(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 1, "score_band": 1}}})
        s = self._score([0, 2, 1, 1, 0, 0])
        codes = R.apply_hysteresis(s, cfg).tolist()
        assert codes == [1, 2, 2, 2, 1, 1]  # entered at 2, held at 1, left at 0

    def test_causal_prefix_invariance(self) -> None:
        cfg = _cfg()
        rng = np.random.default_rng(3)
        s = self._score(list(rng.integers(-4, 7, 300).astype(float)))
        full = R.apply_hysteresis(s, cfg)
        for n in (17, 120, 299):
            pd.testing.assert_series_equal(R.apply_hysteresis(s.iloc[:n], cfg), full.iloc[:n])

    def test_nan_publishes_nan_and_keeps_state(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 1, "score_band": 0}}})
        codes = R.apply_hysteresis(self._score([3, np.nan, 3]), cfg)
        assert codes.iloc[0] == 2 and np.isnan(codes.iloc[1]) and codes.iloc[2] == 2


# ---------------------------------------------------------------------------------------------
# Bundle-derived inputs, coverage and causality
# ---------------------------------------------------------------------------------------------


class TestBundleSeries:
    def test_front_month_returns_never_cross_a_roll(self, bundle: DataBundle) -> None:
        panel = R.front_month_panel(bundle.curves, bundle.contracts, _cfg())
        rolled = panel["m1_contract"] != panel["m1_contract"].shift(1)
        # returns on roll days are on the NEW contract vs its own previous settle: in the
        # synthetic world that equals the common factor return, so no jump bigger than 8 sigma
        assert panel.loc[rolled, "m1_logret"].abs().max() < 0.25
        assert panel.index.is_monotonic_increasing and panel.index.tz is not None

    def test_label_coverage_on_synthetic(self, full_series: pd.DataFrame) -> None:
        after_warmup = full_series.iloc[150:]
        physical = after_warmup.loc[after_warmup["storage_z"].notna()]
        assert len(physical) > 0.4 * len(after_warmup)  # storage needs a prior gas year
        assert physical["label"].notna().mean() > 0.95
        assert after_warmup.loc[after_warmup["storage_z"].isna(), "label"].isna().all()
        assert set(after_warmup["label"].dropna()) <= set(R.DEFAULT_LABELS)

    def test_appending_future_data_does_not_change_past(
        self, bundle: DataBundle, full_series: pd.DataFrame
    ) -> None:
        cut = full_series.index[500]
        part = R.regime_label_series(_truncate(bundle, cut), _cfg())
        cols = ["score", "code", "storage_z", "price_percentile", "vol_percentile",
                "ms_p_high_vol", "cp_cusum_var", "cp_days_since_alarm"]
        pd.testing.assert_frame_equal(part[cols], full_series.loc[:cut, cols], check_exact=False,
                                      rtol=1e-10, atol=1e-12)

    def test_as_of_argument_equals_truncation(
        self, bundle: DataBundle, full_series: pd.DataFrame
    ) -> None:
        cut = full_series.index[400]
        part = R.regime_label_series(bundle, _cfg(), as_of=cut, include_markov=False)
        pd.testing.assert_series_equal(part["code"], full_series.loc[:cut, "code"])


# ---------------------------------------------------------------------------------------------
# Markov switching: filtered vs smoothed
# ---------------------------------------------------------------------------------------------


def _switching_returns(n: int = 500, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.97
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    r = rng.normal(0, np.where(state == 1, 0.06, 0.015))
    return pd.Series(r, index=pd.date_range("2022-01-03 18:00", periods=n, freq="B", tz="UTC"))


class TestMarkov:
    def test_filtered_is_causal(self) -> None:
        cfg = _cfg()
        y = _switching_returns()
        full = R.markov_filtered_probabilities(y, cfg)
        part = R.markov_filtered_probabilities(y.iloc[:333], cfg)
        np.testing.assert_allclose(part["ms_p_high_vol"].to_numpy(),
                                   full["ms_p_high_vol"].iloc[:333].to_numpy(), rtol=1e-9,
                                   atol=1e-12)
        assert full["ms_p_high_vol"].iloc[:149].isna().all()
        assert full["ms_p_high_vol"].iloc[149:].between(-1e-9, 1 + 1e-9).all()

    def test_smoothed_differs_from_filtered(self) -> None:
        cfg = _cfg()
        y = _switching_returns()
        diag = R.markov_smoothed_probabilities_diagnostic(y, cfg)
        gap = (diag["diag_smoothed_p_high_vol"] - diag["diag_insample_filtered_p_high_vol"]).abs()
        assert gap.max() > 0.05

    def test_feature_builder_uses_filtered_only(
        self, bundle: DataBundle, full_series: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise AssertionError("smoothed diagnostic reached the feature path")

        monkeypatch.setattr(R, "markov_smoothed_probabilities_diagnostic", _boom)
        origins = full_series.index[[300, 450, 600]] + pd.Timedelta(hours=2)
        feats = R.build_regime_features(bundle, origins, {"overrides": FAST})
        expected = full_series["ms_p_high_vol"].iloc[[300, 450, 600]].to_numpy()
        np.testing.assert_allclose(feats["regime__ms_p_high_vol"].to_numpy(), expected,
                                   rtol=1e-9)

    def test_forward_filter_matches_brute_force(self) -> None:
        rng = np.random.default_rng(0)
        k, t_len = 2, 5
        le = rng.normal(size=(t_len, k))
        start = np.array([0.6, 0.4])
        trans = np.array([[0.9, 0.1], [0.2, 0.8]])
        filt = R.forward_filter(le, start, trans)
        for t in range(t_len):
            joint = np.zeros(k)
            for path in itertools.product(range(k), repeat=t + 1):
                p = start[path[0]] * np.exp(le[0, path[0]])
                for s in range(1, t + 1):
                    p *= trans[path[s - 1], path[s]] * np.exp(le[s, path[s]])
                joint[path[-1]] += p
            np.testing.assert_allclose(filt[t], joint / joint.sum(), rtol=1e-10)

    def test_hmm_missing_dependency_is_explicit(self) -> None:
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            with pytest.raises(R.OptionalDependencyMissing, match="hmmlearn"):
                R.hmm_filtered_probabilities(_switching_returns(), _cfg())
        else:  # pragma: no cover
            pytest.skip("hmmlearn installed")


# ---------------------------------------------------------------------------------------------
# Change points
# ---------------------------------------------------------------------------------------------


class TestCusum:
    def test_variance_alarm_fires_after_shift_not_before(self) -> None:
        rng = np.random.default_rng(5)
        r = np.concatenate([rng.normal(0, 0.01, 300), rng.normal(0, 0.05, 100)])
        idx = pd.date_range("2023-01-02", periods=400, freq="B", tz="UTC")
        out = R.cusum_alarms(pd.Series(r, index=idx), _cfg())
        alarms = np.flatnonzero(out["alarm_var_up"].to_numpy())
        post = alarms[alarms >= 300]
        assert post.size and post[0] < 315
        assert out["alarm_var_up"].iloc[:300].sum() <= 2  # in-control ARL0 ~330 days

    def test_prefix_invariance(self) -> None:
        y = _switching_returns(400)
        full = R.cusum_alarms(y, _cfg())
        pd.testing.assert_frame_equal(R.cusum_alarms(y.iloc[:250], _cfg()), full.iloc[:250])


# ---------------------------------------------------------------------------------------------
# Transitions, evaluation-only labels, feature builder contract
# ---------------------------------------------------------------------------------------------


class TestOutputs:
    def test_transition_series(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
        codes = pd.Series([1, 1, 2, 2, np.nan, 1], index=idx)
        tr = R.regime_transition_series(codes, 2)
        assert tr["value"].tolist()[:2] == [1.0, 1.0]
        assert np.isnan(tr["value"].iloc[2])  # t+h unknown
        assert np.isnan(tr["value"].iloc[4]) and np.isnan(tr["value"].iloc[5])
        assert tr["label_known_time"].iloc[0] == idx[2]
        with pytest.raises(ValueError):
            R.regime_transition_series(codes, 0)

    def test_evaluation_only_window_inclusive(self) -> None:
        idx = pd.DatetimeIndex(
            ["2021-06-30 17:00", "2021-07-01 17:00", "2023-06-30 17:00", "2023-07-03 17:00"],
            tz="UTC",
        )
        lab = R.evaluation_only_period_labels(idx, R.load_regime_config())
        assert list(lab.columns) == ["eval_crisis_2021_07_2023_06"]
        assert lab.iloc[:, 0].tolist() == [False, True, True, False]

    def test_feature_contract(self, bundle: DataBundle, full_series: pd.DataFrame) -> None:
        origins = pd.DatetimeIndex(
            list(full_series.index[[5, 300, 600]] + pd.Timedelta(hours=1))
            + [full_series.index[-1] + pd.Timedelta(days=20)]
        )
        feats = R.build_regime_features(bundle, origins, {"overrides": FAST})
        assert feats.index.equals(origins) and feats.index.name == "origin_time"
        assert all(c.startswith("regime__") for c in feats.columns)
        assert not any("eval" in c for c in feats.columns)
        assert set(feats.attrs["pit_grade"]) == set(feats.columns)
        assert feats.attrs["pit_grade"]["regime__code"] == "B"  # storage is grade B
        assert feats.attrs["pit_grade"]["regime__ms_p_high_vol"] == "A"
        assert feats.iloc[0].isna()["regime__code"]  # warm-up -> NaN, never 0
        assert feats.iloc[3].isna().all()            # 20 days stale -> NaN
        assert feats.loc[origins[1], "regime__code"] == full_series["code"].iloc[300]
        onehot = feats.filter(like="__is_").iloc[1:3]
        assert (onehot.sum(axis=1) == 1).all()

    def test_leakage_canary(self, bundle: DataBundle, full_series: pd.DataFrame) -> None:
        origins = full_series.index[[350, 500]] + pd.Timedelta(hours=1)
        base = R.build_regime_features(bundle, origins, {"overrides": FAST,
                                                          "include_markov": False})
        t = origins.max()
        obs = bundle.observations.copy()
        cv = bundle.curves.copy()
        obs.loc[obs["publication_time"] > t, "value"] *= 3.0
        cv.loc[cv["publication_time"] > t, "value"] *= 0.2
        pert = DataBundle(obs, bundle.forecasts, cv, bundle.contracts)
        out = R.build_regime_features(pert, origins, {"overrides": FAST, "include_markov": False})
        pd.testing.assert_frame_equal(base, out)

    def test_naive_origins_and_eval_inputs_rejected(self, bundle: DataBundle) -> None:
        with pytest.raises(PointInTimeViolation):
            R.build_regime_features(bundle, pd.DatetimeIndex(["2022-01-03"]), {})
        inp = _inputs([(0.0, 0.5, 0.5, 0.0)])
        inp["eval_crisis_2021_07_2023_06"] = True
        with pytest.raises(PointInTimeViolation, match="evaluation-only"):
            R.build_regime_features(
                bundle, inp.index, {"input_frame": inp, "input_pit_grades": {},
                                    "include_markov": False, "include_changepoint": False},
            )

    def test_input_frame_path(self, bundle: DataBundle) -> None:
        inp = _inputs([(-2.5, 0.97, 0.97, 0.05)] * 5)
        feats = R.build_regime_features(
            bundle, inp.index + pd.Timedelta(hours=1),
            {"input_frame": inp, "input_pit_grades": {c: "A" for c in inp.columns},
             "include_markov": False, "include_changepoint": False},
        )
        assert (feats["regime__is_crisis"] == 1.0).all()
        assert set(feats.attrs["pit_grade"].values()) == {"A"}


# ---------------------------------------------------------------------------------------------
# Gas-fundamentals review changes (2026-10-17)
# ---------------------------------------------------------------------------------------------


def _storage_frame(days: pd.DatetimeIndex, inv: np.ndarray, wgv: float | None) -> pd.DataFrame:
    """SYNTHETIC inventory (+ optional WGV) observations, published D+2 07:00 UTC, grade B."""
    rows = []
    for sid, vals in (("storage.eu.inventory_twh", inv),
                      ("storage.eu.working_gas_volume_twh",
                       None if wgv is None else np.full(len(days), wgv))):
        if vals is None:
            continue
        ev = days.tz_localize("Europe/Amsterdam").tz_convert("UTC")
        pub = ev + pd.Timedelta(days=2, hours=7)
        n = len(days)
        rows.append(pd.DataFrame({
            "series_id": sid, "event_time": ev, "value": np.asarray(vals, dtype=float),
            "geography": "EU", "publication_time": pub, "ingestion_time": pub,
            "revision": np.zeros(n, dtype="int64"), "unit": "TWh", "source": "SYNTHETIC",
            "quality_flag": "synthetic", "entitlement": "synthetic", "pit_grade": "B",
            "publication_time_estimated": np.ones(n, dtype=bool),
            "source_timezone": "Europe/Amsterdam",
        }))
    return pd.concat(rows, ignore_index=True)


class TestReviewChanges:
    def test_config_review_values(self) -> None:
        cfg = R.load_regime_config()
        st = cfg["inputs"]["storage"]
        assert st["min_prior_gas_years"] == 3 and st["min_scale_pct_pts"] == 2.5
        assert "min_scale_twh" not in st
        assert cfg["inputs"]["m1m2"]["min_prior_gas_years"] == 3
        assert cfg["inputs"]["price_percentile"]["window_trading_days"] == 504
        assert cfg["rules"]["require_physical_for"] == ["tight", "crisis"]
        assert cfg["rules"]["market_stress_cap"] == 2
        j = cfg["rules"]["indicators"]["jkm_ttf"]
        assert j["tight_at_or_below"] == 0.0 and j["loose_at_or_above"] == 1.0
        assert cfg["inputs"]["input_frame_aliases"]["jkm_ttf"] == "lng__arb_nea_minus_nwe_usd_mmbtu"

    def test_storage_z_is_in_pct_of_wgv(self) -> None:
        days = pd.date_range("2019-10-01", "2023-12-31", freq="D")
        rng = np.random.default_rng(2)
        inv = 600 + 300 * np.sin(2 * np.pi * days.dayofyear / 365.25) + rng.normal(0, 20, len(days))
        cfg = R.load_regime_config()
        times = pd.DatetimeIndex(["2023-03-15 18:00", "2023-11-20 18:00"], tz="UTC")
        a = R.storage_seasonal_z(_storage_frame(days, inv, 1100.0), times, cfg)
        b = R.storage_seasonal_z(_storage_frame(days, 2 * inv, 2200.0), times, cfg)
        pd.testing.assert_series_equal(a["storage_z"], b["storage_z"])  # unit-free
        assert a["storage_fill_pct"].between(0, 100).all()
        np.testing.assert_allclose(a["storage_dev_pct_pts"],
                                   a["storage_fill_pct"] - a["storage_normal_pct"])
        no_wgv = R.storage_seasonal_z(_storage_frame(days, inv, None), times, cfg)
        assert no_wgv["storage_z"].isna().all()  # no TWh fallback

    def test_storage_needs_three_prior_gas_years(self) -> None:
        days = pd.date_range("2020-10-01", "2023-12-31", freq="D")
        inv = np.full(len(days), 700.0)
        cfg = R.load_regime_config()
        t_two = pd.DatetimeIndex(["2022-11-20 18:00"], tz="UTC")    # GY2022: 2 prior years
        t_three = pd.DatetimeIndex(["2023-11-20 18:00"], tz="UTC")  # GY2023: 3 prior years
        obs = _storage_frame(days, inv, 1000.0)
        assert R.storage_seasonal_z(obs, t_two, cfg)["storage_z"].isna().all()
        assert R.storage_seasonal_z(obs, t_three, cfg)["storage_z"].notna().all()

    def test_gas_year_boundary_excludes_last_days_own_year(self) -> None:
        # Data from 1 Oct 2019 (GY2019) to 30 Sep 2023 (GY2022). On 2 Oct 2023 the latest
        # published day is 30 Sep 2023 (GY2022): GY2022 must NOT enter its own normal.
        days = pd.date_range("2019-10-01", "2023-09-30", freq="D")
        inv = np.full(len(days), 700.0)
        inv[days >= "2022-10-01"] = 950.0  # GY2022 is high: including it would move the normal
        cfg = R.load_regime_config({"inputs": {"storage": {"min_prior_gas_years": 1}}})
        tau = pd.DatetimeIndex(["2023-10-02 09:00"], tz="UTC")
        out = R.storage_seasonal_z(_storage_frame(days, inv, 1000.0), tau, cfg)
        assert out["storage_n_years"].iloc[0] == 3  # GY2019, 2020, 2021
        assert out["storage_normal_pct"].iloc[0] == pytest.approx(70.0)
        assert out["storage_fill_pct"].iloc[0] == pytest.approx(95.0)

    def test_m1m2_deviation_from_same_month_prior_years(self) -> None:
        # Sep front (M2 = Oct, winter): a structural -8 % carry every year must score 0.
        dates = pd.DatetimeIndex(["2019-08-05", "2020-08-05", "2021-08-05", "2022-08-05",
                                  "2023-08-07"])
        back = pd.Series([-0.08, -0.08, -0.08, -0.08, -0.13])
        months = pd.Series([9.0] * 5)
        dev, normal = R.m1m2_seasonal_deviation(back, months, dates, min_years=3)
        assert np.isnan(dev[:3]).all()                 # fewer than 3 prior gas years
        assert dev[3] == pytest.approx(0.0)
        assert dev[4] == pytest.approx(-0.05)          # 5 pp more contango than normal
        assert normal[4] == pytest.approx(-0.08)

    def test_m1m2_dev_removes_season_carry_on_synthetic(self, full_series: pd.DataFrame) -> None:
        sep = full_series.loc[full_series["m1m2_backwardation"] < -0.05]
        assert len(sep) and sep["m1m2_dev"].dropna().abs().max() < 1e-9

    def test_curve_alone_cannot_reach_tight_or_crisis(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 1, "score_band": 0}}})
        inp = _inputs([(0.0, 0.99, 0.99, 0.05)] * 3)  # market stress + m1m2, storage normal
        out = R.classify_regimes(inp, cfg)
        assert (out["score"] == 3.0).all()            # 2 (capped, not 4) + 1
        assert (out["label"] == "normal").all() and (out["raw_code"] == 1.0).all()

    def test_no_physical_input_gives_nan(self) -> None:
        inp = _inputs([(np.nan, 0.5, 0.5, 0.0)])
        out = R.classify_regimes(inp, _cfg())
        assert np.isnan(out["score"].iloc[0]) and out["n_physical"].iloc[0] == 0

    def test_losing_physical_evidence_exits_tight_causally(self) -> None:
        cfg = _cfg({"rules": {"hysteresis": {"min_duration": 2, "score_band": 1}}})
        inp = _inputs([(-1.5, 0.9, 0.9, 0.0)] * 3 + [(0.0, 0.9, 0.9, 0.0)] * 3)
        codes = R.classify_regimes(inp, cfg)["code"].tolist()
        assert codes == [2, 2, 2, 2, 1, 1]  # cap -> normal, confirmed on the 2nd day
        part = R.classify_regimes(inp.iloc[:4], cfg)["code"].tolist()
        assert part == codes[:4]

    def test_arb_is_physical_and_aliased(self, bundle: DataBundle) -> None:
        on = {"inputs": {"jkm_ttf": {"enabled": True}}}
        inp = _inputs([(np.nan, 0.9, 0.9, 0.0), (np.nan, 0.5, 0.5, 0.0)])
        inp["jkm_ttf"] = [-0.2, 1.3]
        out = R.classify_regimes(inp, _cfg(R._deep_merge(
            on, {"rules": {"hysteresis": {"min_duration": 1, "score_band": 0}}})))
        assert out["label"].tolist() == ["tight", "normal"]  # arb alone is physical evidence
        frame = inp.drop(columns="jkm_ttf").rename(columns={})
        frame["lng__arb_nea_minus_nwe_usd_mmbtu"] = [-0.2, 1.3]
        feats = R.build_regime_features(
            bundle, frame.index + pd.Timedelta(hours=1),
            {"input_frame": frame, "overrides": R._deep_merge(FAST, R._deep_merge(on, {
                "rules": {"hysteresis": {"min_duration": 1, "score_band": 0}}})),
             "input_pit_grades": {c: "A" for c in frame.columns},
             "include_markov": False, "include_changepoint": False},
        )
        assert feats["regime__is_tight"].tolist() == [1.0, 0.0]
        assert feats.attrs["pit_grade"]["regime__code"] == "A"
