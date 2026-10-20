"""The model-risk harness on SYNTHETIC toy worlds: ablation, shift test, stability, concentration,
sensitivity, execution grid, label-overlap audits. Nothing here is evidence about TTF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.evaluation import diagnostics as d
from ttf_forecaster.evaluation.metrics import paired_frame

from .conftest import toy_split, toy_world

RIDGE = {"alphas": [1e-2, 1.0, 100.0], "cv_splits": 2}


def _exp(X: pd.DataFrame, t: pd.DataFrame, h: int = 1) -> d.Experiment:
    return d.Experiment(X, t, h, toy_split(), "ridge", dict(RIDGE))


class TestAblation:
    def test_planted_family_is_the_one_that_matters(self, planted) -> None:  # type: ignore[no-untyped-def]
        tab = d.family_ablation(_exp(*planted), ["storage", "weather", "lng", "text"])
        drop = tab.loc[tab["mode"] == "drop"].set_index("family")
        add = tab.loc[tab["mode"] == "add"].set_index("family")
        # dropping the planted family costs skill and the full model beats the reduced one
        assert drop.loc["storage", "delta_oos_r2"] < -0.05
        assert drop.loc["storage", "dm_p"] < 0.01
        # dropping a noise family costs nothing significant
        assert abs(drop.loc["weather", "delta_oos_r2"]) < 0.02
        assert drop.loc["weather", "dm_p"] > 0.05
        # add-one to price-only: storage passes criterion 3, noise does not
        assert add.loc["storage", "dm_p"] < 0.01 and add.loc["lng", "dm_p"] > 0.05
        # an absent family is reported, not silently skipped
        assert drop.loc["text", "n_cols"] == 0 and np.isnan(drop.loc["text", "dm_p"])

    def test_null_world_nothing_significant(self, null) -> None:  # type: ignore[no-untyped-def]
        tab = d.family_ablation(_exp(*null), ["storage", "weather"], modes=("drop",))
        assert (tab.loc[tab["mode"] == "drop", "dm_p"] > 0.01).all()

    def test_price_only_test_positive_and_null(self, planted, null) -> None:  # type: ignore[no-untyped-def]
        assert d.price_only_test(_exp(*planted))["passed"]
        r0 = d.price_only_test(_exp(*null))
        assert not r0["passed"] or r0["dm_p"] > 0.01


class TestShift:
    def test_leaky_feature_flagged(self) -> None:
        X, t = toy_world(beta=0.0, seed=3, leak=True)
        out = d.feature_shift_test(_exp(X, t), ["flows__leak"])
        assert out.loc[0, "oos_r2"] > 0.5            # implausible for a return target
        assert out.loc[1, "oos_r2"] < 0.05           # collapses when lagged one session
        assert out.attrs["suspect_leak"] is True

    def test_weak_legit_feature_not_flagged(self) -> None:
        X, t = toy_world(beta=0.15, seed=4)
        out = d.feature_shift_test(_exp(X, t), ["storage__signal"])
        assert out.attrs["suspect_leak"] is False
        assert 0.0 < out.loc[0, "oos_r2"] < 0.05


class TestStability:
    def test_coefficient_sign_stability(self, planted) -> None:  # type: ignore[no-untyped-def]
        s = d.coefficient_stability(_exp(*planted))
        assert s.loc["storage__signal", "sign_agreement"] == 1.0
        assert s.loc["storage__signal", "share_positive"] == 1.0
        assert s.index[0] == "storage__signal"           # largest |coef| in every fold
        assert len(s.attrs["per_fold"]) == s.loc["storage__signal", "n_folds"]

    def test_skill_concentration_detects_one_year(self) -> None:
        idx = pd.date_range("2020-01-01", "2022-12-31", freq="B", tz="UTC")
        rng = np.random.default_rng(0)
        y = rng.normal(size=len(idx))
        model = np.where(idx.year == 2021, y * 0.8, rng.normal(scale=0.3, size=len(idx)))
        p = pd.DataFrame({"origin_time": idx, "fold_id": "f", "actual": y, "model": model,
                          "bench": 0.0})
        out = d.skill_concentration(p)
        assert out.attrs["oos_r2_all"] > 0
        assert out.attrs["fragile"] is True
        assert out.set_index("year").loc[2021, "gain_share"] > 1.0  # other years lose


class TestOverlapAudits:
    @pytest.mark.parametrize("h", [1, 5, 20])
    def test_t3_row_gap_equal_horizon_is_sufficient(self, h: int) -> None:
        _, t = toy_world(n=300, h=h)
        assert d.label_overlap_rows(t) == h - 1
        assert d.inner_gap_violations(t, gap=h) == 0
        assert d.inner_gap_violations(t, gap=h - 2) > 0 if h > 2 else True
        assert d.target_lag_violations(t, horizon=h, p=3) == 0

    def test_dropped_rows_keep_row_gap_conservative(self) -> None:
        """NaN labels removed from the middle (runner drops them): a row gap then spans MORE
        sessions, never fewer, so it stays safe."""
        _, t = toy_world(n=400, h=5)
        t = t.drop(index=range(100, 140)).reset_index(drop=True)
        assert d.inner_gap_violations(t, gap=5) == 0


class TestSensitivity:
    def test_label_delay_kills_a_one_session_signal(self, planted) -> None:  # type: ignore[no-untyped-def]
        """The toy signal moves only the next session's return: it is forecastable at the origin
        but not tradeable at the next settlement. The executable-label variant must show that."""
        tab = d.sensitivity_table(_exp(*planted), [d.variant_execution_delay(1)])
        base, delayed = tab.iloc[0], tab.iloc[1]
        assert base["oos_r2"] > 0.1 and base["dm_p"] < 0.01
        assert delayed["oos_r2"] < 0.02

    def test_rolling_window_variant_runs_on_same_targets(self, planted) -> None:  # type: ignore[no-untyped-def]
        v = d.variant_split("rolling_1y", scheme="rolling", rolling_window_years=1.0)
        tab = d.sensitivity_table(_exp(*planted), [v], price_families=None)
        assert list(tab["variant"]) == ["<base>", "rolling_1y"]
        assert tab["oos_r2"].notna().all()

    def test_delay_targets_shifts_label_and_known_time(self) -> None:
        _, t = toy_world(n=50, h=5)
        dt = d.delay_targets(t, 2)
        assert len(dt) == len(t) - 2
        assert dt["value"].iloc[0] == t["value"].iloc[2]
        assert dt["label_known_time"].iloc[0] == t["label_known_time"].iloc[2]
        assert dt["origin_time"].iloc[0] == t["origin_time"].iloc[0]

    def test_shift_publication_moves_only_selected_grades(self) -> None:
        from ttf_forecaster.data import synthetic
        from ttf_forecaster.data.base import DataBundle

        cfg = synthetic.SyntheticConfig(start="2022-01-03", end="2022-03-31")
        b = DataBundle(synthetic.storage_observations(cfg), synthetic.weather_forecast_runs(cfg),
                       synthetic.ttf_curve(cfg), DataBundle.empty().contracts)
        s = d.shift_publication(b, 3)
        delta = s.observations["publication_time"] - b.observations["publication_time"]
        assert (delta == pd.Timedelta(days=3)).all()          # storage is grade B
        assert s.forecasts["publication_time"].equals(b.forecasts["publication_time"])  # A
        assert s.curves["publication_time"].equals(b.curves["publication_time"])


class TestExecutionGrid:
    def test_cost_and_delay_grid_on_synthetic_curve(self) -> None:
        from ttf_forecaster.data import synthetic

        cfg = synthetic.SyntheticConfig(start="2022-01-03", end="2022-12-30", seed=3)
        curve = synthetic.ttf_curve(cfg)
        t = d.build_targets(curve, 5)
        fc = pd.Series(np.random.default_rng(0).normal(0, 0.01, len(t)),
                       index=pd.DatetimeIndex(t["origin_time"]))
        g = d.execution_cost_grid(fc, t, curve, delays=(1, 2), cost_multipliers=(0.0, 2.0),
                                  rebalance_every=5)
        assert len(g) == 4
        pnl = g.set_index(["execution_delay", "cost_multiplier"])
        key = [c for c in g.columns if "net" in c and "pnl" in c][0]
        assert pnl.loc[(1, 0.0), key] > pnl.loc[(1, 2.0), key]   # costs only subtract
        with pytest.raises(ValueError):
            d.execution_cost_grid(fc, t, curve, delays=(0,))


class TestNestedSize:
    def test_dm_size_not_inflated_and_cw_reported(self) -> None:
        r = d.nested_test_size(n_reps=20, n=360, k_extra=8, model="ridge",
                               model_params=RIDGE, seed=5)
        assert r["dm_rejection_rate"] <= 0.15
        assert 0.0 <= r["cw_rejection_rate"] <= 1.0


def test_paired_frame_from_run_experiment(planted) -> None:  # type: ignore[no-untyped-def]
    p = d.run_experiment(_exp(*planted))
    pf = paired_frame(p, "model", "zero")
    assert len(pf) > 300 and set(p["model_id"]) == {"model", "zero"}
