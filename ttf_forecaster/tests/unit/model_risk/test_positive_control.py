"""Positive control for spec §8 criterion 3 (D-026). SYNTHETIC: never evidence about TTF.

World: a WHITE, storage-only driver (standardised first-release inventory change, published
D+2 07:00) moves the next session's TTF return; prices carry no memory of it. A correct harness
must find that the full model beats its price-only twin, and must NOT find it in the null twin.

Sizing (beta = 0.02 -> ~30 % of daily variance at h = 1) is deliberately large: the control tests
the harness, not power. Power is reported separately in the model-risk report (at beta = 0.01 a
2.5-year window FAILS criterion 3 although a storage-only OLS finds the signal).
"""

from __future__ import annotations

import numpy as np
import pytest

from ttf_forecaster.evaluation import diagnostics as d


def _exp(w: d.ControlWorld) -> d.Experiment:
    return d.Experiment(w.X, w.targets, w.horizon, d.control_split_config(), "ridge")


@pytest.fixture(scope="module")
def planted_11() -> d.ControlWorld:
    return d.positive_control_world(beta=0.02, seed=11)


@pytest.fixture(scope="module")
def null_11() -> d.ControlWorld:
    return d.positive_control_world(beta=0.0, seed=11)


def test_driver_is_white_and_storage_only(planted_11: d.ControlWorld) -> None:
    """The planted information is in storage, not in past returns: in-sample correlation of the
    target with storage__net_injection is strong, with lagged returns negligible."""
    y = planted_11.targets.set_index("origin_time")["value"]
    X = planted_11.X.loc[y.index]
    x = X["storage__net_injection_gwhd"]
    assert x.autocorr(1) < 0.15                                   # white driver
    assert abs(x.corr(y)) > 0.3
    assert abs(X["price__ret_1d"].corr(y)) < 0.1


def test_positive_control_passes_criterion_3(planted_11: d.ControlWorld) -> None:
    r = d.price_only_test(_exp(planted_11))
    assert r["passed"] and r["dm_p"] < 0.01
    assert r["oos_r2_large"] > 0.05 > r["oos_r2_small"]


def test_second_seed_passes_criterion_3() -> None:
    r = d.price_only_test(_exp(d.positive_control_world(beta=0.02, seed=12)))
    assert r["passed"]


def test_null_twin_does_not_pass(null_11: d.ControlWorld) -> None:
    r = d.price_only_test(_exp(null_11))
    assert not r["passed"]


def test_planted_sign_is_stable_across_folds(planted_11: d.ControlWorld) -> None:
    s = d.coefficient_stability(_exp(planted_11))
    row = s.loc["storage__net_injection_gwhd"]
    assert row["sign_agreement"] == 1.0 and row["share_positive"] == 0.0   # beta enters with -


def test_ablation_attributes_skill_to_storage(planted_11: d.ControlWorld) -> None:
    tab = d.family_ablation(_exp(planted_11), ["storage", "curve"], modes=("drop",))
    drop = tab.loc[tab["mode"] == "drop"].set_index("family")
    assert drop.loc["storage", "delta_oos_r2"] < -0.05 and drop.loc["storage", "dm_p"] < 0.01
    assert drop.loc["curve", "dm_p"] > 0.01


def test_full_model_not_catastrophic_in_short_window_seed_13() -> None:
    r = d.price_only_test(_exp(d.positive_control_world(beta=0.02, seed=13)))
    assert np.isfinite(r["oos_r2_large"]) and r["oos_r2_large"] > -0.25
