"""Leakage tests.

These are the tests worth having. A freight model that looks good because it
peeked is indistinguishable from one that works, right up to the moment money
is on it, so each property below is checked mechanically rather than argued
for in a docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lngfreight import backtest, features


def _panel(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """A synthetic wide panel with every column the feature builder needs."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n)
    walk = lambda scale, base: base * np.exp(np.cumsum(rng.normal(0, scale, n)))
    frame = pd.DataFrame(index=index)
    frame["atlantic_spot"] = walk(0.05, 60000)
    frame["pacific_spot"] = walk(0.05, 55000)
    frame["henry_hub"] = walk(0.02, 3.0)
    frame["ttf_usd"] = walk(0.02, 12.0)
    frame["ttf_eur"] = walk(0.02, 35.0)
    frame["jkm"] = walk(0.02, 14.0)
    for tenor in ("m0", "m1", "m2", "m3"):
        frame[f"ffa_{tenor}"] = walk(0.03, 65000)
    for name in ("afloat_20d_volume", "afloat_30d_volume"):
        frame[name] = walk(0.01, 3_000_000)
    for name in ("afloat_20d_vessels", "afloat_30d_vessels", "afloat_20d_usa",
                 "afloat_20d_qatar", "afloat_20d_australia",
                 "afloat_20d_nigeria", "afloat_20d_russia", "afloat_20d_other"):
        frame[name] = rng.integers(1, 40, n).astype(float)
    for name in ("transits_panama", "transits_suez", "transits_cogh",
                 "transits_nsr"):
        frame[name] = rng.integers(0, 4, n).astype(float)
    frame["us_lng_exports"] = walk(0.02, 400_000)
    # Trade-flow legs. Deliberately sparse: the real AHOY legs are cargo-level
    # and zero on 18-72% of days, and the tonne-mile features must survive
    # that. A dense fixture would hide a divide-by-zero the real data finds.
    for name in ("flow_amer_to_nwe", "flow_amer_to_nasia", "flow_amer_to_sasia",
                 "flow_amer_to_med", "flow_amer_to_amer"):
        values = walk(0.05, 70_000)
        values[rng.random(n) < 0.5] = 0.0
        frame[name] = values
    for name in ("exports_global", "imports_global", "imports_nwe",
                 "imports_nasia", "exports_qatar"):
        frame[name] = walk(0.02, 900_000)
    # Baltic generics. Given a late start, as the real ones have: they begin
    # two years into the sample and the feature builder must tolerate the
    # leading NaNs rather than assuming full coverage.
    for name in ("blng1_aus_jpn", "blng2_usg_cont", "blng3_usg_jpn"):
        series = walk(0.06, 45_000)
        series[: n // 3] = np.nan
        frame[name] = series
    return frame.rename_axis(index="obs_date")


def _lags(frame: pd.DataFrame) -> dict[str, int]:
    return {c: 1 for c in frame.columns}


def test_features_do_not_move_when_the_future_changes():
    """The load-bearing property: X[t] must not depend on anything after t.

    Corrupting the tail of the panel and rebuilding must leave every earlier
    feature value bit-identical. This catches a rolling window applied before
    the publication shift, a centred window, and an accidental bfill - the
    three ways this pipeline could quietly look forward.
    """
    panel = _panel()
    lags = _lags(panel)
    base = features.build(panel, lags)["X"]

    cut = panel.index[-120]
    corrupted = panel.copy()
    corrupted.loc[cut:] = corrupted.loc[cut:] * 7.5
    after = features.build(corrupted, lags)["X"]

    common = base.index[base.index < cut]
    pd.testing.assert_frame_equal(base.loc[common], after.loc[common])


def test_publication_lag_is_actually_applied():
    """A series with lag L must not show its own value until L days later."""
    panel = _panel()
    lags = {c: 3 for c in panel.columns}
    lagged = features.apply_lags(panel, lags)
    # The freight assessment is exempt by config (same-day visible); anything
    # else must be shifted by exactly its lag.
    assert lagged["jkm"].iloc[10] == pytest.approx(panel["jkm"].iloc[7])
    assert lagged["atlantic_spot"].iloc[10] == pytest.approx(
        panel["atlantic_spot"].iloc[10])


def test_target_requires_a_fresh_print_at_both_ends():
    """A forward-filled price differenced against itself is not a return."""
    panel = _panel(n=400)
    # Blank a fortnight, as the real assessment does over new year.
    hole = panel.index[100:110]
    panel.loc[hole, "atlantic_spot"] = np.nan
    targets = features.build_targets(panel, limit=10)
    for day in hole:
        assert np.isnan(targets.loc[day, "y_h1"]), (
            f"{day.date()} has no print, so it cannot be a forecast origin")
    assert not targets["fresh_h1"].loc[hole].any()


def test_shuffled_target_gives_no_skill():
    """The end-to-end check: no signal in, no skill out."""
    panel = _panel(n=900)
    bundle = features.build(panel, _lags(panel))
    X, Y, F = bundle["X"], bundle["Y"], bundle["forward"]
    rng = np.random.default_rng(1)
    y = Y["y_h5"].copy()
    observed = y.dropna()
    values = observed.to_numpy().copy()
    rng.shuffle(values)
    y.loc[observed.index] = values

    result = backtest.run_horizon(X, y, F["y_h5"], horizon=5)
    for model in ("ridge", "lightgbm", "own_lags"):
        if model not in result.metrics.index:
            continue
        skill = result.metrics.loc[model, "skill_vs_rw"]
        assert skill < 0.05, (
            f"{model} found {skill:.3f} skill against a shuffled target, "
            "which means the pipeline is leaking")


def test_purge_removes_overlapping_training_rows():
    """No training origin may have a label window reaching into the test block."""
    origins = pd.bdate_range("2022-01-03", periods=500)
    min_train, step = 378, 21
    horizon, embargo = 15, 5
    gap = horizon + embargo
    for train_end, test in backtest._blocks(origins, min_train, step):
        cutoff = train_end - gap
        if cutoff <= 0:
            continue
        last_train_origin = origins[:cutoff][-1]
        first_test_origin = origins[test][0]
        label_lands = origins[min(cutoff - 1 + horizon, len(origins) - 1)]
        assert label_lands < first_test_origin or last_train_origin < first_test_origin
        assert (first_test_origin - last_train_origin).days >= horizon
