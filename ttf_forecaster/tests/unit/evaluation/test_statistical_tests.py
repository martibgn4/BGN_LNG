"""DM / CW size and power on simulated data (seeded), and the §8 criteria helper. SYNTHETIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.evaluation.statistical_tests import (
    clark_west,
    diebold_mariano,
    evaluate_success_criteria,
    newey_west_variance,
)

REPS = 400
T = 500


def _ma(rng: np.random.Generator, n: int, h: int) -> np.ndarray:
    """MA(h-1) errors with unit-ish variance (overlapping h-step forecast errors)."""
    e = rng.normal(size=n + h - 1)
    return np.convolve(e, np.ones(h), mode="valid") / np.sqrt(h)


@pytest.mark.parametrize("h", [1, 5])
def test_dm_size_equal_accuracy(h: int) -> None:
    rng = np.random.default_rng(123 + h)
    rej = 0
    for _ in range(REPS):
        r = diebold_mariano(_ma(rng, T, h), _ma(rng, T, h), h, alternative="less")
        rej += r.p_value < 0.05
    assert 0.02 <= rej / REPS <= 0.10


def test_dm_power_when_model_better() -> None:
    rng = np.random.default_rng(5)
    e_b = rng.normal(size=T)
    e_m = 0.8 * rng.normal(size=T)
    assert diebold_mariano(e_m, e_b, 1).p_value < 0.01


def test_cw_size_nested_null() -> None:
    """y is unpredictable; the large model estimates a useless slope recursively."""
    rng = np.random.default_rng(99)
    rej = 0
    for _ in range(REPS):
        n = T + 100
        x, y = rng.normal(size=n), rng.normal(size=n)
        sxy, sxx = np.cumsum(x * y), np.cumsum(x * x)
        b = np.r_[0.0, sxy[:-1] / sxx[:-1]]  # slope known before t
        f_large = (b * x)[100:]
        r = clark_west(y[100:], np.zeros(T), f_large, 1)
        rej += r.p_value < 0.05
    assert 0.01 <= rej / REPS <= 0.10


def test_cw_detects_weak_signal_dm_misses_more_often() -> None:
    """Nested comparison with a weak true signal: CW rejects at least as often as DM."""
    rng = np.random.default_rng(11)
    cw_rej = dm_rej = 0
    for _ in range(100):
        n = T + 100
        x = rng.normal(size=n)
        y = 0.08 * x + rng.normal(size=n)
        sxy, sxx = np.cumsum(x * y), np.cumsum(x * x)
        f = (np.r_[0.0, sxy[:-1] / sxx[:-1]] * x)[100:]
        cw_rej += clark_west(y[100:], np.zeros(T), f, 1).p_value < 0.05
        dm_rej += diebold_mariano(y[100:] - f, y[100:], 1).p_value < 0.05
    assert cw_rej >= dm_rej
    assert cw_rej > 20


def test_newey_west_lag0_is_variance() -> None:
    x = np.random.default_rng(0).normal(size=200)
    assert newey_west_variance(x, 0) == pytest.approx(np.var(x))


def _paired(signal: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.bdate_range("2020-01-01", "2024-06-28")
    x = rng.normal(size=len(t))
    y = signal * x + rng.normal(size=len(t))
    folds = [f"outer_{i:02d}" for i in (np.arange(len(t)) * 9 // len(t))]
    return pd.DataFrame({
        "origin_time": (t + pd.Timedelta(hours=18)).tz_localize("UTC"),
        "fold_id": folds, "actual": y, "model": signal * x, "bench": 0.0,
    })


def test_success_criteria_pass_and_fail() -> None:
    good = evaluate_success_criteria(_paired(0.5, 1), 1, clean_pit=True)
    assert good["criterion_1"]["passed"] and good["criterion_2"]["passed"]
    assert good["criterion_1"]["n_folds"] == 9
    assert good["criterion_2"]["n_ex_crisis"] < len(_paired(0.5, 1))
    dirty = evaluate_success_criteria(_paired(0.5, 1), 1, clean_pit=False)
    assert not dirty["criterion_1"]["passed"]
    null = _paired(0.0, 2)
    null["model"] = 0.05 * np.random.default_rng(3).normal(size=len(null))  # pure noise forecast
    bad = evaluate_success_criteria(null, 1, clean_pit=True)
    assert not bad["criterion_1"]["passed"] and not bad["criterion_2"]["passed"]
    assert bad["criterion_2"]["oos_r2_all"] < 0
