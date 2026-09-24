"""Estimator tests.

These are recovery tests: simulate a process whose parameters are known, fit
it, and check the estimator returns what was put in. A model that cannot
recover a parameter from data it generated itself will not recover one from the
market either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nwebasis.model import ecm, envelope, seasonal_ou

SEED = 20260904


def simulate_ou(kappa: float, sigma: float, theta: float, n: int,
                periods: int = 252) -> pd.Series:
    """Exact-discretisation OU path, so the test does not inherit Euler bias."""
    rng = np.random.default_rng(SEED)
    dt = 1 / periods
    phi = np.exp(-kappa * dt)
    shock_sd = sigma * np.sqrt((1 - phi**2) / (2 * kappa))
    values = np.empty(n)
    values[0] = theta
    for i in range(1, n):
        values[i] = theta + phi * (values[i - 1] - theta) + shock_sd * rng.standard_normal()
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(values, index=index, name="basis")


def test_ou_recovers_its_own_parameters() -> None:
    kappa, sigma, theta = 4.0, 0.5, -0.4
    series = simulate_ou(kappa, sigma, theta, n=3000)
    fitted = seasonal_ou.fit_ou(series, "M+1", "2020-01-01", "2031-01-01")

    assert fitted.kappa_annual == pytest.approx(kappa, rel=0.15)
    assert fitted.sigma_annual == pytest.approx(sigma, rel=0.10)
    assert fitted.theta == pytest.approx(theta, abs=0.05)


def test_ou_refuses_a_short_sample() -> None:
    series = simulate_ou(4.0, 0.5, -0.4, n=50)
    with pytest.raises(seasonal_ou.InsufficientHistory):
        seasonal_ou.fit_ou(series, "M+1", "2020-01-01", "2020-03-01",
                           min_observations=250)


def test_ou_refuses_a_random_walk() -> None:
    """A unit root has no half-life. The estimator must say so, not report one.

    A finite sample of a random walk fits phi slightly BELOW one, so the
    (0, 1) check alone passes it and the estimator would happily report a
    several-hundred-day half-life. The half-life ceiling is what actually
    catches this case, which is why both guards exist.
    """
    rng = np.random.default_rng(SEED)
    walk = pd.Series(np.cumsum(rng.standard_normal(1000)))
    with pytest.raises(seasonal_ou.InsufficientHistory, match="random walk"):
        seasonal_ou.fit_ou(walk, "M+1", "2020-01-01", "2024-01-01")


def test_deseasonalise_removes_the_month_effect() -> None:
    rng = np.random.default_rng(SEED)
    months = np.tile(np.arange(1, 13), 60)
    seasonal = np.where(months <= 3, -0.5, 0.0)
    frame = pd.DataFrame({
        "delivery_month": months,
        "value": seasonal + rng.standard_normal(len(months)) / 100,
    })
    adjusted, offsets = seasonal_ou.deseasonalise(frame)

    assert offsets.sum() == pytest.approx(0.0, abs=1e-9)
    spread = adjusted.groupby(frame["delivery_month"]).mean()
    assert spread.max() - spread.min() < 0.01


def test_ecm_rejects_a_positive_alpha() -> None:
    """An explosive error-correction term is a broken spec, not a finding."""
    rng = np.random.default_rng(SEED)
    n = 500
    anchor = rng.standard_normal(n).cumsum() / 100
    # Build a series that runs AWAY from its anchor.
    value = anchor.copy()
    for i in range(1, n):
        value[i] = value[i - 1] + (value[i - 1] - anchor[i - 1]) / 10
    frame = pd.DataFrame({
        "value": value,
        "anchor_feature": anchor,
        "short": rng.standard_normal(n),
    })
    with pytest.raises(ecm.SpecificationError, match="error-correction"):
        ecm.fit(frame, ["anchor_feature"], ["short"])


def test_standardiser_uses_train_statistics_only() -> None:
    train = np.array([[0.0], [2.0], [4.0]])
    scaler = seasonal_ou_standardiser(train)
    shifted = scaler.apply(np.array([[100.0]]))
    # If the scaler had refitted on the new data this would come back at 0.
    assert shifted[0, 0] > 10


def seasonal_ou_standardiser(matrix: np.ndarray) -> ecm.Standardiser:
    return ecm.Standardiser.fit(matrix)


def test_cost_stack_refuses_to_guess() -> None:
    """regas.yaml ships unset. The floor must raise, not default."""
    with pytest.raises(envelope.RegasCostUnset) as excinfo:
        envelope.cost_stack()
    message = str(excinfo.value)
    assert "terminals.gate.tariff" in message
    assert "empirical" in message


def test_regime_classifier_flags_a_thin_regime() -> None:
    slots = pd.Series(np.arange(100, dtype=float))
    labels = envelope.classify(slots)
    assert set(labels.unique()) <= {"scarce", "ample"}
    assert (labels == "scarce").sum() > 0
    assert "warning" in labels.attrs
