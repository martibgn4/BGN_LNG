"""Closed-form ridge with HAC standard errors and a block bootstrap.

numpy only, deliberately, so every coefficient traces to an expression here
rather than to a library's defaults - the same choice NWEBasisModel made.

THE PROBLEM THIS MODULE IS BUILT AROUND is overlapping returns. Sampling an
h-day return daily gives h-fold overlap, so consecutive observations share
h-1 days of the same price path. The residuals are strongly autocorrelated by
construction, and a naive t-statistic is inflated by roughly sqrt(h): at h=30
that is a factor of 5.5, which turns ordinary noise into a three-sigma result.

Two independent defences, and neither is optional:

  * Newey-West standard errors with truncation at least the horizon.
  * A stationary block bootstrap with mean block length scaled to the horizon,
    which makes no assumption about the autocorrelation's shape at all.

Where they disagree, believe the bootstrap. And `effective_n` is reported
alongside nominal N everywhere, because a scorecard that says N=1000 when the
independent count is 33 is not telling the truth about its own precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from wxreturns.config import get

log = logging.getLogger(__name__)


@dataclass
class Standardiser:
    """Fitted on TRAIN ONLY. Fitting on the full sample leaks the test mean."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "Standardiser":
        mean = features.mean(axis=0)
        scale = features.std(axis=0, ddof=1)
        # A constant column carries no information; leaving scale at zero would
        # divide by zero, and imputing a small number would blow it up into the
        # dominant regressor.
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def apply(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale


@dataclass
class RidgeFit:
    coefficients: np.ndarray
    intercept: float
    names: list[str]
    lambda_: float
    standardiser: Standardiser
    n_obs: int
    horizon: int
    hac_lags: int
    standard_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    t_stats: np.ndarray = field(default_factory=lambda: np.array([]))
    bootstrap_p: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def effective_n(self) -> int:
        """Nominal N divided by the horizon. Crude, conservative, checkable."""
        return int(self.n_obs // max(self.horizon, 1))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.standardiser.apply(features) @ self.coefficients + self.intercept


def hac_lag_for(horizon: int) -> int:
    """Newey-West truncation. At least the horizon, plus a configured margin."""
    rule = get("model", "inference", "hac_lag_rule")
    if rule != "horizon_plus_margin":
        raise ValueError(f"unknown hac_lag_rule {rule!r}")
    return int(horizon) + int(get("model", "inference", "hac_extra_lags"))


def _newey_west(design: np.ndarray, residuals: np.ndarray, lags: int
                ) -> np.ndarray:
    """HAC meat matrix with Bartlett weights."""
    n = design.shape[0]
    weighted = design * residuals[:, None]
    meat = weighted.T @ weighted
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        weight = 1.0 - lag / (lags + 1)
        cross = weighted[lag:].T @ weighted[:-lag]
        meat += weight * (cross + cross.T)
    return meat / n


def fit(features: np.ndarray, target: np.ndarray, names: list[str],
        horizon: int, lambda_: float | None = None) -> RidgeFit:
    """Ridge on standardised features, with HAC standard errors.

    The intercept is not penalised. On a return series a non-zero intercept is
    a claim about drift, which weather has no business making, so it is fitted
    and then watched rather than assumed away.
    """
    mask = np.isfinite(target) & np.isfinite(features).all(axis=1)
    x_raw, y = features[mask], target[mask]
    if len(y) == 0:
        raise ValueError("no complete observations after dropping missing rows")

    minimum = get("model", "sample", "min_observations")
    if len(y) < minimum:
        raise ValueError(
            f"{len(y)} usable observations, below the configured minimum of "
            f"{minimum}. At a {horizon}-day horizon that is only "
            f"{len(y) // max(horizon, 1)} independent observations, which is "
            "not enough to say anything."
        )

    lambda_ = float(lambda_ if lambda_ is not None
                    else get("model", "ridge", "default_lambda"))
    standardiser = Standardiser.fit(x_raw)
    x = standardiser.apply(x_raw)

    n, k = x.shape
    y_mean = float(y.mean())
    y_centred = y - y_mean

    gram = x.T @ x + lambda_ * np.eye(k)
    inverse = np.linalg.inv(gram)
    beta = inverse @ (x.T @ y_centred)

    residuals = y_centred - x @ beta
    lags = hac_lag_for(horizon)
    meat = _newey_west(x, residuals, lags)
    # Sandwich for the ridge estimator: (X'X + lI)^-1 X'ΩX (X'X + lI)^-1.
    covariance = inverse @ (meat * n) @ inverse
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(errors > 0.0, beta / errors, np.nan)

    return RidgeFit(
        coefficients=beta, intercept=y_mean, names=list(names),
        lambda_=lambda_, standardiser=standardiser, n_obs=int(n),
        horizon=int(horizon), hac_lags=lags,
        standard_errors=errors, t_stats=t_stats,
    )


def stationary_bootstrap_p(features: np.ndarray, target: np.ndarray,
                           horizon: int, lambda_: float,
                           draws: int | None = None) -> np.ndarray:
    """Two-sided p-values from a stationary block bootstrap (Politis-Romano).

    Blocks of geometrically distributed length are resampled with wraparound,
    which preserves the local dependence the overlap creates without assuming
    any particular autocorrelation shape. The null is imposed by recentring the
    bootstrap coefficients on the full-sample estimate.
    """
    cfg = get("model", "inference", "bootstrap")
    if not cfg["enabled"]:
        raise ValueError("bootstrap is disabled in config")
    draws = draws or cfg["draws"]
    multiplier = float(cfg["block_length_multiplier"])
    rng = np.random.default_rng(cfg["seed"])

    mask = np.isfinite(target) & np.isfinite(features).all(axis=1)
    x_raw, y = features[mask], target[mask]
    n = len(y)
    if n == 0:
        raise ValueError("no complete observations for the bootstrap")

    point = fit(x_raw, y, [f"x{i}" for i in range(x_raw.shape[1])],
                horizon, lambda_).coefficients

    mean_block = max(multiplier * float(horizon), 1.0)
    probability = 1.0 / mean_block

    samples = np.empty((draws, x_raw.shape[1]))
    for draw in range(draws):
        index = np.empty(n, dtype=int)
        position = rng.integers(0, n)
        for i in range(n):
            index[i] = position
            if rng.random() < probability:
                position = rng.integers(0, n)
            else:
                position = (position + 1) % n
        try:
            samples[draw] = fit(x_raw[index], y[index],
                                [f"x{i}" for i in range(x_raw.shape[1])],
                                horizon, lambda_).coefficients
        except (ValueError, np.linalg.LinAlgError):
            samples[draw] = np.nan

    centred = samples - point
    valid = np.isfinite(centred).all(axis=1)
    if valid.sum() == 0:
        raise ValueError("every bootstrap draw failed to fit")
    centred = centred[valid]
    return np.mean(np.abs(centred) >= np.abs(point), axis=0)


def oos_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Out-of-sample R² against a ZERO forecast, not against the sample mean.

    The benchmark is zero because a futures price is a martingale under an
    efficient market, so "no forecast" is the honest competitor. Using the test
    mean instead would credit the model with knowing the realised drift, which
    it could not have known.
    """
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() == 0:
        return float("nan")
    errors = actual[mask] - predicted[mask]
    benchmark = actual[mask]
    denominator = float(np.sum(np.square(benchmark)))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(errors))) / denominator


def directional_hit_rate(actual: np.ndarray, predicted: np.ndarray,
                         threshold: float | None = None) -> dict:
    """Hit rate on days the model actually takes a position.

    Days below the signal threshold are excluded rather than counted as
    correct, and the count of traded days is returned so the rate can be read
    against its own sample.
    """
    threshold = (threshold if threshold is not None
                 else get("model", "signal", "min_abs_forecast_return"))
    mask = (np.isfinite(actual) & np.isfinite(predicted)
            & (np.abs(predicted) >= threshold))
    traded = int(mask.sum())
    if traded == 0:
        return {"hit_rate": float("nan"), "n_traded": 0}
    hits = int(np.sum(np.sign(actual[mask]) == np.sign(predicted[mask])))
    return {"hit_rate": hits / traded, "n_traded": traded}
