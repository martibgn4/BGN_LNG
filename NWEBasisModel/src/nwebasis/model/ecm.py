"""Error-correction model for the NWE basis.

Two stages, in the Engle-Granger shape:

  Stage 1, the anchor.  basis_t = gamma' Z_t + u_t

    Z is the long-run block: the delivery-month seasonal, regas slot
    availability, storage headroom, the TTF time spread. These are the things
    that can move where the basis SITS. u_t is the disequilibrium - how far the
    basis is from the level its fundamentals justify.

  Stage 2, the correction.  d basis_t = alpha * u_{t-1} + beta' d X_t + e_t

    alpha is the speed the gap closes and must come out negative; a positive
    alpha means the specification is wrong, not that the basis is explosive,
    and ``fit`` says so rather than reporting it. X is the short-run block:
    the east-west arb, freight changes, the SWE spread.

WHY NOT A SINGLE REGRESSION ON LEVELS. The basis and its drivers are
persistent. Regressing one on the others in levels produces a high R-squared
and standard errors that are wrong by an order of magnitude - the classic
spurious regression. Splitting the long run from the short run is what makes
the coefficients mean anything.

WHY RIDGE ON THE SHORT-RUN BLOCK ONLY. Atlantic and Pacific freight correlate
0.93; unpenalised OLS on both gives coefficients that flip sign between folds
while the fitted values barely move. The anchor terms are left unpenalised
because they carry the economics and shrinking them would bias the equilibrium.

Standard errors are Newey-West. The features are overlapping five-day changes,
so the residuals are autocorrelated by construction and plain OLS errors would
overstate significance roughly threefold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nwebasis.config import get


class SpecificationError(RuntimeError):
    """The fitted model contradicts the economics it is supposed to encode."""


@dataclass
class Standardiser:
    """Column means and scales, kept so test data is transformed by TRAIN stats.

    Standardising on the full sample before splitting is a small, common, and
    entirely real leak: the test period's mean and variance end up inside the
    training transform. Held explicitly here so that cannot happen by accident.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardiser":
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=1)
        scale[scale == 0] = 1
        return cls(mean=mean, scale=scale)

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.scale


@dataclass
class ECMFit:
    anchor_names: list[str]
    anchor_coefficients: np.ndarray
    short_run_names: list[str]
    short_run_coefficients: np.ndarray
    alpha: float
    alpha_stderr: float
    short_run_stderr: np.ndarray
    residual_sd: float
    n_observations: int
    r_squared: float
    standardiser: Standardiser

    @property
    def half_life_days(self) -> float:
        """Implied by alpha. Comparable with the OU half-life; they should agree."""
        return float(np.log(2) / -np.log(1 + self.alpha))

    def summary(self) -> pd.DataFrame:
        rows = [{"term": "alpha (error correction)", "coefficient": self.alpha,
                 "std_error": self.alpha_stderr,
                 "t_stat": self.alpha / self.alpha_stderr if self.alpha_stderr else np.nan}]
        for name, coefficient, stderr in zip(self.short_run_names,
                                             self.short_run_coefficients,
                                             self.short_run_stderr):
            rows.append({"term": name, "coefficient": float(coefficient),
                         "std_error": float(stderr),
                         "t_stat": float(coefficient / stderr) if stderr else np.nan})
        return pd.DataFrame(rows)


def _newey_west(design: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance. Bartlett kernel, the standard choice for this."""
    n, k = design.shape
    xtx_inverse = np.linalg.pinv(design.T @ design)
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    return covariance * n / max(n - k, 1)


def fit_anchor(frame: pd.DataFrame, anchor_names: list[str],
               target: str = "value") -> tuple[np.ndarray, pd.Series]:
    """Stage 1. Returns the cointegrating coefficients and the residual gap."""
    usable = frame.dropna(subset=anchor_names + [target])
    if usable.empty:
        raise SpecificationError(
            f"no rows have all of {anchor_names} plus {target}. Check the "
            "manifest - a feature is probably present as a column of NaN."
        )
    design = np.column_stack([np.ones(len(usable)),
                              usable[anchor_names].to_numpy(dtype=float)])
    observed = usable[target].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    gap = pd.Series(observed - design @ coefficients, index=usable.index, name="gap")
    return coefficients, gap


def fit(frame: pd.DataFrame, anchor_names: list[str], short_run_names: list[str],
        target: str = "value") -> ECMFit:
    """Fit both stages on one tenor's time series.

    ``frame`` must be a single tenor sorted by release date. Pooling tenors
    here would mix processes with different reversion speeds into one alpha.
    """
    coefficients, gap = fit_anchor(frame, anchor_names, target)
    working = frame.loc[gap.index].copy()
    working["gap"] = gap
    working["d_target"] = working[target].diff()
    working["gap_lagged"] = working["gap"].shift(1)

    columns = ["gap_lagged"] + short_run_names
    usable = working.dropna(subset=["d_target"] + columns)
    if usable.empty:
        raise SpecificationError(
            f"no rows survive the differencing with short-run terms {short_run_names}"
        )

    raw_short = usable[short_run_names].to_numpy(dtype=float)
    standardiser = Standardiser.fit(raw_short)
    scaled_short = (standardiser.apply(raw_short)
                    if get("model", "ecm", "standardise_features") else raw_short)

    error_correction = usable["gap_lagged"].to_numpy(dtype=float)[:, None]
    design = np.column_stack([np.ones(len(usable)), error_correction, scaled_short])
    observed = usable["d_target"].to_numpy(dtype=float)

    # Ridge on the short-run block only: the penalty matrix zeroes the
    # intercept and the error-correction term.
    lam = float(get("model", "ecm", "ridge_lambda"))
    penalty = np.eye(design.shape[1]) * lam
    penalty[0, 0] = 0
    penalty[1, 1] = 0
    estimates = np.linalg.solve(design.T @ design + penalty * len(usable),
                                design.T @ observed)

    residuals = observed - design @ estimates
    covariance = _newey_west(design, residuals, int(get("model", "ecm", "hac_lags")))
    stderrs = np.sqrt(np.clip(np.diag(covariance), 0, None))

    alpha = float(estimates[1])
    if alpha >= 0:
        raise SpecificationError(
            f"alpha came out {alpha:+.4f}. A non-negative error-correction term "
            "means the gap widens rather than closes, which is a broken "
            "specification, not a tradeable finding. Check the anchor block."
        )

    total_variance = float(((observed - observed.mean()) ** 2).sum())
    r_squared = 1 - float((residuals**2).sum()) / total_variance if total_variance else np.nan

    return ECMFit(
        anchor_names=anchor_names,
        anchor_coefficients=coefficients,
        short_run_names=short_run_names,
        short_run_coefficients=estimates[2:],
        alpha=alpha,
        alpha_stderr=float(stderrs[1]),
        short_run_stderr=stderrs[2:],
        residual_sd=float(residuals.std(ddof=design.shape[1])),
        n_observations=len(usable),
        r_squared=r_squared,
        standardiser=standardiser,
    )


def predict(model: ECMFit, frame: pd.DataFrame, horizon: int,
            target: str = "value") -> pd.Series:
    """Forecast the basis ``horizon`` business days ahead.

    Iterating the one-step equation compounds the short-run terms, which we do
    not have future values for. So the short-run block is held at its last
    observed value and only the error-correction term is rolled forward. That
    is the honest version: beyond a few days the forecast IS the anchor plus a
    decaying gap, and dressing it up as something richer would overstate what
    the model knows.
    """
    usable = frame.dropna(subset=model.anchor_names + [target])
    design = np.column_stack([np.ones(len(usable)),
                              usable[model.anchor_names].to_numpy(dtype=float)])
    anchor = design @ model.anchor_coefficients
    gap = usable[target].to_numpy(dtype=float) - anchor
    decayed = gap * (1 + model.alpha) ** horizon
    return pd.Series(anchor + decayed, index=usable.index, name=f"forecast_h{horizon}")
