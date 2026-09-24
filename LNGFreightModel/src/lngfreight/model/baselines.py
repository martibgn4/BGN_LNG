"""The two benchmarks the model has to beat.

Both are fitted inside the walk-forward loop on the training window only, the
same as any estimator. A benchmark that is allowed to peek is a benchmark
that is too easy or too hard to beat, and either way the comparison stops
meaning anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomWalk:
    """Predict no change.

    For a spot assessment this is the honest null. At h=1 it is genuinely
    hard to beat, because most of the day-to-day variation in a freight
    assessment is unforecastable from anything published daily.
    """

    name = "random_walk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalk":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class FfaForward:
    """The FFA curve's implied change, de-biased on the training window.

    The raw signal log(FFA / spot) is NOT an unbiased forecast of an h-day
    spot change. The near FFA contract settles on a month average while the
    target is a daily assessment, so the two carry a structural basis - which
    was -0.60 in log terms in 2022 and around +0.08 by 2026 as the FFA market
    matured. Handing the raw signal to the comparison table would be setting
    up a benchmark that fails for a units reason rather than an economic one,
    and beating it would prove nothing.

    So the mean basis is estimated on the training window and removed, and a
    slope is optionally fitted the same way. Both are causal: the test block
    never contributes to its own calibration.

    ``scale`` restores the horizon. The curve expresses a view about a month;
    over h business days only part of that convergence can happen, and the
    fitted slope is what measures how much.
    """

    name = "ffa_forward"

    def __init__(self, fit_slope: bool = True) -> None:
        self.fit_slope = fit_slope
        self.intercept_ = 0.0
        self.slope_ = 1.0

    def fit(self, signal: pd.Series, y: pd.Series) -> "FfaForward":
        aligned = pd.concat([signal.rename("s"), y.rename("y")], axis=1).dropna()
        if aligned.empty:
            raise ValueError("no overlapping rows to calibrate the FFA benchmark")
        if self.fit_slope and aligned["s"].std() > 0:
            # Least squares of realised change on the forward signal. The
            # slope is the fraction of the curve's view that actually shows
            # up over h days; the intercept absorbs the structural basis.
            slope, intercept = np.polyfit(aligned["s"], aligned["y"], deg=1)
            self.slope_, self.intercept_ = float(slope), float(intercept)
        else:
            self.slope_ = 0.0
            self.intercept_ = float(aligned["y"].mean())
        return self

    def predict(self, signal: pd.Series) -> np.ndarray:
        return (self.intercept_ + self.slope_ * signal.to_numpy(dtype=float))
