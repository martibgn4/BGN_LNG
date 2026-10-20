"""Level 6 combinations: constrained linear stacking and performance-weighted averaging.

**Not for use until single models are understood** (research spec §9: no Level 3+ for a target
that fails Levels 1–2). Interface and guards only.

Inputs ``X`` are **out-of-fold** predictions: one column per base model, indexed by origin time.
Leakage guards (all raise :class:`LeakageGuardError`):

* ``fit`` requires ``train_end`` and refuses any OOF row dated after it;
* if ``label_known_time`` is given, rows whose label is not known by ``train_end`` are refused;
* if ``prediction_train_end`` (same shape as ``X``: the ``train_end`` of the fold that produced
  each OOF prediction) is given, every prediction must come from a fold that ended **before** its
  origin — i.e. truly out-of-fold;
* ``predict`` refuses origins at or before ``train_end`` (in-sample) unless
  ``allow_in_sample=True``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ttf_forecaster.models.interface import (
    ForecastModel,
    InsufficientDataError,
    LeakageGuardError,
    register_model,
)


def _as_ts(t: Any, like: pd.Index) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    tz = getattr(like, "tz", None)
    if tz is not None and ts.tzinfo is None:
        raise LeakageGuardError("train_end must be tz-aware when the index is tz-aware")
    if tz is not None:
        ts = ts.tz_convert(tz)
    return ts


class _Combiner(ForecastModel):
    level = 6
    default_params: ClassVar[dict[str, Any]] = {"allow_in_sample": False}

    def fit(self, X: pd.DataFrame, y: pd.Series, *, train_end: Any = None,  # type: ignore[override]
            label_known_time: pd.Series | None = None,
            prediction_train_end: pd.DataFrame | None = None) -> _Combiner:
        """Fit combination weights on OOF predictions ``X`` (columns = base models)."""
        if train_end is None:
            raise LeakageGuardError(f"{self.name}.fit requires train_end")
        te = _as_ts(train_end, X.index)
        if (X.index > te).any():
            raise LeakageGuardError(
                f"{self.name}: {int((X.index > te).sum())} OOF rows dated after train_end {te}")
        if label_known_time is not None:
            lkt = label_known_time.reindex(X.index)
            if lkt.isna().any() or (lkt > te).any():
                raise LeakageGuardError(f"{self.name}: OOF rows with label_known_time after "
                                        "train_end (or missing) cannot be used")
        if prediction_train_end is not None:
            pte = prediction_train_end.reindex(index=X.index, columns=X.columns)
            origins = pd.Series(X.index, index=X.index)
            bad = sum(int((pte[c].isna() | ~(pte[c] < origins)).sum()) for c in pte.columns)
            if bad:
                raise LeakageGuardError(
                    f"{self.name}: {bad} predictions were not produced by a fold "
                    "ending before their origin (not out-of-fold)")
        self.combine_train_end_ = te
        complete = X.notna().all(axis=1)
        if complete.sum() < 2 * X.shape[1] + 10:
            raise InsufficientDataError(f"{self.name}: {int(complete.sum())} complete OOF rows")
        lkt_c = None if label_known_time is None else label_known_time.reindex(X.index)[complete]
        super().fit(X[complete], y[complete], label_known_time=lkt_c)
        return self

    def _check_predict_inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super()._check_predict_inputs(X)
        if not self.params["allow_in_sample"]:
            te = self.combine_train_end_
            if (X.index <= te).any():
                raise LeakageGuardError(
                    f"{self.name}: prediction origins at or before train_end {te} are in-sample")
        return X

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        # NaN in any component -> NaN combined forecast (no silent re-weighting).
        return X.to_numpy(dtype=float) @ self.weights_.to_numpy()


@register_model
class ConstrainedStacking(_Combiner):
    """Weights ``w >= 0``, ``Σw = 1`` minimising in-sample OOF MSE (SLSQP, deterministic start at
    equal weights). No intercept: a combination of forecasts, not a new regression."""

    name = "constrained_stacking"

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from scipy.optimize import minimize

        P, t = X.to_numpy(dtype=float), y.to_numpy()
        k = P.shape[1]

        def loss(w: np.ndarray) -> float:
            r = t - P @ w
            return float(r @ r / len(t))

        def grad(w: np.ndarray) -> np.ndarray:
            return -2.0 * P.T @ (t - P @ w) / len(t)

        res = minimize(loss, np.full(k, 1.0 / k), jac=grad, method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                       options={"ftol": 1e-12, "maxiter": 500})
        if not res.success:
            raise RuntimeError(f"constrained_stacking optimisation failed: {res.message}")
        w = np.clip(res.x, 0.0, None)
        self.weights_: pd.Series = pd.Series(w / w.sum(), index=list(X.columns), name="weight")


@register_model
class PerformanceWeighted(_Combiner):
    """Weights ∝ ``MSE^-power`` of each model over the last ``window`` OOF rows (``None`` = all)."""

    name = "performance_weighted"
    default_params: ClassVar[dict[str, Any]] = {**_Combiner.default_params, "window": None,
                                                "power": 1.0}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        w = self.params["window"]
        Xw, yw = (X, y) if w is None else (X.iloc[-int(w):], y.iloc[-int(w):])
        mse = ((Xw.sub(yw, axis=0)) ** 2).mean()
        inv = (mse.clip(lower=1e-300)) ** (-float(self.params["power"]))
        self.mse_: pd.Series = mse
        self.weights_ = (inv / inv.sum()).rename("weight")
