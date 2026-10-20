"""Level 4 probabilistic models: quantile regression, time-series split conformal intervals
(optionally adaptive), and calibrated threshold-event probabilities.

Every held-out slice used here (conformal calibration, probability calibration, early stopping)
is taken from the **end of the training rows**, separated from the fitting head by ``horizon``
rows. Nothing is calibrated on test data. Quantile outputs are rearranged (sorted) so they never
cross.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ttf_forecaster.models.boosting import _require_lightgbm, eval_kwargs, lgb_params
from ttf_forecaster.models.interface import (
    ForecastModel,
    InsufficientDataError,
    TabularPreprocessor,
    grid_search_time_cv,
    make_model,
    register_model,
)

DEFAULT_LEVELS = (0.1, 0.5, 0.9)


def _fitted_level_index(fitted: list[float], levels: list[float], who: str) -> list[int]:
    idx = []
    for q in levels:
        hits = [i for i, f in enumerate(fitted) if abs(f - q) < 1e-9]
        if not hits:
            raise ValueError(f"{who}: level {q} was not fitted; fitted levels {fitted}")
        idx.append(hits[0])
    return idx


def _levels_with_median(levels: Any) -> list[float]:
    lv = sorted({float(v) for v in levels} | {0.5})
    if any(not 0 < v < 1 for v in lv):
        raise ValueError(f"quantile levels must be in (0, 1): {lv}")
    return lv


# ---------------------------------------------------------------------------------------------
# Quantile regression
# ---------------------------------------------------------------------------------------------


@register_model
class LinearQuantileRegression(ForecastModel):
    """Linear quantile regression per level (sklearn ``QuantileRegressor``, HiGHS solver).

    X is imputed/standardised (optionally family-PCA'd) on train; y is standardised by train
    moments so the L1 ``alpha`` (fixed, from config) is scale-free. ``predict`` = the median.
    """

    name = "linear_quantile"
    level = 4
    supports_quantiles = True
    default_params: ClassVar[dict[str, Any]] = {
        "levels": list(DEFAULT_LEVELS),
        "alpha": 0.01,
        "family_pca": None,
    }

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.linear_model import QuantileRegressor

        self.levels_ = _levels_with_median(self.params["levels"])
        self.prep_ = TabularPreprocessor(impute=True, standardize=True,
                                         family_pca=self.params["family_pca"])
        Z = self.prep_.fit_transform(X)
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std(ddof=0)) or 1.0
        t = (y.to_numpy() - self.y_mean_) / self.y_scale_
        self.models_ = [
            QuantileRegressor(quantile=q, alpha=float(self.params["alpha"]), solver="highs").fit(Z, t)
            for q in self.levels_
        ]

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep_.transform(X)
        cols = [m.predict(Z) for m in self.models_]
        return self.y_mean_ + self.y_scale_ * np.column_stack(cols)

    def _predict_quantiles(self, X: pd.DataFrame, levels: list[float]) -> np.ndarray:
        full = np.sort(self._raw(X), axis=1)
        return full[:, _fitted_level_index(self.levels_, levels, self.name)]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_quantiles(X, [0.5])[:, 0]


@register_model
class LightGBMQuantile(ForecastModel):
    """LightGBM with ``objective="quantile"`` per level; early stopping on the training tail,
    then refit on all training rows with the selected rounds. Outputs are rearranged."""

    name = "lightgbm_quantile"
    level = 4
    supports_quantiles = True
    default_params: ClassVar[dict[str, Any]] = {
        "levels": list(DEFAULT_LEVELS),
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 100,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "early_stopping_rounds": 50,
        "es_fraction": 0.2,
    }

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        lgb = _require_lightgbm()
        self.levels_ = _levels_with_median(self.params["levels"])
        Z, t = X.to_numpy(dtype=float), y.to_numpy()
        try:
            head, tail = self._tail_split(X, fraction=float(self.params["es_fraction"]))
        except InsufficientDataError:
            head = tail = None
        self.models_, self.best_iterations_ = [], []
        for q in self.levels_:
            rounds = int(self.params["n_estimators"])
            if head is not None and int(self.params["early_stopping_rounds"]) > 0:
                kw = lgb_params(self.params, self.seed, len(head))
                est = lgb.LGBMRegressor(objective="quantile", alpha=q, **kw)
                est.fit(Z[head], t[head], **eval_kwargs(Z[tail], t[tail]),
                        eval_metric="quantile",
                        callbacks=[lgb.early_stopping(int(self.params["early_stopping_rounds"]),
                                                      verbose=False)])
                rounds = int(est.best_iteration_ or rounds)
            kw = lgb_params(self.params, self.seed, len(t))
            kw["n_estimators"] = max(1, rounds)
            self.models_.append(lgb.LGBMRegressor(objective="quantile", alpha=q, **kw).fit(Z, t))
            self.best_iterations_.append(rounds)

    def _predict_quantiles(self, X: pd.DataFrame, levels: list[float]) -> np.ndarray:
        Z = X.to_numpy(dtype=float)
        full = np.sort(np.column_stack([m.predict(Z) for m in self.models_]), axis=1)
        return full[:, _fitted_level_index(self.levels_, levels, self.name)]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_quantiles(X, [0.5])[:, 0]


# ---------------------------------------------------------------------------------------------
# Split conformal for time series (+ adaptive conformal inference)
# ---------------------------------------------------------------------------------------------


def conformal_quantile(residuals: np.ndarray, level: float) -> float:
    """Empirical residual quantile with the split-conformal finite-sample correction
    (rank ``ceil((n+1)·q)`` above the median, ``floor((n+1)·q)`` below)."""
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        raise InsufficientDataError("no calibration residuals")
    if level >= 0.5:
        k = min(n, int(np.ceil((n + 1) * level)))
    else:
        k = max(1, int(np.floor((n + 1) * level)))
    return float(r[k - 1])


@register_model
class ConformalInterval(ForecastModel):
    """Split-conformal quantiles around any registered point model, adapted to time series.

    ``fit``: the base model is fitted on the training head; signed residuals are collected on a
    calibration window taken from the **end** of the training rows (``calib_fraction``, after a
    ``horizon``-row gap). ``predict`` = base point forecast; ``predict_quantiles`` = point +
    conformal residual quantile per level (signed → asymmetric intervals allowed).

    ``adaptive=True`` enables adaptive conformal inference (Gibbs & Candès 2021) through
    :meth:`predict_interval` / :meth:`update`. The runner must call ``update`` only with labels whose
    ``label_known_time`` precedes the next origin it predicts; ``update`` never refits anything.
    """

    name = "conformal"
    level = 4
    supports_quantiles = True
    default_params: ClassVar[dict[str, Any]] = {
        "base": "ridge",
        "base_params": {},
        "calib_fraction": 0.25,
        "adaptive": False,
        "aci_gamma": 0.005,
        "aci_alpha": 0.2,
    }

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        head, tail = self._tail_split(X, fraction=float(self.params["calib_fraction"]))
        bp = {"horizon": self.horizon, "seed": self.seed, **dict(self.params["base_params"])}
        self.base_ = make_model(self.params["base"], **bp)
        self.base_.fit(X.iloc[head], y.iloc[head],
                       label_known_time=self._lkt_for(X.iloc[head]))
        self.residuals_ = (y.iloc[tail] - self.base_.predict(X.iloc[tail])).to_numpy()
        self.alpha_t_: float = float(self.params["aci_alpha"])
        self.aci_history_: list[float] = []

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.predict(X).to_numpy()

    def _predict_quantiles(self, X: pd.DataFrame, levels: list[float]) -> np.ndarray:
        point = self._predict(X)
        offs = np.array([conformal_quantile(self.residuals_, q) for q in levels])
        return point[:, None] + offs[None, :]

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """Central interval at the current (adaptive) miscoverage ``alpha_t_``: columns
        ``lower``, ``upper``, ``alpha_t``."""
        X = self._check_predict_inputs(X)
        a = float(np.clip(self.alpha_t_, 1e-3, 0.999))
        point = self._predict(X)
        lo = point + conformal_quantile(self.residuals_, a / 2)
        hi = point + conformal_quantile(self.residuals_, 1 - a / 2)
        return pd.DataFrame({"lower": lo, "upper": hi, "alpha_t": a}, index=X.index)

    def update(self, y_true: float, lower: float, upper: float) -> float:
        """ACI step ``alpha_t += gamma·(alpha − err_t)``; returns the new ``alpha_t``."""
        if not self.params["adaptive"]:
            raise RuntimeError("update() requires adaptive=True")
        err = 0.0 if lower <= y_true <= upper else 1.0
        self.alpha_t_ += float(self.params["aci_gamma"]) * (float(self.params["aci_alpha"]) - err)
        self.aci_history_.append(self.alpha_t_)
        return self.alpha_t_


# ---------------------------------------------------------------------------------------------
# Threshold-event probabilities with time-ordered calibration
# ---------------------------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class _CalibratedEvent(ForecastModel):
    """Base for event classifiers: label construction, head/tail split, calibrator.

    ``threshold=None`` → ``y`` must already be binary. Otherwise the label is ``y > threshold``
    (``event="above"``), ``y < threshold`` (``"below"``) or ``|y| > threshold`` (``"abs_above"``).
    The base classifier is fitted on the training head; the calibrator (Platt ``sigmoid`` on the
    logit, ``isotonic`` or ``none``) on the last ``calib_fraction`` of training rows after a
    ``horizon`` gap. The base is **not** refit on the tail, so the calibrator matches it.
    """

    level = 4
    task = "classification"
    supports_proba = True
    default_params: ClassVar[dict[str, Any]] = {
        "threshold": None,
        "event": "above",
        "calibration": "sigmoid",
        "calib_fraction": 0.25,
    }

    def _validate_params(self) -> None:
        if self.params["event"] not in ("above", "below", "abs_above"):
            raise ValueError("event must be above | below | abs_above")
        if self.params["calibration"] not in ("sigmoid", "isotonic", "none"):
            raise ValueError("calibration must be sigmoid | isotonic | none")

    def to_label(self, y: pd.Series) -> pd.Series:
        thr = self.params["threshold"]
        if thr is None:
            return y
        ev = self.params["event"]
        lab = y > thr if ev == "above" else (y < thr if ev == "below" else y.abs() > thr)
        return lab.astype(float)

    def _check_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(y, pd.Series) and not y.isna().any():
            y = self.to_label(y)
        return super()._check_fit_inputs(X, y)

    def _fit_base(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        method = self.params["calibration"]
        if method == "none":
            head, tail = np.arange(len(y)), None
        else:
            head, tail = self._tail_split(X, fraction=float(self.params["calib_fraction"]))
        if y.iloc[head].nunique() < 2:
            raise InsufficientDataError(f"{self.name}: training head contains a single class")
        self._fit_base(X.iloc[head], y.iloc[head])
        self.calibrator_: Any = None
        self.calibration_used_ = "none"
        if tail is None:
            return
        yt = y.iloc[tail].to_numpy()
        if len(np.unique(yt)) < 2:
            warnings.warn(f"{self.name}: calibration tail has one class; probabilities left "
                          "uncalibrated (calibration_used_='none')", stacklevel=2)
            return
        raw = self._base_proba(X.iloc[tail])
        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw)[:, None], yt)
        else:
            from sklearn.isotonic import IsotonicRegression

            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0,
                                                  out_of_bounds="clip").fit(raw, yt)
        self.calibration_used_ = method

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._base_proba(X)
        if self.calibrator_ is None:
            return raw
        if self.calibration_used_ == "sigmoid":
            return self.calibrator_.predict_proba(_logit(raw)[:, None])[:, 1]
        return self.calibrator_.predict(raw)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_proba(X)


@register_model
class ThresholdLogistic(_CalibratedEvent):
    """L2 logistic regression; ``C`` chosen by time-ordered inner CV (log loss, gap = horizon) on
    the training head; inputs imputed/standardised (optionally family-PCA'd) on train."""

    name = "threshold_logistic"
    default_params: ClassVar[dict[str, Any]] = {
        **_CalibratedEvent.default_params,
        "Cs": [0.001, 0.01, 0.1, 1.0, 10.0],
        "cv_splits": 3,
        "cv_val_fraction": 0.25,
        "family_pca": None,
    }

    def _fit_core(self, X: pd.DataFrame, y: pd.Series, C: float) -> tuple[Any, Any]:
        from sklearn.linear_model import LogisticRegression

        prep = TabularPreprocessor(impute=True, standardize=True,
                                   family_pca=self.params["family_pca"])
        Z = prep.fit_transform(X)
        est = LogisticRegression(C=float(C), max_iter=2000).fit(Z, y.to_numpy().astype(int))
        return prep, est

    def _fit_base(self, X: pd.DataFrame, y: pd.Series) -> None:
        def fit_score(Xt: pd.DataFrame, yt: pd.Series, Xv: pd.DataFrame, yv: pd.Series,
                      C: float) -> float:
            if yt.nunique() < 2:
                return float(np.inf)
            prep, est = self._fit_core(Xt, yt, C)
            p = np.clip(est.predict_proba(prep.transform(Xv))[:, 1], 1e-6, 1 - 1e-6)
            v = yv.to_numpy()
            return float(-np.mean(v * np.log(p) + (1 - v) * np.log(1 - p)))

        Cs = sorted(float(c) for c in self.params["Cs"])  # most regularised first
        self.C_, self.cv_table_ = grid_search_time_cv(
            X, y, Cs, fit_score, gap=self.horizon, n_splits=int(self.params["cv_splits"]),
            val_fraction=float(self.params["cv_val_fraction"]),
            label_known_time=self._lkt_for(X))
        self.prep_, self.est_ = self._fit_core(X, y, self.C_)
        self.coef_: pd.Series = pd.Series(self.est_.coef_[0], index=self.prep_.output_names_)

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.est_.predict_proba(self.prep_.transform(X))[:, 1]


@register_model
class LightGBMEventClassifier(_CalibratedEvent):
    """``boosting.LightGBMClassifier`` (its own early stopping inside the head) + calibration."""

    name = "lightgbm_event"
    default_params: ClassVar[dict[str, Any]] = {
        **_CalibratedEvent.default_params,
        "lgb_params": {},
    }

    def _fit_base(self, X: pd.DataFrame, y: pd.Series) -> None:
        bp = {"horizon": self.horizon, "seed": self.seed, **dict(self.params["lgb_params"])}
        self.base_ = make_model("lightgbm_classifier", **bp)
        self.base_.fit(X, y, label_known_time=self._lkt_for(X))

    def _base_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.predict_proba(X).to_numpy()
