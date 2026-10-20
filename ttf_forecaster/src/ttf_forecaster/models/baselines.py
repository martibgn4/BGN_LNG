"""Level 0 benchmarks (research spec §4; target_definitions "Benchmarks per target").

A model is compared with the **best** appropriate benchmark, not the weakest. All parameters a
benchmark estimates (rolling mean, AR order and coefficients, monthly means, storage slope, HAR
weights, base rate) come from the training rows passed to ``fit``.

Benchmarks that need information at the origin (current level, trailing vol, the target's own
lags) read it from named columns of ``X``. Those columns are produced PIT-safely by the feature
layer, or by :func:`make_target_lags` below for the target's own history.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ttf_forecaster.models.interface import (
    ForecastModel,
    InsufficientDataError,
    origin_months,
    register_model,
)
from ttf_forecaster.models.linear import _linear_predict, hac_ols
from ttf_forecaster.targets.common import session_horizon


def lag_horizon(targets: pd.DataFrame, declared: int | None = None) -> int:
    """Session horizon to pass to :func:`make_target_lags` for a target frame (D-032 F1).

    Returns ``label_overlap_rows + 1``. Refuses a ``declared`` horizon that disagrees with it, so a
    month count (T4/T5 ``horizon`` column) cannot silently shift the AR lags.
    """
    return session_horizon(targets, declared)


def make_target_lags(y: pd.Series, *, horizon: int, p: int, prefix: str = "ar") -> pd.DataFrame:
    """Lag columns of an h-day target that are **known at each origin**.

    The label at origin ``t - horizon`` is known at the close of ``t`` (target_definitions §3), so
    lag ``k`` (1-based) is ``y.shift(horizon + k - 1)``. Columns ``<prefix>__lag<k>``. Build on the
    full, contiguous origin grid (one row per trading day) **before** splitting; NaN targets stay
    NaN. With a non-contiguous index a row shift is not a trading-day shift. ``horizon`` is a
    session count: derive it with :func:`lag_horizon`, never from a month-based target.
    """
    if horizon < 1 or p < 1:
        raise ValueError("horizon and p must be >= 1")
    return pd.DataFrame({f"{prefix}__lag{k}": y.shift(horizon + k - 1) for k in range(1, p + 1)},
                        index=y.index)


# ---------------------------------------------------------------------------------------------
# Return / excess-return benchmarks
# ---------------------------------------------------------------------------------------------


@register_model
class ZeroReturn(ForecastModel):
    """The forward-curve benchmark: predicted excess (or fixed-contract) return = 0.

    ``predict_quantiles`` returns empirical quantiles of the training target re-centred on 0
    (``center="zero"``, the market-consistent interval) or as-is (``center="none"``).
    """

    name = "zero_return"
    level = 0
    supports_quantiles = True
    default_params: ClassVar[dict[str, Any]] = {"center": "zero"}

    def _validate_params(self) -> None:
        if self.params["center"] not in ("zero", "none"):
            raise ValueError("center must be 'zero' or 'none'")

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        resid = y.to_numpy()
        if self.params["center"] == "zero":
            resid = resid - np.median(resid)
        self.train_values_ = resid

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def _predict_quantiles(self, X: pd.DataFrame, levels: list[float]) -> np.ndarray:
        q = np.quantile(self.train_values_, levels)
        return np.tile(q, (len(X), 1))


@register_model
class NoChange(ForecastModel):
    """Random walk in levels/spreads: forecast = value of ``level_column`` at the origin.

    For T2 the column is F_c(t) of the held contract; for T4 the settlement of month M+k at t; for
    T1 the M1/BOM settlement. Spread-*change* targets use :class:`ZeroReturn` instead.
    """

    name = "no_change"
    level = 0
    default_params: ClassVar[dict[str, Any]] = {"level_column": None}

    def _validate_params(self) -> None:
        if not self.params["level_column"]:
            raise ValueError("no_change needs level_column")

    def required_columns(self) -> list[str]:
        return [self.params["level_column"]]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.params["level_column"]].to_numpy(dtype=float)


@register_model
class RollingMean(ForecastModel):
    """Mean of the last ``window`` training targets, held constant over the prediction block.

    Tests "drift" models. With overlapping h-day labels the last ``horizon`` training rows are
    assumed already purged by the runner. ``window=None`` uses all training rows.
    """

    name = "rolling_mean"
    level = 0
    default_params: ClassVar[dict[str, Any]] = {"window": 250}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        w = self.params["window"]
        tail = y if w is None else y.iloc[-int(w):]
        self.mean_ = float(tail.mean())

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


@register_model
class ARp(ForecastModel):
    """Direct h-step AR(p) on the target's own known lags, p in 0..max_p chosen by AIC on train.

    ``lag_columns`` lists X columns holding lag 1, 2, … of the target known at the origin (see
    :func:`make_target_lags`). All candidate orders are compared on the common sample (rows where
    all ``max_p`` lags exist) so AIC values are comparable; the chosen order is then refit on every
    row where its own lags exist. NaN lags at prediction time are filled with training means.
    """

    name = "ar"
    level = 0
    default_params: ClassVar[dict[str, Any]] = {"lag_columns": None, "max_p": 5,
                                                "fit_intercept": True}

    def _validate_params(self) -> None:
        if not self.params["lag_columns"]:
            raise ValueError("ar needs lag_columns (ordered lag 1, 2, ...)")
        if int(self.params["max_p"]) > len(self.params["lag_columns"]):
            raise ValueError("max_p exceeds the number of lag_columns provided")

    def required_columns(self) -> list[str]:
        return list(self.params["lag_columns"])[: int(self.params["max_p"])]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        cols = self.required_columns()
        max_p = len(cols)
        common = X[cols].notna().all(axis=1)
        if common.sum() < max_p + 20:
            raise InsufficientDataError(f"ar: {int(common.sum())} rows with all {max_p} lags")
        icpt = bool(self.params["fit_intercept"])
        aic = {}
        for p in range(0, max_p + 1):
            if p == 0 and not icpt:
                aic[0] = float(len(y[common]) * np.log(np.mean(y[common].to_numpy() ** 2)))
                continue
            Xp = X.loc[common, cols[:p]].to_numpy() if p else np.empty((int(common.sum()), 0))
            aic[p] = hac_ols(Xp, y[common].to_numpy(), cols[:p], hac_lags=0,
                             fit_intercept=icpt)["aic"]
        self.aic_: dict[int, float] = aic
        self.p_: int = min(aic, key=lambda k: (aic[k], k))
        use = cols[: self.p_]
        self.use_columns_: list[str] = use
        rows = X[use].notna().all(axis=1) if use else pd.Series(True, index=X.index)
        if self.p_ == 0 and not icpt:
            self.coef_ = pd.Series(dtype=float)
        else:
            Xp = X.loc[rows, use].to_numpy() if use else np.empty((int(rows.sum()), 0))
            self.coef_ = hac_ols(Xp, y[rows].to_numpy(), use, hac_lags=self.horizon - 1,
                                 fit_intercept=icpt)["params"]
        self.lag_means_: pd.Series = X[use].mean() if use else pd.Series(dtype=float)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        use = self.use_columns_
        if self.coef_.empty:
            return np.zeros(len(X))
        Z = X[use].fillna(self.lag_means_).to_numpy() if use else np.empty((len(X), 0))
        return _linear_predict(self.coef_, Z, use)


@register_model
class SeasonalMean(ForecastModel):
    """Training mean of the target by calendar month of the origin (market time); months unseen
    in training fall back to the overall training mean. For T1/T4 only (spec §4)."""

    name = "seasonal_mean"
    level = 0
    default_params: ClassVar[dict[str, Any]] = {"min_obs_per_month": 1}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        months = origin_months(X.index)
        g = pd.Series(y.to_numpy(), index=months).groupby(level=0)
        counts = g.size()
        means = g.mean()
        self.overall_ = float(y.mean())
        ok = counts >= int(self.params["min_obs_per_month"])
        self.month_means_: dict[int, float] = {int(m): float(v) for m, v in means[ok].items()}

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.month_means_.get(int(m), self.overall_)
                         for m in origin_months(X.index)])


@register_model
class StorageDeviationOLS(ForecastModel):
    """The "obvious fundamental" benchmark: ``y = a + b·storage_deviation`` (HAC lag h−1).

    NaN regressor values are filled with the training median for both fit and predict.
    """

    name = "storage_deviation_ols"
    level = 0
    family_requirements = ("storage",)
    default_params: ClassVar[dict[str, Any]] = {"column": None}

    def _validate_params(self) -> None:
        if not self.params["column"]:
            raise ValueError("storage_deviation_ols needs the storage deviation column name")

    def required_columns(self) -> list[str]:
        return [self.params["column"]]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        c = self.params["column"]
        x = X[c]
        if x.notna().sum() < 20:
            raise InsufficientDataError(f"storage_deviation_ols: {int(x.notna().sum())} non-NaN {c}")
        self.fill_ = float(x.median())
        res = hac_ols(x.fillna(self.fill_).to_numpy()[:, None], y.to_numpy(), [c],
                      hac_lags=self.horizon - 1)
        self.coef_: pd.Series = res["params"]
        self.tvalues_: pd.Series = res["tvalues"]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        c = self.params["column"]
        return _linear_predict(self.coef_, X[c].fillna(self.fill_).to_numpy()[:, None], [c])


# ---------------------------------------------------------------------------------------------
# Volatility benchmarks (T8). Inputs are vol columns in the target's units (annualised).
# ---------------------------------------------------------------------------------------------


class _VolPassthrough(ForecastModel):
    """Forecast = a vol column known at the origin, optionally rescaled by a train-fitted
    no-intercept factor ``c = Σ y·x / Σ x²`` (``scale="ols"``)."""

    level = 0
    task = "volatility"
    default_params: ClassVar[dict[str, Any]] = {"column": None, "scale": "none"}

    def _validate_params(self) -> None:
        if not self.params["column"]:
            raise ValueError(f"{self.name} needs column")
        if self.params["scale"] not in ("none", "ols"):
            raise ValueError("scale must be 'none' or 'ols'")

    def required_columns(self) -> list[str]:
        return [self.params["column"]]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        x = X[self.params["column"]]
        ok = x.notna()
        self.fill_ = float(x.median())
        self.factor_ = 1.0
        if self.params["scale"] == "ols":
            xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
            self.factor_ = float((xv @ yv) / (xv @ xv))

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.factor_ * X[self.params["column"]].fillna(self.fill_).to_numpy(dtype=float)


@register_model
class TrailingVol(_VolPassthrough):
    """Trailing realised vol over the last h days (h-matched), the principal T8 benchmark."""

    name = "trailing_vol"


@register_model
class EWMAVol(_VolPassthrough):
    """EWMA vol (RiskMetrics-style) computed by the feature layer; decay is a feature parameter."""

    name = "ewma_vol"


@register_model
class HAR(ForecastModel):
    """HAR(1,5,22) (Corsi 2009): ``y = b0 + b_d·RV_d + b_w·RV_w + b_m·RV_m`` fitted by OLS on
    train, optionally in logs (``log=True`` → prediction ``exp(fit)``, no bias correction)."""

    name = "har"
    level = 0
    task = "volatility"
    default_params: ClassVar[dict[str, Any]] = {"rv_columns": None, "log": False}

    def _validate_params(self) -> None:
        cols = self.params["rv_columns"]
        if not cols or len(cols) != 3:
            raise ValueError("har needs rv_columns = [daily, weekly, monthly] column names")

    def required_columns(self) -> list[str]:
        return list(self.params["rv_columns"])

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        cols = self.required_columns()
        ok = X[cols].notna().all(axis=1)
        Z, t = X.loc[ok, cols], y[ok]
        if self.params["log"]:
            if (Z <= 0).any().any() or (t <= 0).any():
                raise ValueError("har log=True needs strictly positive vols")
            Z, t = np.log(Z), np.log(t)
        self.means_: pd.Series = X[cols].median()
        self.coef_: pd.Series = hac_ols(Z.to_numpy(), t.to_numpy(), cols,
                                        hac_lags=self.horizon - 1)["params"]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.required_columns()
        Z = X[cols].fillna(self.means_)
        if self.params["log"]:
            return np.exp(_linear_predict(self.coef_, np.log(Z.to_numpy()), cols))
        return _linear_predict(self.coef_, Z.to_numpy(), cols)


# ---------------------------------------------------------------------------------------------
# Classification benchmark (T9)
# ---------------------------------------------------------------------------------------------


@register_model
class BaseRate(ForecastModel):
    """Training-window frequency of the event; the T9 benchmark. ``predict`` = probability."""

    name = "base_rate"
    level = 0
    task = "classification"
    supports_proba = True
    default_params: ClassVar[dict[str, Any]] = {}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.rate_ = float(y.mean())

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(X)
