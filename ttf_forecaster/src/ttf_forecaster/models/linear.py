"""Level 1 linear models: OLS with Newey–West HAC, penalised regressions with time-ordered CV,
distributed-lag helper, and an error-correction model behind an Engle–Granger gate.

ARIMAX is **not implemented**: a SARIMAX on overlapping h-day targets would put y(t-1, h) — not
known at origin t for h > 1 — into the AR terms, and a dynamic multi-step forecast across a test
block needs contiguous indices the walk-forward runner does not guarantee. The AR(p) benchmark in
``baselines.py`` covers the target's own dynamics PIT-safely through lag columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ttf_forecaster.models.interface import (
    CointegrationGateError,
    ForecastModel,
    InsufficientDataError,
    TabularPreprocessor,
    grid_search_time_cv,
    register_model,
)

# ---------------------------------------------------------------------------------------------
# HAC OLS core (shared with baselines)
# ---------------------------------------------------------------------------------------------


def hac_ols(X: np.ndarray, y: np.ndarray, names: Sequence[str], *, hac_lags: int,
            fit_intercept: bool = True) -> dict[str, Any]:
    """OLS with Newey–West (Bartlett) HAC standard errors, ``maxlags = hac_lags``.

    Zero-variance columns (e.g. a feature all-NaN in the window and imputed to a constant) are
    dropped from the fit and reported with NaN coefficient, so the name↔coefficient mapping never
    shifts. Returns a dict with ``params`` (Series incl. ``const``), ``bse``, ``tvalues``,
    ``pvalues``, ``nobs``, ``rsquared``, ``aic``, ``dropped``.
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    names = list(names)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X shape and names disagree")
    keep = [j for j in range(X.shape[1]) if np.nanstd(X[:, j]) > 0]
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    design = X[:, keep]
    kept_names = [names[j] for j in keep]
    if fit_intercept:
        design = np.column_stack([np.ones(len(y)), design])
        kept_names = ["const", *kept_names]
    if design.shape[1] == 0:
        raise ValueError("hac_ols: no regressors and no intercept")
    if len(y) <= design.shape[1]:
        raise InsufficientDataError(f"OLS needs more rows ({len(y)}) than regressors "
                                    f"({design.shape[1]})")
    cov_type, cov_kwds = ("HAC", {"maxlags": int(hac_lags)}) if hac_lags > 0 else ("HC0", {})
    res = sm.OLS(y, design).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    all_names = (["const"] if fit_intercept else []) + names

    def _full(vals: np.ndarray) -> pd.Series:
        s = pd.Series(np.nan, index=all_names, dtype=float)
        s.loc[kept_names] = np.asarray(vals, dtype=float)
        return s

    return {
        "params": _full(res.params),
        "bse": _full(res.bse),
        "tvalues": _full(res.tvalues),
        "pvalues": _full(res.pvalues),
        "nobs": int(res.nobs),
        "rsquared": float(res.rsquared),
        "aic": float(res.aic),
        "dropped": dropped,
    }


def _linear_predict(params: pd.Series, X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    coef = params.reindex(list(names)).fillna(0.0).to_numpy()
    const = float(params.get("const", 0.0)) if "const" in params.index else 0.0
    if np.isnan(const):
        const = 0.0
    return const + np.asarray(X, dtype=float) @ coef


# ---------------------------------------------------------------------------------------------
# OLS with HAC inference
# ---------------------------------------------------------------------------------------------


@register_model
class OLSHAC(ForecastModel):
    """OLS point forecast with Newey–West HAC inference (lag = ``horizon - 1`` by default).

    Inputs are median-imputed (train medians, empty columns kept) and **not** standardised by
    default so coefficients stay in native units. Exposes ``coef_``, ``bse_``, ``tvalues_``,
    ``pvalues_`` (Series indexed by output feature names plus ``const``) and ``summary_frame()``.
    """

    name = "ols_hac"
    level = 1
    default_params: ClassVar[dict[str, Any]] = {
        "hac_lags": None,  # None -> horizon - 1
        "fit_intercept": True,
        "standardize": False,
        "family_pca": None,
    }

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.prep_ = TabularPreprocessor(impute=True, standardize=bool(self.params["standardize"]),
                                         family_pca=self.params["family_pca"])
        Z = self.prep_.fit_transform(X)
        lags = self.params["hac_lags"]
        lags = self.horizon - 1 if lags is None else int(lags)
        self.hac_lags_ = lags
        self.result_ = hac_ols(Z, y.to_numpy(), self.prep_.output_names_, hac_lags=lags,
                               fit_intercept=bool(self.params["fit_intercept"]))
        self.coef_: pd.Series = self.result_["params"]
        self.bse_: pd.Series = self.result_["bse"]
        self.tvalues_: pd.Series = self.result_["tvalues"]
        self.pvalues_: pd.Series = self.result_["pvalues"]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return _linear_predict(self.coef_, self.prep_.transform(X), self.prep_.output_names_)

    def summary_frame(self) -> pd.DataFrame:
        """coef / HAC se / t / p per regressor."""
        return pd.DataFrame({"coef": self.coef_, "se_hac": self.bse_, "t": self.tvalues_,
                             "p": self.pvalues_})


# ---------------------------------------------------------------------------------------------
# Penalised regressions with time-ordered CV inside fit
# ---------------------------------------------------------------------------------------------


class _PenalisedLinear(ForecastModel):
    """Ridge / Lasso / ElasticNet on standardised X and standardised y.

    The penalty is chosen by :func:`grid_search_time_cv` (expanding inner folds over the last
    ``val_fraction`` of training rows, gap = ``horizon``), refitting the preprocessor on every inner
    training slice. ``y`` is centred and scaled by its training moments so one alpha grid is
    scale-free across targets. Grids are ordered from most to least regularised (ties → simpler).
    """

    level = 1
    default_params: ClassVar[dict[str, Any]] = {
        "alphas": None,
        "l1_ratio": 0.5,
        "cv_splits": 3,
        "cv_val_fraction": 0.25,
        "family_pca": None,
        "max_iter": 10000,
        # D-032 F5: a floor on the penalty (no alpha below it is ever chosen) and the inner-CV
        # selection rule ("min" or "one_se"). Short windows with p/n ~ 0.1 otherwise pick a
        # near-OLS alpha and overfit (pooled OOS R^2 ~ -1.2 in the positive-control world). The
        # floor is the active fix; "one_se" is available but off (see modelling_log D-033).
        "min_alpha": 0.0,
        "selection": "min",
    }
    _default_alphas: ClassVar[tuple[float, ...]] = ()

    def _alphas(self) -> list[float]:
        a = self.params["alphas"]
        grid = list(self._default_alphas) if a is None else [float(v) for v in a]
        if not grid or min(grid) <= 0:
            raise ValueError(f"{self.name}: alphas must be positive")
        floor = float(self.params["min_alpha"])
        grid = [v for v in grid if v >= floor]
        if not grid:
            raise ValueError(f"{self.name}: every alpha is below min_alpha={floor}")
        return sorted(grid, reverse=True)

    def _estimator(self, alpha: float, n: int) -> Any:
        raise NotImplementedError

    def _fit_core(self, X: pd.DataFrame, y: pd.Series, alpha: float
                  ) -> tuple[TabularPreprocessor, Any, float, float]:
        prep = TabularPreprocessor(impute=True, standardize=True,
                                   family_pca=self.params["family_pca"])
        Z = prep.fit_transform(X)
        mu = float(y.mean())
        sd = float(y.std(ddof=0)) or 1.0
        est = self._estimator(alpha, len(y))
        est.fit(Z, (y.to_numpy() - mu) / sd)
        return prep, est, mu, sd

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        def fit_score(Xt: pd.DataFrame, yt: pd.Series, Xv: pd.DataFrame, yv: pd.Series,
                      alpha: float) -> float:
            prep, est, mu, sd = self._fit_core(Xt, yt, alpha)
            pred = mu + sd * est.predict(prep.transform(Xv))
            return float(np.mean((yv.to_numpy() - pred) ** 2))

        alphas = self._alphas()
        self.alpha_, self.cv_table_ = grid_search_time_cv(
            X, y, alphas, fit_score, gap=self.horizon, n_splits=int(self.params["cv_splits"]),
            val_fraction=float(self.params["cv_val_fraction"]),
            label_known_time=self._lkt_for(X), rule=str(self.params["selection"]))
        self.alpha_at_boundary_: bool = self.alpha_ in (alphas[0], alphas[-1])
        self.prep_, self.est_, self.y_mean_, self.y_scale_ = self._fit_core(X, y, self.alpha_)
        self.coef_: pd.Series = pd.Series(self.est_.coef_ * self.y_scale_,
                                          index=self.prep_.output_names_,
                                          name="coef_per_train_sd")

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.y_mean_ + self.y_scale_ * self.est_.predict(self.prep_.transform(X))


@register_model
class RidgeTS(_PenalisedLinear):
    """Ridge. sklearn's alpha multiplies the *sum* of squares, so the grid value is × n_rows."""

    name = "ridge"
    _default_alphas = tuple(np.logspace(-4, 2, 13).tolist())

    def _estimator(self, alpha: float, n: int) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=alpha * n)


@register_model
class LassoTS(_PenalisedLinear):
    """Lasso on standardised data (alpha on the 1/(2n) scale, max useful alpha ≈ max |corr|)."""

    name = "lasso"
    _default_alphas = tuple(np.logspace(-4, 0, 13).tolist())

    def _estimator(self, alpha: float, n: int) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(alpha=alpha, max_iter=int(self.params["max_iter"]),
                     random_state=self.seed, selection="cyclic")


@register_model
class ElasticNetTS(_PenalisedLinear):
    """Elastic net with fixed ``l1_ratio`` (from config) and CV'd alpha."""

    name = "elasticnet"
    _default_alphas = tuple(np.logspace(-4, 0, 13).tolist())

    def _estimator(self, alpha: float, n: int) -> Any:
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=alpha, l1_ratio=float(self.params["l1_ratio"]),
                          max_iter=int(self.params["max_iter"]), random_state=self.seed,
                          selection="cyclic")


# ---------------------------------------------------------------------------------------------
# Distributed lags
# ---------------------------------------------------------------------------------------------


def distributed_lags(df: pd.DataFrame, columns: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    """Return ``df`` with ``<col>__lag<k>`` columns (row shifts by ``k`` >= 1).

    Apply to the **full** origin-grid feature frame before splitting: a positive shift only looks
    backwards, so it is PIT-safe, and building lags after splitting would lose the first ``k`` test
    rows. Assumes one row per trading day; with gaps in the index a row lag is not a day lag.
    """
    if any(int(k) < 1 for k in lags):
        raise ValueError("lags must be >= 1 (a lag of 0 or a lead would not be a lag)")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"distributed_lags: columns {missing} not in frame")
    out = df.copy()
    for c in columns:
        for k in lags:
            out[f"{c}__lag{int(k)}"] = df[c].shift(int(k))
    return out


# ---------------------------------------------------------------------------------------------
# Error-correction model behind an Engle–Granger gate
# ---------------------------------------------------------------------------------------------


def engle_granger_test(y_level: pd.Series, x_levels: pd.DataFrame, *, trend: str = "c"
                       ) -> dict[str, float]:
    """Engle–Granger two-step cointegration test (statsmodels ``coint``, MacKinnon p-values)."""
    from statsmodels.tsa.stattools import coint

    stat, pvalue, crit = coint(y_level.to_numpy(dtype=float), x_levels.to_numpy(dtype=float),
                               trend=trend, autolag="aic")
    return {"stat": float(stat), "pvalue": float(pvalue), "crit_5pct": float(crit[1])}


@register_model
class ECM(ForecastModel):
    """Single-equation error-correction model for an h-day change, gated on cointegration.

    Step 0 (gate): Engle–Granger test of ``level_column`` on ``x_level_columns`` over the training
    rows; if ``p >= gate_pvalue`` the fit **refuses** with :class:`CointegrationGateError`.
    Step 1: long-run OLS ``level = a + b·x_levels`` → equilibrium error ``u_t``.
    Step 2: ``y(t,h) = c + γ·u_t + Σ δ_j·short_run_j + ε`` with HAC (lag h−1) inference.
    ``gamma_`` should be negative for error correction; it is reported, not forced.
    """

    name = "ecm"
    level = 1
    default_params: ClassVar[dict[str, Any]] = {
        "level_column": None,
        "x_level_columns": None,
        "short_run_columns": [],
        "gate_pvalue": 0.05,
        "trend": "c",
    }

    def _validate_params(self) -> None:
        if not self.params["level_column"] or not self.params["x_level_columns"]:
            raise ValueError("ecm needs level_column and x_level_columns")

    def required_columns(self) -> list[str]:
        return [self.params["level_column"], *self.params["x_level_columns"],
                *self.params["short_run_columns"]]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        lvl, xs = self.params["level_column"], list(self.params["x_level_columns"])
        sr = list(self.params["short_run_columns"])
        ok = X[[lvl, *xs]].notna().all(axis=1)
        if ok.sum() < 50:
            raise InsufficientDataError(f"ecm: only {int(ok.sum())} complete level rows")
        self.gate_ = engle_granger_test(X.loc[ok, lvl], X.loc[ok, xs], trend=self.params["trend"])
        if not self.gate_["pvalue"] < float(self.params["gate_pvalue"]):
            raise CointegrationGateError(
                f"ecm refused: Engle–Granger p={self.gate_['pvalue']:.3f} >= "
                f"{self.params['gate_pvalue']} for {lvl} on {xs} over the training window; "
                "no cointegration, so an error-correction term would be spurious"
            )
        lr = hac_ols(X.loc[ok, xs].to_numpy(), X.loc[ok, lvl].to_numpy(), xs, hac_lags=0)
        self.long_run_: pd.Series = lr["params"]
        u = X[lvl] - _linear_predict(self.long_run_, X[xs].to_numpy(), xs)
        Z = pd.concat([u.rename("ecm__u"), X[sr]], axis=1)
        self.sr_prep_ = TabularPreprocessor(impute=True, standardize=False)
        Zt = self.sr_prep_.fit_transform(Z)
        res = hac_ols(Zt, y.to_numpy(), self.sr_prep_.output_names_,
                      hac_lags=self.horizon - 1)
        self.coef_: pd.Series = res["params"]
        self.tvalues_: pd.Series = res["tvalues"]
        self.gamma_: float = float(self.coef_["ecm__u"])

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        lvl, xs = self.params["level_column"], list(self.params["x_level_columns"])
        u = X[lvl] - _linear_predict(self.long_run_, X[xs].to_numpy(), xs)
        Z = pd.concat([u.rename("ecm__u"), X[list(self.params["short_run_columns"])]], axis=1)
        return _linear_predict(self.coef_, self.sr_prep_.transform(Z), self.sr_prep_.output_names_)
