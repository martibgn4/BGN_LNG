"""Level 2 gradient boosting: LightGBM regressor / classifier with conservative defaults.

* Shallow trees, strong L2, large ``min_child_samples``, column/row subsampling — defaults live in
  ``configs/models.yaml``.
* Early stopping uses only the **last** ``es_fraction`` of the training rows as validation, with
  ``horizon`` rows dropped between the fitting head and the validation tail. By default the model
  is then refit on all training rows with the selected number of rounds.
* Deterministic: ``seed``, ``deterministic=True``, ``force_row_wise=True``, ``n_jobs=1``.
* NaN is handled natively; with ``family_pca`` the inputs are imputed/standardised/PCA'd on train.

XGBoost / CatBoost wrappers exist only to give a clear error: those packages are optional extras
(``pip install ttf_forecaster[boosting_extra]``) and are not installed here (modelling_log D-005).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ttf_forecaster.models.interface import (
    ForecastModel,
    InsufficientDataError,
    TabularPreprocessor,
    register_model,
)

_LGB_DEFAULTS: dict[str, Any] = {
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
}


def _require_lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - installed in the research env
        raise ImportError(
            "LightGBM models need the 'boosting' extra: pip install ttf_forecaster[boosting]"
        ) from exc
    return lightgbm


def eval_kwargs(Z_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    """Validation-set kwargs for ``LGBM*.fit``: ``eval_X/eval_y`` on LightGBM >= 4.6 (where
    ``eval_set`` is deprecated), ``eval_set`` on older 4.x."""
    import inspect

    lgb = _require_lightgbm()
    if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
        return {"eval_X": (Z_val,), "eval_y": (y_val,)}
    return {"eval_set": [(Z_val, y_val)]}


def lgb_params(params: dict[str, Any], seed: int, n_train: int) -> dict[str, Any]:
    """sklearn-API LightGBM kwargs with determinism flags. ``min_child_samples`` is capped at
    ``n_train // 4`` so tiny windows still fit (it is reported in ``effective_params_``)."""
    out = {k: params[k] for k in _LGB_DEFAULTS}
    out["min_child_samples"] = int(max(5, min(int(out["min_child_samples"]), n_train // 4)))
    out.update(random_state=int(seed), n_jobs=1, deterministic=True, force_row_wise=True,
               verbose=-1)
    return out


class _LightGBMBase(ForecastModel):
    level = 2
    default_params: ClassVar[dict[str, Any]] = {
        **_LGB_DEFAULTS,
        "early_stopping_rounds": 50,
        "es_fraction": 0.2,
        "refit_full": True,
        "family_pca": None,
    }

    def _make(self, n_train: int, **overrides: Any) -> Any:
        raise NotImplementedError

    def _fit_kwargs(self) -> dict[str, Any]:
        return {}

    def _prep_fit(self, X: pd.DataFrame) -> np.ndarray:
        if self.params["family_pca"]:
            self.prep_: TabularPreprocessor | None = TabularPreprocessor(
                impute=True, standardize=True, family_pca=self.params["family_pca"])
            Z = self.prep_.fit_transform(X)
            self.model_features_: list[str] = self.prep_.output_names_
            return Z
        self.prep_ = None
        self.model_features_ = list(X.columns)
        return X.to_numpy(dtype=float)

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        return self.prep_.transform(X) if self.prep_ is not None else X.to_numpy(dtype=float)

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        lgb = _require_lightgbm()
        Z = self._prep_fit(X)
        yv = y.to_numpy()
        self.best_iteration_: int | None = None
        try:
            head, tail = self._tail_split(X, fraction=float(self.params["es_fraction"]))
        except InsufficientDataError:
            head = tail = None
        if head is not None and int(self.params["early_stopping_rounds"]) > 0:
            # PCA/imputation fitted on all training rows is still train-only; the ES tail is part
            # of the training window, so no test information is involved.
            est = self._make(len(head))
            est.fit(Z[head], yv[head], **eval_kwargs(Z[tail], yv[tail]),
                    callbacks=[lgb.early_stopping(int(self.params["early_stopping_rounds"]),
                                                  verbose=False)],
                    **self._fit_kwargs())
            self.best_iteration_ = int(est.best_iteration_ or self.params["n_estimators"])
            if not self.params["refit_full"]:
                self.est_ = est
                self._finish()
                return
        n_rounds = self.best_iteration_ or int(self.params["n_estimators"])
        self.est_ = self._make(len(yv), n_estimators=max(1, n_rounds))
        self.est_.fit(Z, yv, **self._fit_kwargs())
        self._finish()

    def _finish(self) -> None:
        self.effective_params_ = self.est_.get_params()
        imp = self.est_.booster_.feature_importance(importance_type="gain")
        self.feature_importance_: pd.Series = pd.Series(imp, index=self.model_features_,
                                                        name="gain")


@register_model
class LightGBMRegressor(_LightGBMBase):
    """LightGBM L2 regressor (point forecast)."""

    name = "lightgbm"

    def _make(self, n_train: int, **overrides: Any) -> Any:
        lgb = _require_lightgbm()
        kw = lgb_params(self.params, self.seed, n_train)
        kw.update(overrides)
        return lgb.LGBMRegressor(**kw)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est_.predict(self._prep(X))


@register_model
class LightGBMClassifier(_LightGBMBase):
    """LightGBM binary classifier; ``predict`` returns the (uncalibrated) event probability.
    For calibrated probabilities use ``probabilistic.LightGBMEventClassifier``."""

    name = "lightgbm_classifier"
    task = "classification"
    supports_proba = True

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if y.nunique() < 2:
            raise ValueError("lightgbm_classifier: training labels contain a single class")
        super()._fit(X, y)

    def _make(self, n_train: int, **overrides: Any) -> Any:
        lgb = _require_lightgbm()
        kw = lgb_params(self.params, self.seed, n_train)
        kw.update(overrides)
        return lgb.LGBMClassifier(**kw)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.est_.predict_proba(self._prep(X))[:, 1]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_proba(X)


class _OptionalBoosterStub(ForecastModel):
    """Wrapper for an optional booster that is not installed. Construction raises ImportError."""

    level = 2
    package: ClassVar[str] = ""
    default_params: ClassVar[dict[str, Any]] = {"estimator_params": {}}

    def __init__(self, **params: Any) -> None:
        try:
            __import__(self.package)
        except ImportError as exc:
            raise ImportError(
                f"model {self.name!r} needs the optional package {self.package!r}: "
                "pip install ttf_forecaster[boosting_extra] (not installed; modelling_log D-005)"
            ) from exc
        super().__init__(**params)
        raise NotImplementedError(  # pragma: no cover - only reachable once installed
            f"{self.package} is installed but the {self.name} wrapper is not implemented in v1; "
            "LightGBM answers the same question (D-005)"
        )

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # pragma: no cover
        raise NotImplementedError

    def _predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@register_model
class XGBoostRegressor(_OptionalBoosterStub):
    name = "xgboost"
    package = "xgboost"


@register_model
class CatBoostRegressor(_OptionalBoosterStub):
    name = "catboost"
    package = "catboost"
