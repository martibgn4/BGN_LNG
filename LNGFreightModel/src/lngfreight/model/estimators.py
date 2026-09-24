"""The fitted models.

Each is a sklearn Pipeline so that imputation and scaling are fitted inside
the training window and applied to the test block, never the other way round.
Imputing with a full-sample median is a quiet, very effective leak - the
median of a series that has not happened yet is information.

Hyperparameters come from config/model.yaml. No float literal appears here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lngfreight.config import get

log = logging.getLogger(__name__)


class SklearnModel:
    """Thin adapter so every estimator and benchmark has the same surface."""

    def __init__(self, name: str, pipeline: Pipeline) -> None:
        self.name = name
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        self.pipeline.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X.to_numpy(dtype=float))

    def coefficients(self, columns: list[str]) -> pd.Series | None:
        final = self.pipeline.steps[-1][1]
        if not hasattr(final, "coef_"):
            return None
        return pd.Series(np.ravel(final.coef_), index=columns)


def _preprocessing() -> list:
    """Median impute, then standardise. Both fitted on the training window.

    ``keep_empty_features=True`` matters more than it looks. The Baltic
    columns are entirely absent before 2024-02, and by default SimpleImputer
    DROPS an all-NaN column rather than imputing it - which silently changes
    the number of features between refits, breaks the mapping from
    coefficients back to column names, and means the model quietly trains on
    a different design matrix in early windows than in late ones. Keeping
    them turns an absent feature into a constant zero, which carries no
    information and says so.
    """
    return [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]


def _ridge() -> SklearnModel:
    cfg = get("model", "estimators", "ridge")
    steps = _preprocessing() if cfg["standardise"] else [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    steps.append(("model", RidgeCV(
        alphas=[float(a) for a in cfg["alphas"]],
        cv=TimeSeriesSplit(n_splits=int(cfg["cv_splits"])),
    )))
    return SklearnModel("ridge", Pipeline(steps))


def _huber() -> SklearnModel:
    cfg = get("model", "estimators", "huber")
    steps = _preprocessing()
    steps.append(("model", HuberRegressor(
        epsilon=float(cfg["epsilon"]),
        alpha=float(cfg["alpha"]),
        max_iter=int(cfg["max_iter"]),
    )))
    return SklearnModel("huber", Pipeline(steps))


def _lightgbm() -> SklearnModel | None:
    cfg = get("model", "estimators", "lightgbm")
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        log.warning("lightgbm is not installed; the non-linear model is skipped "
                    "rather than silently replaced by a linear one")
        return None
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]  # trees need no scaling
    steps.append(("model", LGBMRegressor(
        n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        num_leaves=int(cfg["num_leaves"]),
        max_depth=int(cfg["max_depth"]),
        min_child_samples=int(cfg["min_child_samples"]),
        subsample=float(cfg["subsample"]),
        subsample_freq=int(cfg["subsample_freq"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        verbosity=int(cfg["verbosity"]),
        random_state=0,
    )))
    return SklearnModel("lightgbm", Pipeline(steps))


class ColumnSubsetModel(SklearnModel):
    """A model that sees only the feature families it is given.

    Used for the own-lags benchmark: the identical ridge, fitted on freight's
    own past and nothing else. Holding the estimator fixed and varying only
    the information set is what makes the comparison attributable to the
    fundamental features rather than to a change of model.
    """

    def __init__(self, name: str, pipeline: Pipeline, families: list[str]) -> None:
        super().__init__(name, pipeline)
        self.families = list(families)
        self.columns_: list[str] = []

    def _subset(self, X: pd.DataFrame) -> pd.DataFrame:
        from lngfreight.features import family_of
        return X[[c for c in X.columns if family_of(c) in self.families]]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnSubsetModel":
        subset = self._subset(X)
        if subset.empty or subset.shape[1] == 0:
            raise ValueError(
                f"{self.name}: no columns match families {self.families}"
            )
        self.columns_ = list(subset.columns)
        return super().fit(subset, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return super().predict(X[self.columns_])

    def coefficients(self, columns: list[str]) -> pd.Series | None:
        return super().coefficients(self.columns_)


def own_lags_model() -> ColumnSubsetModel | None:
    """The own-lags benchmark: ridge on the freight family alone."""
    cfg = get("model", "benchmarks", "own_lags")
    if not cfg.get("enabled"):
        return None
    ridge = _ridge()
    return ColumnSubsetModel("own_lags", ridge.pipeline, cfg["families"])


def _family_pca() -> SklearnModel | None:
    """Ridge on family scores: each fundamental family compressed to k PCs.

    Every step - impute, scale, PCA - is fitted inside the training window by
    the surrounding Pipeline, so no test-block variance reaches the rotation.
    Column ORDER is fixed at construction from the training frame, which is
    why this model carries its own column list rather than trusting the
    caller to present columns identically each refit.
    """
    cfg = get("model", "estimators", "family_pca_ridge")
    if not cfg.get("enabled"):
        return None
    return FamilyPcaModel("family_pca_ridge", int(cfg["components_per_family"]),
                          list(cfg["passthrough_families"]))


class FamilyPcaModel(SklearnModel):
    def __init__(self, name: str, components: int,
                 passthrough: list[str]) -> None:
        self.name = name
        self.components = components
        self.passthrough = list(passthrough)
        self.columns_: list[str] = []
        self.pipeline: Pipeline | None = None

    def _build(self, X: pd.DataFrame) -> Pipeline:
        from lngfreight.features import family_of
        families: dict[str, list[int]] = {}
        for position, column in enumerate(X.columns):
            families.setdefault(family_of(column), []).append(position)

        blocks = []
        for family, positions in sorted(families.items()):
            if family in self.passthrough:
                blocks.append((f"raw_{family}", "passthrough", positions))
                continue
            # Never ask for more components than the family has columns.
            k = min(self.components, len(positions))
            blocks.append((f"pca_{family}", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=k, random_state=0)),
            ]), positions))

        ridge_cfg = get("model", "estimators", "ridge")
        return Pipeline([
            ("families", ColumnTransformer(blocks, remainder="drop")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=[float(a) for a in ridge_cfg["alphas"]],
                              cv=TimeSeriesSplit(n_splits=int(ridge_cfg["cv_splits"])))),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FamilyPcaModel":
        self.columns_ = list(X.columns)
        self.pipeline = self._build(X)
        self.pipeline.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError(f"{self.name} has not been fitted")
        return self.pipeline.predict(X[self.columns_].to_numpy(dtype=float))

    def coefficients(self, columns: list[str]) -> pd.Series | None:
        # Weights live on rotated family scores, not on the original columns,
        # so they are not comparable to the other models' and are not
        # reported rather than being reported misleadingly.
        return None


_BUILDERS = {"ridge": _ridge, "huber": _huber, "lightgbm": _lightgbm,
             "family_pca_ridge": _family_pca}


def build_estimators() -> list[SklearnModel]:
    """Every estimator switched on in config, in declaration order."""
    out: list[SklearnModel] = []
    for name, builder in _BUILDERS.items():
        if not get("model", "estimators", name, "enabled"):
            continue
        model = builder()
        if model is not None:
            out.append(model)
    if not out:
        raise ValueError("no estimators are enabled in config/model.yaml")
    return out
