"""Level 2 boosting. SYNTHETIC data only."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import linear_world
from ttf_forecaster.models.interface import make_model


def test_lightgbm_deterministic_with_early_stopping() -> None:
    X, y = linear_world(n=600)
    a = make_model("lightgbm", horizon=5).fit(X.iloc[:500], y.iloc[:500])
    b = make_model("lightgbm", horizon=5).fit(X.iloc[:500], y.iloc[:500])
    pd.testing.assert_series_equal(a.predict(X.iloc[500:]), b.predict(X.iloc[500:]))
    assert a.best_iteration_ is not None and a.best_iteration_ >= 1
    assert a.effective_params_["n_estimators"] == a.best_iteration_
    assert list(a.feature_importance_.index) == list(X.columns)


def test_lightgbm_no_refit_and_small_window() -> None:
    X, y = linear_world(n=600)
    m = make_model("lightgbm", horizon=1, refit_full=False).fit(X, y)
    assert m.predict(X).notna().all()
    small = make_model("lightgbm", horizon=1).fit(X.iloc[:40], y.iloc[:40])  # too small for ES split
    assert small.best_iteration_ is None


def test_lightgbm_family_pca() -> None:
    X, y = linear_world(n=400, k=8)
    m = make_model("lightgbm", horizon=1, family_pca={"k": 2}).fit(X, y)
    assert all(c.startswith("pca__") for c in m.feature_importance_.index)


def test_lightgbm_classifier_proba() -> None:
    X, y = linear_world(n=500)
    m = make_model("lightgbm_classifier", horizon=1).fit(X, (y > 0).astype(int))
    p = m.predict_proba(X)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("name,pkg", [("xgboost", "xgboost"), ("catboost", "catboost")])
def test_optional_boosters_raise_clear_import_error(name: str, pkg: str) -> None:
    if importlib.util.find_spec(pkg) is not None:
        pytest.skip(f"{pkg} installed")
    with pytest.raises(ImportError, match="boosting_extra"):
        make_model(name, horizon=1)


def test_unused_seed_changes_nothing_structural() -> None:
    X, y = linear_world(n=300)
    m = make_model("lightgbm", horizon=1, seed=1).fit(X, y)
    assert np.isfinite(m.predict(X)).all()
