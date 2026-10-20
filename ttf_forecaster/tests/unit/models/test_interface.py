"""Interface, registry, inner CV and train-only preprocessing. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import linear_world, origin_index
from ttf_forecaster.data.base import PREDICTION_SCHEMA, validate_frame
from ttf_forecaster.models.interface import (
    InsufficientDataError,
    LeakageGuardError,
    NotFittedError,
    TabularPreprocessor,
    available_models,
    inner_time_splits,
    make_model,
    quantile_column,
    tail_split,
    to_prediction_frame,
)

EXPECTED = {
    "zero_return", "no_change", "rolling_mean", "ar", "seasonal_mean", "storage_deviation_ols",
    "trailing_vol", "ewma_vol", "har", "base_rate", "ols_hac", "ridge", "lasso", "elasticnet",
    "ecm", "lightgbm", "lightgbm_classifier", "xgboost", "catboost", "linear_quantile",
    "lightgbm_quantile", "conformal", "threshold_logistic", "lightgbm_event",
    "constrained_stacking", "performance_weighted",
}


def test_registry_complete_and_levels() -> None:
    reg = available_models()
    assert EXPECTED <= set(reg)
    assert reg["zero_return"].level == 0 and reg["ridge"].level == 1
    assert reg["lightgbm"].level == 2 and reg["conformal"].level == 4
    assert reg["constrained_stacking"].level == 6
    assert reg["storage_deviation_ols"].family_requirements == ("storage",)


def test_make_model_reads_config_and_overrides() -> None:
    m = make_model("rolling_mean_60", horizon=1)
    assert m.params["window"] == 60 and m.name == "rolling_mean"
    assert m.params["seed"] == 20261016
    m2 = make_model("rolling_mean_60", window=5, horizon=20)
    assert m2.params["window"] == 5 and m2.horizon == 20
    assert m.model_id != m2.model_id
    assert make_model("rolling_mean_60", horizon=1).model_id == m.model_id
    with pytest.raises(TypeError, match="unknown parameter"):
        make_model("ridge", horizon=1, not_a_param=1)
    with pytest.raises(KeyError, match="unknown model"):
        make_model("does_not_exist", horizon=1)


def test_every_config_entry_constructs() -> None:
    from ttf_forecaster.models.interface import model_config

    for name in model_config():
        if name in {"xgboost", "catboost"}:
            continue
        cls = available_models()[model_config()[name].get("model", name)]
        # Column-name params are null in config; only check that the param names are accepted.
        assert set(model_config()[name].get("params", {})) <= set(cls.default_params), name


def test_inner_time_splits_gap_and_order() -> None:
    splits = inner_time_splits(200, gap=5, n_splits=3, val_fraction=0.25)
    assert len(splits) == 3
    for tr, va in splits:
        assert tr.max() + 5 < va.min()  # gap >= horizon rows between last train and first val
        assert np.all(np.diff(va) == 1) and tr.min() == 0
    assert splits[-1][1].max() == 199
    with pytest.raises(InsufficientDataError):
        inner_time_splits(20, gap=5)


def test_tail_split() -> None:
    head, tail = tail_split(100, fraction=0.2, gap=3)
    assert tail.min() == 80 and tail.max() == 99 and head.max() == 76


def test_fit_input_validation(lin_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = lin_xy
    m = make_model("zero_return", horizon=1)
    with pytest.raises(NotFittedError):
        m.predict(X)
    with pytest.raises(ValueError, match="time-ordered"):
        m.fit(X.iloc[::-1], y.iloc[::-1])
    y_nan = y.copy()
    y_nan.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.fit(X, y_nan)
    with pytest.raises(ValueError, match="same index"):
        m.fit(X, y.iloc[1:])
    m.fit(X, y)
    with pytest.raises(KeyError):
        m.predict(X.drop(columns=X.columns[0]))


def test_quantile_column_name() -> None:
    assert quantile_column(0.1) == "q0.1" and quantile_column(0.95) == "q0.95"


def test_preprocessor_uses_train_stats_only() -> None:
    X, _ = linear_world(n=300, k=4, nan_frac=0.0)
    train, test = X.iloc[:200], X.iloc[200:] * 10 + 5
    p = TabularPreprocessor().fit(train)
    np.testing.assert_allclose(p.scaler_.mean_, train.mean().to_numpy())
    np.testing.assert_allclose(p.scaler_.scale_, train.std(ddof=0).to_numpy())
    z = p.transform(test)
    np.testing.assert_allclose(z, (test - train.mean()) / train.std(ddof=0))


def test_preprocessor_keeps_all_nan_columns() -> None:
    X, _ = linear_world(n=200, k=4)
    X["price__empty"] = np.nan
    p = TabularPreprocessor().fit(X)
    assert p.output_names_ == list(X.columns)
    assert p.transform(X).shape[1] == X.shape[1]


def test_family_pca_train_only_and_names() -> None:
    X, _ = linear_world(n=300, k=8)
    p = TabularPreprocessor(family_pca={"k": 2}).fit(X.iloc[:200])
    assert [c for c in p.output_names_ if c.startswith("pca__")] == [
        "pca__storage__pc1", "pca__storage__pc2", "pca__price__pc1", "pca__price__pc2"]
    comps = {f: pca.components_.copy() for f, (_, pca) in p.pca_.items()}
    p.transform(X.iloc[200:] * 100)
    for f, (_, pca) in p.pca_.items():
        np.testing.assert_array_equal(pca.components_, comps[f])
    p2 = TabularPreprocessor(family_pca={"k": 2, "families": ["price"]}).fit(X)
    assert "storage__f0" in p2.output_names_ and "pca__price__pc1" in p2.output_names_


def test_to_prediction_frame_matches_schema() -> None:
    idx = origin_index(10)
    point = pd.Series(np.arange(10.0), index=idx)
    lkt = pd.Series(idx + pd.Timedelta(days=7), index=idx)
    out = to_prediction_frame(
        point, target_id="m1_logret_5", horizon=5, model_id="zero_return-abc", fold_id="f0",
        train_end=idx[0] - pd.Timedelta(days=1), label_known_time=lkt, feature_hash="h",
        is_synthetic=True, quantiles=pd.DataFrame({"q0.1": -point}, index=idx))
    validate_frame(out, PREDICTION_SCHEMA)
    assert "q0.1" in out.columns
    with pytest.raises(LeakageGuardError):
        to_prediction_frame(point, target_id="t", horizon=5, model_id="m", fold_id="f",
                            train_end=idx[3], label_known_time=lkt, feature_hash="h",
                            is_synthetic=True)
