"""Level 1 linear models. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import linear_world, origin_index
from ttf_forecaster.models.interface import CointegrationGateError, make_model
from ttf_forecaster.models.linear import distributed_lags


def test_ols_hac_matches_lstsq_and_reports_t() -> None:
    X, y = linear_world(n=400, k=3, nan_frac=0.0)
    m = make_model("ols_hac", horizon=5).fit(X, y)
    A = np.column_stack([np.ones(len(y)), X.to_numpy()])
    b = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(m.coef_.to_numpy(), b, atol=1e-10)
    assert m.hac_lags_ == 4
    np.testing.assert_allclose(m.tvalues_, m.coef_ / m.bse_)
    assert set(m.summary_frame().columns) == {"coef", "se_hac", "t", "p"}


def test_ols_all_nan_column_keeps_mapping() -> None:
    X, y = linear_world(n=300, k=3, nan_frac=0.0)
    X.insert(1, "lng__not_yet_available", np.nan)
    m = make_model("ols_hac", horizon=1).fit(X, y)
    assert list(m.coef_.index) == ["const", *X.columns]
    assert np.isnan(m.coef_["lng__not_yet_available"])
    assert m.coef_["storage__f0"] == pytest.approx(0.5, abs=0.15)


@pytest.mark.parametrize("name", ["ridge", "lasso", "elasticnet"])
def test_penalised_deterministic_and_train_only(name: str) -> None:
    X, y = linear_world(n=500)
    tr = slice(0, 400)
    m1 = make_model(name, horizon=5).fit(X.iloc[tr], y.iloc[tr])
    m2 = make_model(name, horizon=5).fit(X.iloc[tr], y.iloc[tr])
    pd.testing.assert_series_equal(m1.predict(X.iloc[400:]), m2.predict(X.iloc[400:]))
    # Scaler moments equal the (median-imputed) TRAIN moments, never the test ones.
    imputed = X.iloc[tr].fillna(X.iloc[tr].median())
    np.testing.assert_allclose(m1.prep_.scaler_.mean_, imputed.mean().to_numpy())
    assert m1.y_mean_ == pytest.approx(y.iloc[tr].mean())
    # Perturbing the test rows cannot change anything fitted.
    coef = m1.coef_.copy()
    m1.predict(X.iloc[400:] * 1000)
    pd.testing.assert_series_equal(m1.coef_, coef)
    assert m1.alpha_ in m1.cv_table_["value"].tolist()


def test_lasso_finds_planted_column() -> None:
    X, y = linear_world(n=600, k=8, beta=1.0, noise=0.5)
    m = make_model("lasso", horizon=1).fit(X, y)
    assert m.coef_.abs().idxmax() == "storage__f0"


def test_ridge_family_pca_path() -> None:
    X, y = linear_world(n=400, k=8)
    m = make_model("ridge_family_pca", horizon=1).fit(X, y)
    assert all(c.startswith("pca__") for c in m.coef_.index)
    assert m.predict(X).notna().all()


def _random_walks(n: int, seed: int, cointegrated: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).cumsum()
    if cointegrated:
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.7 * u[t - 1] + rng.normal()
        lvl = 1.0 + 0.8 * x + u
    else:
        lvl = rng.normal(size=n).cumsum()
    return pd.DataFrame({"curve__log_m1": lvl, "cross_commodity__log_jkm": x},
                        index=origin_index(n))


def test_ecm_gate_refuses_non_cointegrated() -> None:
    X = _random_walks(600, seed=5, cointegrated=False)
    y = X["curve__log_m1"].diff(5).shift(-5).fillna(0.0)
    m = make_model("ecm", level_column="curve__log_m1",
                   x_level_columns=["cross_commodity__log_jkm"], horizon=5)
    with pytest.raises(CointegrationGateError, match="Engle"):
        m.fit(X, y)


def test_ecm_fits_when_cointegrated() -> None:
    X = _random_walks(600, seed=5, cointegrated=True)
    y = X["curve__log_m1"].diff(5).shift(-5).fillna(0.0)
    m = make_model("ecm", level_column="curve__log_m1",
                   x_level_columns=["cross_commodity__log_jkm"], horizon=5).fit(X, y)
    assert m.gate_["pvalue"] < 0.05
    assert m.gamma_ < 0  # error correction pulls back to equilibrium
    assert m.predict(X).notna().all()


def test_distributed_lags() -> None:
    X, _ = linear_world(n=10, k=2, nan_frac=0.0)
    out = distributed_lags(X, ["storage__f0"], [1, 3])
    assert out["storage__f0__lag3"].iloc[5] == X["storage__f0"].iloc[2]
    with pytest.raises(ValueError):
        distributed_lags(X, ["storage__f0"], [0])
