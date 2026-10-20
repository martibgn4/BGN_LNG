"""Level 0 benchmarks: exact values against hand computations. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import origin_index
from ttf_forecaster.models.baselines import make_target_lags
from ttf_forecaster.models.interface import make_model


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), x])
    return np.linalg.lstsq(A, y, rcond=None)[0]


@pytest.fixture
def frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    n = 400
    idx = origin_index(n)
    X = pd.DataFrame({"curve__m1": 30 + rng.normal(size=n).cumsum(),
                      "storage__dev": rng.normal(size=n),
                      "price__rv1": np.abs(rng.normal(0.5, 0.1, n)),
                      "price__rv5": np.abs(rng.normal(0.5, 0.05, n)),
                      "price__rv22": np.abs(rng.normal(0.5, 0.02, n))}, index=idx)
    y = pd.Series(0.3 * X["storage__dev"].to_numpy() + rng.normal(size=n), index=idx)
    return X, y


def test_zero_return(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = frame
    m = make_model("zero_return", horizon=1).fit(X, y)
    assert (m.predict(X) == 0).all()
    q = m.predict_quantiles(X.iloc[:3], [0.1, 0.5, 0.9])
    assert list(q.columns) == ["q0.1", "q0.5", "q0.9"]
    assert q["q0.5"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_no_change(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = frame
    m = make_model("no_change", horizon=1, level_column="curve__m1").fit(X, y)
    pd.testing.assert_series_equal(m.predict(X), X["curve__m1"], check_names=False)
    with pytest.raises(ValueError):
        make_model("no_change", horizon=1)


def test_rolling_mean_exact(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = frame
    m = make_model("rolling_mean_60", horizon=1).fit(X.iloc[:300], y.iloc[:300])
    assert m.predict(X.iloc[300:]).iloc[0] == pytest.approx(y.iloc[240:300].mean())


def test_seasonal_mean_exact(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = frame
    tr = X.index < pd.Timestamp("2019-06-01", tz="UTC")
    m = make_model("seasonal_mean", horizon=1).fit(X[tr], y[tr])
    feb = X.index[tr].tz_convert("Europe/Amsterdam").month == 2
    assert m.month_means_[2] == pytest.approx(y[tr][feb].mean())
    later = X[X.index.tz_convert("Europe/Amsterdam").month == 9]
    assert m.predict(later).iloc[0] == pytest.approx(y[tr].mean())  # unseen month -> overall


def test_make_target_lags_known_at_origin() -> None:
    idx = origin_index(10)
    y = pd.Series(np.arange(10.0), index=idx)
    lags = make_target_lags(y, horizon=5, p=2)
    # at row 7 the latest known label is the one from origin 7-5 = 2
    assert lags["ar__lag1"].iloc[7] == 2.0 and lags["ar__lag2"].iloc[7] == 1.0
    assert lags["ar__lag1"].iloc[:5].isna().all()


def test_ar_selects_order_and_matches_ols() -> None:
    rng = np.random.default_rng(11)
    n = 800
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.3 * y[t - 2] + e[t]
    idx = origin_index(n)
    ys = pd.Series(y, index=idx)
    X = make_target_lags(ys, horizon=1, p=5)
    keep = X.notna().all(axis=1)
    m = make_model("ar", horizon=1, lag_columns=list(X.columns), max_p=5).fit(X[keep], ys[keep])
    assert m.p_ == min(m.aic_, key=m.aic_.get)
    assert m.p_ >= 2
    use = list(X.columns[: m.p_])
    b = _lstsq(X.loc[keep, use].to_numpy(), ys[keep].to_numpy())
    np.testing.assert_allclose(m.coef_.to_numpy(), b, atol=1e-10)


def test_storage_deviation_ols_exact(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = frame
    m = make_model("storage_deviation_ols", column="storage__dev", horizon=5).fit(X, y)
    b = _lstsq(X["storage__dev"].to_numpy(), y.to_numpy())
    np.testing.assert_allclose(m.coef_.to_numpy(), b, atol=1e-10)
    np.testing.assert_allclose(m.predict(X).to_numpy(), b[0] + b[1] * X["storage__dev"].to_numpy())


def test_vol_benchmarks(frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = frame
    yv = 0.1 + 0.2 * X["price__rv1"] + 0.3 * X["price__rv5"] + 0.4 * X["price__rv22"]
    tv = make_model("trailing_vol", horizon=1, column="price__rv5").fit(X, yv)
    np.testing.assert_allclose(tv.predict(X).to_numpy(), X["price__rv5"].to_numpy())
    ew = make_model("ewma_vol", horizon=1, column="price__rv5", scale="ols").fit(X, yv)
    xv = X["price__rv5"].to_numpy()
    assert ew.factor_ == pytest.approx(float(xv @ yv.to_numpy() / (xv @ xv)))
    har = make_model("har", horizon=1, rv_columns=["price__rv1", "price__rv5", "price__rv22"]).fit(X, yv)
    np.testing.assert_allclose(har.coef_.to_numpy(), [0.1, 0.2, 0.3, 0.4], atol=1e-8)


def test_base_rate() -> None:
    idx = origin_index(10)
    X = pd.DataFrame({"a": np.zeros(10)}, index=idx)
    y = pd.Series([0, 1, 1, 0, 1, 0, 0, 0, 1, 0], index=idx)
    m = make_model("base_rate", horizon=1).fit(X, y)
    assert m.predict_proba(X).iloc[0] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="binary"):
        make_model("base_rate", horizon=1).fit(X, y * 2.5)
