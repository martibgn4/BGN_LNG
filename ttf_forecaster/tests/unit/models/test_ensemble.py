"""Level 6 combiners: leakage guards and weight constraints. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import origin_index
from ttf_forecaster.models.interface import LeakageGuardError, make_model


@pytest.fixture
def oof() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    n = 300
    idx = origin_index(n)
    y = pd.Series(rng.normal(size=n), index=idx)
    P = pd.DataFrame({"good": y + rng.normal(scale=0.3, size=n),
                      "bad": rng.normal(size=n),
                      "zero": np.zeros(n)}, index=idx)
    return P, y


@pytest.mark.parametrize("name", ["constrained_stacking", "performance_weighted"])
def test_weights_simplex(name: str, oof: tuple[pd.DataFrame, pd.Series]) -> None:
    P, y = oof
    m = make_model(name, horizon=1).fit(P.iloc[:200], y.iloc[:200], train_end=P.index[199])
    w = m.weights_
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert w.idxmax() == "good"
    np.testing.assert_allclose(m.predict(P.iloc[200:]).to_numpy(),
                               P.iloc[200:].to_numpy() @ w.to_numpy())


def test_guards(oof: tuple[pd.DataFrame, pd.Series]) -> None:
    P, y = oof
    te = P.index[199]
    m = make_model("constrained_stacking", horizon=1)
    with pytest.raises(LeakageGuardError, match="requires train_end"):
        m.fit(P.iloc[:200], y.iloc[:200])
    with pytest.raises(LeakageGuardError, match="after train_end"):
        m.fit(P, y, train_end=te)
    lkt = pd.Series(P.index + pd.Timedelta(days=7), index=P.index)
    with pytest.raises(LeakageGuardError, match="label_known_time"):
        m.fit(P.iloc[:200], y.iloc[:200], train_end=te, label_known_time=lkt)
    pte = pd.DataFrame({c: P.index - pd.Timedelta(days=30) for c in P.columns}, index=P.index)
    pte.iloc[10, 0] = P.index[10]  # this prediction was produced by a fold not ending before it
    with pytest.raises(LeakageGuardError, match="out-of-fold"):
        m.fit(P.iloc[:200], y.iloc[:200], train_end=te, prediction_train_end=pte)
    pte.iloc[10, 0] = P.index[10] - pd.Timedelta(days=1)
    m.fit(P.iloc[:200], y.iloc[:200], train_end=te, prediction_train_end=pte)
    with pytest.raises(LeakageGuardError, match="in-sample"):
        m.predict(P.iloc[150:250])
    with pytest.raises(LeakageGuardError, match="tz-aware"):
        make_model("constrained_stacking", horizon=1).fit(P.iloc[:200], y.iloc[:200],
                                               train_end=pd.Timestamp("2019-10-01"))
