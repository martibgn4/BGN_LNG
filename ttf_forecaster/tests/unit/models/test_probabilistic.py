"""Level 4 probabilistic models. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.models.conftest import linear_world
from ttf_forecaster.models.interface import make_model
from ttf_forecaster.models.probabilistic import conformal_quantile

LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


@pytest.mark.parametrize("name", ["linear_quantile", "lightgbm_quantile"])
def test_quantiles_monotone(name: str) -> None:
    X, y = linear_world(n=500)
    m = make_model(name, horizon=1).fit(X.iloc[:400], y.iloc[:400])
    q = m.predict_quantiles(X.iloc[400:], LEVELS)
    assert list(q.columns) == [f"q{v:g}" for v in LEVELS]
    assert (np.diff(q.to_numpy(), axis=1) >= 0).all()
    np.testing.assert_allclose(m.predict(X.iloc[400:]).to_numpy(), q["q0.5"].to_numpy())
    with pytest.raises(ValueError, match="not fitted"):
        m.predict_quantiles(X.iloc[400:], [0.33])


def test_lightgbm_quantile_deterministic() -> None:
    X, y = linear_world(n=400)
    a = make_model("lightgbm_quantile", horizon=1).fit(X, y).predict_quantiles(X, [0.1, 0.9])
    b = make_model("lightgbm_quantile", horizon=1).fit(X, y).predict_quantiles(X, [0.1, 0.9])
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_conformal_calibrates_on_training_tail_only() -> None:
    X, y = linear_world(n=1200, noise=1.0, nan_frac=0.0)
    m = make_model("conformal", horizon=5).fit(X.iloc[:800], y.iloc[:800])
    assert len(m.residuals_) == 200  # last 25 % of the 800 training rows
    assert m.base_.train_end_ < X.index[600]  # base fitted strictly before the calibration tail
    q = m.predict_quantiles(X.iloc[800:], [0.1, 0.9])
    yt = y.iloc[800:]
    cover = ((yt >= q["q0.1"]) & (yt <= q["q0.9"])).mean()
    assert 0.7 < cover < 0.9  # iid synthetic: nominal 80 %; a code-path check, not a result


def test_conformal_quantile_finite_sample() -> None:
    r = np.arange(1.0, 10.0)  # n = 9
    assert conformal_quantile(r, 0.9) == 9.0  # ceil(10*0.9) = 9th order statistic
    assert conformal_quantile(r, 0.1) == 1.0


def test_aci_update_moves_alpha() -> None:
    X, y = linear_world(n=600)
    m = make_model("conformal", horizon=1, adaptive=True, aci_gamma=0.05).fit(X, y)
    iv = m.predict_interval(X.iloc[-1:])
    a0 = m.alpha_t_
    m.update(1e9, float(iv["lower"].iloc[0]), float(iv["upper"].iloc[0]))  # miss -> widen
    assert m.alpha_t_ < a0
    a1 = m.alpha_t_
    m.update(float(iv["lower"].iloc[0]), float(iv["lower"].iloc[0]), float(iv["upper"].iloc[0]))
    assert m.alpha_t_ > a1
    with pytest.raises(RuntimeError):
        make_model("conformal", horizon=1).fit(X, y).update(0.0, -1.0, 1.0)


@pytest.mark.parametrize("name", ["threshold_logistic", "lightgbm_event"])
@pytest.mark.parametrize("calibration", ["sigmoid", "isotonic", "none"])
def test_event_classifiers(name: str, calibration: str) -> None:
    X, y = linear_world(n=700)
    m = make_model(name, threshold=0.0, event="above", calibration=calibration, horizon=5)
    m.fit(X.iloc[:600], y.iloc[:600])
    p = m.predict_proba(X.iloc[600:])
    assert ((p >= 0) & (p <= 1)).all()
    assert m.calibration_used_ == calibration
    p2 = make_model(name, threshold=0.0, calibration=calibration, horizon=5).fit(
        X.iloc[:600], y.iloc[:600]).predict_proba(X.iloc[600:])
    np.testing.assert_array_equal(p.to_numpy(), p2.to_numpy())


def test_event_label_construction() -> None:
    X, y = linear_world(n=50)
    m = make_model("threshold_logistic", horizon=1, threshold=1.0, event="abs_above")
    np.testing.assert_array_equal(m.to_label(y).to_numpy(), (y.abs() > 1.0).astype(float))
