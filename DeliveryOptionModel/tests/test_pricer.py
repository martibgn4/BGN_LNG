"""The engine against results that are known exactly, before any market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from DeliveryOptionModel.src.deliveryoption import calibration, pricer
from DeliveryOptionModel.src.deliveryoption.market_data import contract_ticker
from DeliveryOptionModel.src.deliveryoption.pricer import BasketModel, SwitchPayoff

N_PATHS = 400_000
SEED = 11


def two_asset(fa=10.0, fb=10.5, va=0.3, vb=0.4, rho=0.6, t=1.0):
    return BasketModel(names=("A", "B"), fwd=np.array([fa, fb]), vol=np.array([va, vb]),
                       horizon=np.array([t, t]), corr=np.array([[1, rho], [rho, 1.0]]))


def margrabe(fa, fb, va, vb, rho, t):
    s = np.sqrt(va ** 2 + vb ** 2 - 2 * rho * va * vb) * np.sqrt(t)
    d1 = np.log(fb / fa) / s + 0.5 * s
    return fb * norm.cdf(d1) - fa * norm.cdf(d1 - s)


def hh_ttf(corr_ttf=0.95, corr_hh=0.93, x_hub=0.25, t=35 / 365, can_cancel=False, k=21.5):
    """Today's market, roughly: TTF Nov/Dec 72.9/72.7, HH 3.18/3.51."""
    c = np.array([[1, corr_ttf, x_hub, x_hub],
                  [corr_ttf, 1, x_hub, x_hub],
                  [x_hub, x_hub, 1, corr_hh],
                  [x_hub, x_hub, corr_hh, 1.0]])
    m = BasketModel(names=("TTF_M", "TTF_M1", "HH_M", "HH_M1"),
                    fwd=np.array([72.94, 72.69, 3.177, 3.511]),
                    vol=np.array([0.89, 0.85, 0.51, 0.50]),
                    horizon=np.full(4, t), corr=pricer.nearest_correlation(c))
    usd = 1.1414 / 3.412141633
    p = SwitchPayoff(w0=np.array([-usd, 0, 1, 0]), w1=np.array([0, -usd, 0, 1]),
                     k0=k, k1=k, df0=0.9950, df1=0.9917, can_cancel=can_cancel)
    return m, p


def test_margrabe_is_exact_for_single_asset_legs():
    fa, fb, va, vb, rho, t = 10.0, 10.5, 0.3, 0.4, 0.6, 1.0
    m = two_asset(fa, fb, va, vb, rho, t)
    p = SwitchPayoff(w0=np.array([1.0, 0]), w1=np.array([0, 1.0]), k0=0, k1=0, df0=1, df1=1)
    exact = fa + margrabe(fa, fb, va, vb, rho, t)

    assert pricer.value_kirk_basket(m, p)["value"] == pytest.approx(exact, abs=1e-10)
    res_mc = pricer.value_mc(m, p, N_PATHS, SEED)
    assert abs(res_mc["value_mc"] - exact) < 3 * res_mc["stderr_mc"]


def test_zero_vol_collapses_to_intrinsic():
    m, p = hh_ttf()
    m = m.bumped(vol=np.zeros(4))
    intrinsic = p.intrinsic(m.fwd)
    assert pricer.value_mc(m, p, 10_000, SEED)["value_mc"] == pytest.approx(intrinsic, abs=1e-12)
    assert pricer.value_kirk_basket(m, p)["value"] == pytest.approx(intrinsic, abs=1e-12)
    assert pricer.value_bachelier(m, p)["value"] == pytest.approx(intrinsic, abs=1e-12)


def test_closed_forms_track_mc_on_the_real_shape():
    m, p = hh_ttf()
    res_mc = pricer.value_mc(m, p, N_PATHS, SEED)
    intrinsic = p.intrinsic(m.fwd)
    ext_mc = res_mc["value_mc"] - intrinsic
    ext_kirk = pricer.value_kirk_basket(m, p)["value"] - intrinsic
    ext_bach = pricer.value_bachelier(m, p)["value"] - intrinsic
    # Compared on the extrinsic, where an error actually shows.
    assert abs(ext_kirk - ext_mc) < max(4 * res_mc["stderr_mc"], 0.02 * ext_mc)
    assert abs(ext_bach - ext_mc) < 0.10 * ext_mc


def test_common_strike_cancels_from_the_option():
    """Must-deliver: moving a strike common to both months shifts only intrinsic."""
    m, p_lo = hh_ttf(k=15.0)
    _, p_hi = hh_ttf(k=30.0)
    same_df = dict(df0=1.0, df1=1.0)
    p_lo = SwitchPayoff(**{**p_lo.__dict__, **same_df})
    p_hi = SwitchPayoff(**{**p_hi.__dict__, **same_df})
    ext_mc = [pricer.value_mc(m, p, N_PATHS, SEED)["value_mc"] - p.intrinsic(m.fwd) for p in (p_lo, p_hi)]
    assert ext_mc[0] == pytest.approx(ext_mc[1], abs=1e-10)


def test_cancel_floor_adds_value_and_vanishes_when_deep_in_the_money():
    m, p = hh_ttf()
    _, p_c = hh_ttf(can_cancel=True)
    v_mc, v_c_mc = (pricer.value_mc(m, q, N_PATHS, SEED)["value_mc"] for q in (p, p_c))
    assert v_c_mc >= v_mc
    _, deep = hh_ttf(k=200.0)
    _, deep_c = hh_ttf(k=200.0, can_cancel=True)
    assert pricer.value_mc(m, deep_c, N_PATHS, SEED)["prob_cancel_mc"] == 0.0
    assert (pricer.value_mc(m, deep_c, N_PATHS, SEED)["value_mc"]
            == pytest.approx(pricer.value_mc(m, deep, N_PATHS, SEED)["value_mc"]))


def test_pathwise_delta_matches_bump():
    m, p = hh_ttf()
    _, z = m.simulate_mc(N_PATHS, SEED)
    delta_mc = pricer.value_mc(m, p, N_PATHS, SEED, z=z)["delta_mc"]
    for i in range(4):
        h = 1e-4 * m.fwd[i]
        up, dn = m.fwd.copy(), m.fwd.copy()
        up[i] += h
        dn[i] -= h
        fd_mc = (pricer.value_mc(m.bumped(fwd=up), p, N_PATHS, SEED, z=z)["value_mc"]
              - pricer.value_mc(m.bumped(fwd=dn), p, N_PATHS, SEED, z=z)["value_mc"]) / (2 * h)
        assert delta_mc[i] == pytest.approx(fd_mc, rel=1e-3, abs=1e-6)


def test_simulated_covariance_respects_each_legs_horizon():
    """A leg that fixes before t_d stops diffusing; cross terms use min(h_i, h_j)."""
    m = BasketModel(names=("A", "B", "C"), fwd=np.ones(3), vol=np.array([0.5, 0.4, 0.3]),
                    horizon=np.array([0.5, 1.0, 0.0]),
                    corr=np.array([[1, 0.7, 0.2], [0.7, 1, 0.2], [0.2, 0.2, 1.0]]))
    f_mc, _ = m.simulate_mc(N_PATHS, SEED)
    assert np.all(f_mc[:, 2] == 1.0)
    sample_mc = np.cov(np.log(f_mc[:, :2]).T)
    assert sample_mc == pytest.approx(m.cov[:2, :2], rel=0.02)
    assert m.cov[0, 1] == pytest.approx(0.7 * 0.5 * 0.4 * 0.5)


def test_mc_is_a_martingale():
    m, _ = hh_ttf()
    f_mc, _ = m.simulate_mc(N_PATHS, SEED)
    assert f_mc.mean(axis=0) == pytest.approx(m.fwd, rel=5e-3)


# ---------------------------------------------------------------------------
# calibration windows
# ---------------------------------------------------------------------------

HUB_SPECS = {"TTF": {"future_root": "TZT"}, "HH": {"future_root": "NG"}}
HUBS = ["TTF", "HH"]


def synthetic_prices(anchors):
    dates = pd.bdate_range("2024-01-01", "2025-06-30")
    rng = np.random.default_rng(0)
    cols, expiry = {}, {}
    for k in anchors:
        for m in (k, k + 1):
            for root in ("TZT", "NG"):
                t = contract_ticker(root, m)
                cols[t] = np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
                expiry[t] = (m - 1).end_time.normalize() - pd.Timedelta(days=3)
    return pd.DataFrame(cols, index=dates), pd.Series(expiry)


def test_returns_stay_inside_their_anchor_window():
    anchors = [pd.Period("2024-06", "M") + i for i in range(6)]
    px, expiry = synthetic_prices(anchors)
    r = calibration.window_returns(px, expiry, HUB_SPECS, HUBS, anchors, 1, (2, 40), 5,
                                   as_of=pd.Timestamp("2025-06-30"))
    for k, block in r.groupby(level="anchor"):
        k = pd.Period(k, "M")
        fix = expiry[contract_ticker("TZT", k)]
        dates = block.index.get_level_values("date")
        assert dates.min() > fix - pd.Timedelta(days=40)
        assert dates.max() <= fix - pd.Timedelta(days=2)
    # Non-overlapping: the first return of a block uses a price inside the window,
    # so no return spans two anchors, and each block has ~(40-2)/7 weekly returns.
    assert r.groupby(level="anchor").size().max() <= 6


def test_incomplete_window_is_skipped():
    anchors = [pd.Period("2025-05", "M"), pd.Period("2025-06", "M")]
    px, expiry = synthetic_prices(anchors)
    r = calibration.window_returns(px, expiry, HUB_SPECS, HUBS, anchors, 1, (0, 40), 5,
                                   as_of=pd.Timestamp("2025-05-10"))  # May fixes 27-Apr, June 28-May
    assert set(r.index.get_level_values("anchor")) == {"2025-05"}


def test_corr_sensitivity_is_scaled_by_the_move_actually_made_near_one():
    """At 0.999 the +0.01 bump is capped at 1.0: the sensitivity must be per
    0.01 of actual move, i.e. match a small symmetric bump's slope."""
    m, p = hh_ttf(corr_ttf=0.995)
    g = {"TTF_M/TTF_M1": [(0, 1)]}
    near = pricer.risk_mc(m, p, N_PATHS, SEED, 0.01, 0.01, g)["corr_sens_mc"]["TTF_M/TTF_M1"]
    small = pricer.risk_mc(m, p, N_PATHS, SEED, 0.01, 0.004, g)["corr_sens_mc"]["TTF_M/TTF_M1"]
    # 0.004 fits inside [0.991, 0.999]; the 0.01 bump is clipped to [0.985, 1.0].
    assert near == pytest.approx(small * 0.01 / 0.004, rel=0.35)
    unclipped = pricer.risk_mc(m.bumped(corr=hh_ttf(corr_ttf=0.95)[0].corr), p, N_PATHS, SEED,
                               0.01, 0.01, g)["corr_sens_mc"]["TTF_M/TTF_M1"]
    assert unclipped < 0
