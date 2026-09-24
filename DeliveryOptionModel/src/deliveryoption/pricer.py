"""Pricing engine for the choice of delivering in month M or M+1.

For delivery month m the margin per MMBtu is linear in the forwards:

    X_m = w_m . F(t_d) + K_m

and the holder, at the decision date t_d, takes the better month on a PV basis:

    V = E[ max(DF_0 X_0, DF_1 X_1) ]            must deliver in one of the two
    V = E[ max(DF_0 X_0, DF_1 X_1, 0) ]         may also cancel (floor)

Without the floor this is DF_0 X_0 plus a call struck at zero on

    D = DF_1 X_1 - DF_0 X_0 = d . F(t_d) + k,   d = DF_1 w_1 - DF_0 w_0,
                                                k = DF_1 K_1 - DF_0 K_0

which, for an HH - TTF margin, is a four-asset basket: +HH_{M+1} - HH_M
- TTF_{M+1} + TTF_M. That is the calendar spread option on the inter-hub
spread. Note that a strike common to both months cancels out of D (up to the
DF difference): it only matters through the cancel floor.

Dynamics: each forward is driftless lognormal up to its own horizon
h_i = min(t_d, fixing_i); a leg that fixes before t_d is frozen from then on.
So Cov(ln F_i, ln F_j) = rho_ij sigma_i sigma_j min(h_i, h_j).

Three prices are produced and are meant to be read together:
  * Monte Carlo -- exact under the model, the reference. Every name that
    holds a Monte Carlo number ends in _mc (value_mc, risk_mc, simulate_mc).
  * Kirk on moment-matched baskets -- the positive and negative legs of D
    each matched to a lognormal (Fenton-Wilkinson), then Kirk. With k = 0 this
    is Margrabe, exact for the matched baskets.
  * Bachelier on the exact mean and variance of D -- the quick sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BasketModel:
    """Joint lognormal forwards observed at the decision date."""

    names: tuple[str, ...]
    fwd: np.ndarray       # today's forwards, native units
    vol: np.ndarray       # annualised lognormal vols
    horizon: np.ndarray   # years from today to min(t_d, fixing_i)
    corr: np.ndarray      # correlation of log returns

    def __post_init__(self):
        n = len(self.names)
        for arr in (self.fwd, self.vol, self.horizon):
            assert len(arr) == n, "fwd, vol and horizon must match names"
        assert self.corr.shape == (n, n), "corr must be n x n"
        assert np.allclose(self.corr, self.corr.T), "corr must be symmetric"
        assert np.allclose(np.diag(self.corr), 1.0), "corr must have unit diagonal"

    @property
    def cov(self) -> np.ndarray:
        """Total covariance of log forwards to the decision date."""
        s = self.vol[:, None] * self.vol[None, :]
        t = np.minimum(self.horizon[:, None], self.horizon[None, :])
        return self.corr * s * t

    def bumped(self, *, fwd=None, vol=None, corr=None) -> "BasketModel":
        return replace(
            self,
            fwd=self.fwd if fwd is None else fwd,
            vol=self.vol if vol is None else vol,
            corr=self.corr if corr is None else corr,
        )

    @property
    def steps(self) -> np.ndarray:
        """Distinct horizons: the grid the Brownian increments live on."""
        return np.unique(self.horizon[self.horizon > 0])

    def simulate_mc(self, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(terminal forwards, standard normals), antithetic.

        The normals are returned so bumped revaluations can reuse them
        (common random numbers) instead of adding fresh MC noise to a greek.
        """
        z = _normals(n_paths, len(self.steps) * len(self.names), seed)
        return self.terminal_mc(z), z

    def terminal_mc(self, z: np.ndarray) -> np.ndarray:
        """Forwards at the decision date, built interval by interval.

        Each leg diffuses only until its own horizon, which gives exactly
        Cov = rho sigma_i sigma_j min(h_i, h_j). Factorising the correlation
        (not the covariance) with Cholesky keeps the map from normals to paths
        smooth in vol and correlation, so CRN bumps are clean.
        """
        n = len(self.names)
        z = z.reshape(len(z), -1, n)
        chol = _cholesky(self.corr)
        log_f = np.zeros((len(z), n))
        prev = 0.0
        for s, t in enumerate(self.steps):
            dt = t - prev
            active = self.horizon >= t
            sd = np.where(active, self.vol * np.sqrt(dt), 0.0)
            log_f += (z[:, s, :] @ chol.T) * sd - 0.5 * sd ** 2
            prev = t
        return self.fwd * np.exp(log_f)


@dataclass(frozen=True)
class SwitchPayoff:
    """Per-MMBtu margins for the two delivery months, and their discounting."""

    w0: np.ndarray        # weights on the forwards for delivery in M
    w1: np.ndarray        # ... and in M+1
    k0: float             # constant term for M (the strike), USD/MMBtu
    k1: float
    df0: float            # discount factor to the M payment date
    df1: float
    can_cancel: bool = False

    @property
    def d(self) -> np.ndarray:
        return self.df1 * self.w1 - self.df0 * self.w0

    @property
    def k(self) -> float:
        return self.df1 * self.k1 - self.df0 * self.k0

    def pv_legs(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.df0 * (f @ self.w0 + self.k0),
                self.df1 * (f @ self.w1 + self.k1))

    def intrinsic(self, fwd: np.ndarray) -> float:
        x0, x1 = self.pv_legs(fwd[None, :])
        best = max(x0[0], x1[0])
        return max(best, 0.0) if self.can_cancel else best


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def value_mc(model: BasketModel, payoff: SwitchPayoff, n_paths: int, seed: int,
             z: np.ndarray | None = None) -> dict:
    """Reference price, with pathwise deltas and the exercise split."""
    if z is None:
        f_mc, z = model.simulate_mc(n_paths, seed)
    else:
        f_mc = model.terminal_mc(z)
    x0_mc, x1_mc = payoff.pv_legs(f_mc)
    take1_mc = x1_mc > x0_mc
    best_mc = np.where(take1_mc, x1_mc, x0_mc)
    cancel_mc = np.zeros_like(take1_mc)
    if payoff.can_cancel:
        cancel_mc = best_mc < 0.0
        best_mc = np.where(cancel_mc, 0.0, best_mc)

    # The option alone: the right to switch away from delivering in M (and,
    # with a floor, to walk away). best_mc >= x0_mc on every path, so it is >= 0
    # path by path, not just on average.
    switch_mc = best_mc - x0_mc
    # The deal is the M margin plus the option. The M margin is linear in the
    # forwards, so its value is known exactly; adding it analytically instead
    # of averaging it over paths is a control variate - the X_M noise cancels,
    # and deal value = PV(X_M) + option value holds to the last digit.
    x0_exact = payoff.df0 * (model.fwd @ payoff.w0 + payoff.k0)

    # Antithetic pairs are not independent: the standard error has to be taken
    # over pair means, or it is understated. Deal and option share it.
    half = len(best_mc) // 2
    pair_means_mc = 0.5 * (switch_mc[:half] + switch_mc[half:])
    stderr_mc = pair_means_mc.std(ddof=1) / np.sqrt(half)

    # Pathwise delta: the payoff is piecewise linear in F(t_d) and
    # dF_i(t_d)/dF_i(0) = F_i(t_d)/F_i(0), so no bump noise.
    grad_mc = np.where(take1_mc[:, None], payoff.df1 * payoff.w1, payoff.df0 * payoff.w0)
    grad_mc = np.where(cancel_mc[:, None], 0.0, grad_mc)
    delta_mc = (grad_mc * f_mc / model.fwd).mean(axis=0)

    return {
        "value_mc": float(x0_exact + switch_mc.mean()),
        "stderr_mc": float(stderr_mc),
        "option_value_mc": float(switch_mc.mean()),
        "option_stderr_mc": float(stderr_mc),
        "prob_m1_mc": float((take1_mc & ~cancel_mc).mean()),
        "prob_cancel_mc": float(cancel_mc.mean()),
        "delta_mc": delta_mc,
    }


# ---------------------------------------------------------------------------
# Closed-form approximations (no cancel floor)
# ---------------------------------------------------------------------------

def moments_of_linear(model: BasketModel, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(E[aF], E[bF], E[aF * bF]) for F lognormal at the decision date."""
    ef = model.fwd
    cross = np.outer(a * ef, b * ef) * np.exp(model.cov)
    return float(a @ ef), float(b @ ef), float(cross.sum())


def value_bachelier(model: BasketModel, payoff: SwitchPayoff) -> dict:
    """Normal approximation of D with its exact mean and variance."""
    _require_no_floor(payoff)
    d = payoff.d
    mean_d, _, second = moments_of_linear(model, d, d)
    mean_d += payoff.k
    var_d = second - (mean_d - payoff.k) ** 2
    sd = np.sqrt(max(var_d, 0.0))
    call = _bachelier_call(mean_d, sd)
    base = payoff.df0 * (model.fwd @ payoff.w0 + payoff.k0)
    return {"value": base + call, "spread_mean": mean_d, "spread_sd": sd}


def value_kirk_basket(model: BasketModel, payoff: SwitchPayoff) -> dict:
    """Kirk on the positive and negative legs of D, each moment-matched.

    E[(P - N + k)^+] with P, N lognormal baskets. Kirk needs N - k > 0; if a
    large constant makes that fail the method returns NaN rather than a number
    from outside its domain.
    """
    _require_no_floor(payoff)
    d = payoff.d
    p_w = np.where(d > 0, d, 0.0)
    n_w = np.where(d < 0, -d, 0.0)

    mp, _, mpp = moments_of_linear(model, p_w, p_w)
    mn, _, mnn = moments_of_linear(model, n_w, n_w)
    _, _, mpn = moments_of_linear(model, p_w, n_w)
    vp = max(np.log(mpp / mp ** 2), 0.0)   # rounding can give -1e-16 at zero vol
    vn = max(np.log(mnn / mn ** 2), 0.0)
    rho = np.log(mpn / (mp * mn)) / np.sqrt(vp * vn) if vp * vn > 0 else 0.0

    strike = -payoff.k          # D^+ = (P - (N + strike))^+
    base = payoff.df0 * (model.fwd @ payoff.w0 + payoff.k0)
    out = {"basket_pos": mp, "basket_neg": mn, "basket_corr": rho,
           "basket_vol_pos": np.sqrt(vp), "basket_vol_neg": np.sqrt(vn)}
    if mn + strike <= 0:
        return {**out, "value": np.nan}
    b = mn / (mn + strike)
    total_var = vp + b * b * vn - 2 * rho * b * np.sqrt(vp * vn)
    call = _black_call(mp, mn + strike, np.sqrt(max(total_var, 0.0)))
    return {**out, "value": base + call}


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

def risk_mc(model: BasketModel, payoff: SwitchPayoff, n_paths: int, seed: int,
            bump_vol: float, bump_corr: float,
            corr_groups: dict[str, list[tuple[int, int]]]) -> dict:
    """Value, pathwise deltas, and CRN bump vegas / correlation sensitivities.

    Vega is per absolute vol point of `bump_vol` (0.01 = one vol point). A
    correlation sensitivity is per `bump_corr` applied to every pair of a
    named group at once: a group can be one pair (TTF_M/TTF_M1) or all four
    cross-hub pairs, which is the only cross-hub move that keeps the calendar
    spreads' own correlation - what the option actually depends on - intact.
    Both are central differences on the same normals.
    """
    _, z = model.simulate_mc(n_paths, seed)
    base_mc = value_mc(model, payoff, n_paths, seed, z=z)

    vega_mc = np.zeros(len(model.names))
    for i in range(len(model.names)):
        up, dn = model.vol.copy(), model.vol.copy()
        up[i] += bump_vol
        dn[i] = max(dn[i] - bump_vol, 0.0)
        v_up_mc = value_mc(model.bumped(vol=up), payoff, n_paths, seed, z=z)["value_mc"]
        v_dn_mc = value_mc(model.bumped(vol=dn), payoff, n_paths, seed, z=z)["value_mc"]
        vega_mc[i] = (v_up_mc - v_dn_mc) * bump_vol / (up[i] - dn[i])

    corr_sens_mc = {}
    for label, pairs in corr_groups.items():
        vals_mc, moved = [], []
        for sign in (1.0, -1.0):
            c = model.corr.copy()
            for i, j in pairs:
                c[i, j] = c[j, i] = np.clip(c[i, j] + sign * bump_corr, -1.0, 1.0)
            moved.append(np.mean([c[i, j] for i, j in pairs]))
            vals_mc.append(value_mc(model.bumped(corr=nearest_correlation(c)),
                                    payoff, n_paths, seed, z=z)["value_mc"])
        # Scale by the move actually made: a correlation at 0.999 can only go
        # up 0.001, and dividing by the nominal 2 x bump would understate it.
        span = moved[0] - moved[1]
        corr_sens_mc[label] = (vals_mc[0] - vals_mc[1]) * bump_corr / span if span > 0 else 0.0

    return {**base_mc, "vega_mc": vega_mc, "corr_sens_mc": corr_sens_mc, "bump_corr": bump_corr}


def nearest_correlation(c: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues and renormalise to a unit diagonal.

    A bumped or separately-estimated correlation matrix need not be PSD; this
    is the minimal repair, not Higham's full projection, and it is reported
    when it moves anything.
    """
    w, v = np.linalg.eigh((c + c.T) / 2.0)
    fixed = v @ np.diag(np.maximum(w, floor)) @ v.T
    s = np.sqrt(np.diag(fixed))
    return fixed / np.outer(s, s)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normals(n_paths: int, n: int, seed: int) -> np.ndarray:
    half = np.random.default_rng(seed).standard_normal((n_paths // 2, n))
    return np.vstack([half, -half])


def _cholesky(corr: np.ndarray) -> np.ndarray:
    """Cholesky of a correlation matrix, repairing it first if not PD."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_correlation(corr, floor=1e-8))


def _bachelier_call(mean: float, sd: float) -> float:
    if sd <= 0:
        return max(mean, 0.0)
    x = mean / sd
    return mean * norm.cdf(x) + sd * norm.pdf(x)


def _black_call(f: float, k: float, total_sd: float) -> float:
    if total_sd <= 0:
        return max(f - k, 0.0)
    d1 = np.log(f / k) / total_sd + 0.5 * total_sd
    return f * norm.cdf(d1) - k * norm.cdf(d1 - total_sd)


def _require_no_floor(payoff: SwitchPayoff):
    if payoff.can_cancel:
        raise ValueError("closed forms price max(X0, X1); with a cancel floor use MC")
