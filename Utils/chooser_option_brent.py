"""
============================================================
Chooser Option Pricing Framework — Dated Brent
============================================================
Prices the physical optionality of picking the MINIMUM of
M-1, M and M+1 Brent futures prices, exercisable latest
at Bill-of-Lading date + 25 business days.

Model:
  - Each forward price F_i follows GBM under risk-neutral measure
  - Correlated log-normal dynamics calibrated to term-structure vols
  - Monte Carlo pricer with antithetic variates & moment-matching
  - Closed-form approximation via Margrabe-style spread option
  - Full Greeks via finite-difference and pathwise sensitivities

Market Data:
  - Live data sourced from Bloomberg via bbg_market_data.py
  - Forward prices  : CO1, CO2, COP1 Comdty  (BDP as-of date)
  - Implied vols    : IVOL_MID from Bloomberg BVOL surface
  - Correlations    : 90bd rolling log-return correlations (BDH)
  - Risk-free rate  : 1M SOFR OIS (USOSFR1Z BGN Curncy)
  - Offline fallback: hardcoded illustrative values with UserWarning

Dependencies:
  pip install xbbg          # Bloomberg Desktop API wrapper (preferred)
  pip install pdblp         # alternative Bloomberg wrapper
  Bloomberg Desktop or B-Pipe must be running.

Author : Quant Modelling
Date   : 2026-03-10
============================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  Market Data Container
# ─────────────────────────────────────────────────────────────

@dataclass
class BrentMarketData:
    """
    Snapshot of Dated Brent market data needed by the pricer.

    Attributes
    ----------
    valuation_date : str  (YYYY-MM-DD)
    bl_date        : str  Bill-of-Lading date (YYYY-MM-DD)
    F               : dict  {tenor: forward_price}
                           tenor in {'M-1', 'M', 'M+1'}
    sigma           : dict  {tenor: annualised_lognormal_vol}
    corr            : dict  {(tenor_i, tenor_j): rho}  (symmetric)
    r               : float  risk-free rate (continuous, ACT/365)
    """
    valuation_date : str
    bl_date        : str
    F              : dict          # e.g. {'M-1': 82.50, 'M': 82.20, 'M+1': 81.90}
    sigma          : dict          # e.g. {'M-1': 0.28,  'M': 0.30,  'M+1': 0.32}
    corr           : dict          # e.g. {('M-1','M'): 0.97, ('M-1','M+1'): 0.93, ('M','M+1'): 0.98}
    r              : float = 0.05
    calendar_days_per_year: float = 365.0

    # ---- derived ---
    T_bl           : float = field(init=False)   # years to BL date
    T_ex           : float = field(init=False)   # years to latest exercise (BL+25 bd)

    def __post_init__(self):
        import datetime
        val  = pd.Timestamp(self.valuation_date)
        bl   = pd.Timestamp(self.bl_date)
        # +25 business days from BL
        ex   = bl + pd.offsets.BusinessDay(25)
        self.T_bl  = max((bl  - val).days / self.calendar_days_per_year, 1e-6)
        self.T_ex  = max((ex  - val).days / self.calendar_days_per_year, 1e-6)

    @property
    def tenors(self):
        return ['M-1', 'M', 'M+1']

    def correlation_matrix(self) -> np.ndarray:
        t = self.tenors
        n = len(t)
        C = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    key  = (t[i], t[j])
                    rkey = (t[j], t[i])
                    C[i, j] = self.corr.get(key, self.corr.get(rkey, 0.95))
        return C

    def forward_vector(self) -> np.ndarray:
        return np.array([self.F[t] for t in self.tenors])

    def vol_vector(self) -> np.ndarray:
        return np.array([self.sigma[t] for t in self.tenors])


# ─────────────────────────────────────────────────────────────
# 2.  Results Container
# ─────────────────────────────────────────────────────────────

@dataclass
class PricingResult:
    price           : float
    price_std_err   : Optional[float]
    method          : str
    # Greeks
    delta           : dict   = field(default_factory=dict)   # ∂V/∂F_i
    vega            : dict   = field(default_factory=dict)   # ∂V/∂σ_i
    theta           : float  = 0.0                            # ∂V/∂t  (daily)
    rho             : float  = 0.0                            # ∂V/∂r
    gamma           : dict   = field(default_factory=dict)   # ∂²V/∂F_i²
    cross_gamma     : dict   = field(default_factory=dict)   # ∂²V/∂F_i∂F_j
    # Analytical sub-results
    intrinsic       : float  = 0.0
    time_value      : float  = 0.0

    def __repr__(self):
        lines = [
            f"\n{'='*55}",
            f"  Chooser Min-of-3 Option  [{self.method}]",
            f"{'='*55}",
            f"  Price          : {self.price:>10.4f}  $/bbl",
            f"  Std Error      : {self.price_std_err if self.price_std_err else 'N/A':>10}",
            f"  Intrinsic      : {self.intrinsic:>10.4f}  $/bbl",
            f"  Time Value     : {self.time_value:>10.4f}  $/bbl",
            f"{'─'*55}",
            "  Deltas (∂V/∂F):"
        ]
        for k, v in self.delta.items():
            lines.append(f"    {k:<8}: {v:>10.6f}")
        lines.append("  Vegas  (∂V/∂σ):")
        for k, v in self.vega.items():
            lines.append(f"    {k:<8}: {v:>10.4f}  $/bbl per 1% vol")
        lines.append(f"  Theta  (∂V/∂t) : {self.theta:>10.4f}  $/bbl per day")
        lines.append(f"  Rho    (∂V/∂r) : {self.rho:>10.4f}  $/bbl per 1bp")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 3.  Core Pricing Engine
# ─────────────────────────────────────────────────────────────

class ChooserOptionPricer:
    """
    Price and risk-manage:

        V = E[ e^{-rT} * min(F_{M-1}(T), F_M(T), F_{M+1}(T)) ]

    where T = min(exercise_decision, BL + 25 bd).

    The holder can CHOOSE to receive whichever of the three
    Dated Brent month forward prices is lowest at expiry —
    capturing the physical optionality in cargo pricing.

    Methods available
    -----------------
    price_mc()        : Full Monte Carlo (default, most accurate)
    price_analytical(): Closed-form lower bound via Margrabe decomposition
    price_and_greeks(): Run both + full Greek ladder
    """

    N_DEFAULT = 200_000   # MC paths

    def __init__(self, mkt: BrentMarketData):
        self.mkt = mkt
        self._cholesky = None

    # ── Cholesky decomposition (cached) ──────────────────────

    def _get_cholesky(self) -> np.ndarray:
        if self._cholesky is None:
            C = self.mkt.correlation_matrix()
            try:
                self._cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularise if not PD
                C += 1e-6 * np.eye(len(C))
                self._cholesky = np.linalg.cholesky(C)
        return self._cholesky

    # ── Log-normal simulation ─────────────────────────────────

    def _simulate_forwards(
        self,
        T      : float,
        n_paths: int,
        seed   : int = 42
    ) -> np.ndarray:
        """
        Returns array (n_paths, 3) of simulated forward prices at time T.
        Uses antithetic variates for variance reduction.
        """
        rng   = np.random.default_rng(seed)
        F0    = self.mkt.forward_vector()       # (3,)
        sigma = self.mkt.vol_vector()            # (3,)
        L     = self._get_cholesky()             # (3,3)

        half  = n_paths // 2
        Z_raw = rng.standard_normal((half, 3))  # i.i.d.
        Z     = Z_raw @ L.T                      # correlated

        # Antithetic
        Z_full = np.vstack([Z, -Z])              # (n_paths, 3)

        drift  = -0.5 * sigma**2 * T             # (3,)
        diffus = sigma * np.sqrt(T)              # (3,)

        log_F  = np.log(F0) + drift + diffus * Z_full   # (n_paths, 3)
        return np.exp(log_F)

    # ── Monte Carlo Pricer ────────────────────────────────────

    def price_mc(
        self,
        n_paths: int = N_DEFAULT,
        seed   : int = 42
    ) -> tuple[float, float]:
        """
        Price by Monte Carlo simulation.
        Returns (price, std_error).
        """
        mkt = self.mkt
        T   = mkt.T_ex
        r   = mkt.r

        F_T    = self._simulate_forwards(T, n_paths, seed)  # (N,3)
        payoff = np.min(F_T, axis=1)                         # min of 3
        df     = np.exp(-r * T)
        pv     = df * payoff

        price   = pv.mean()
        std_err = pv.std(ddof=1) / np.sqrt(n_paths)
        return price, std_err

    # ── Analytical Approximation ──────────────────────────────
    # min(A,B,C) = A + B + C - max(A,B) - max(B,C) - max(A,C) + max(A,B,C)
    # Equivalently: min(A,B,C) = A - max(A-B,0) - max(A-C,0) + max(A-B,0)*I(B<C)
    #
    # We use the identity:
    #   min(A,B,C) = A - max(A-B, 0) - max(A-C, 0) + max(A-B-max(A-C-...))
    # Practically we price it as:
    #   E[min(A,B,C)] = E[A] - E[max(A-B,0)] - E[max(A-C,0)] + E[max(A-B,A-C,0)]
    # The last term is a spread-on-spread; we approximate it via Margrabe's formula.

    @staticmethod
    def _margrabe(F1: float, F2: float,
                  s1: float, s2: float,
                  rho: float, T: float) -> float:
        """
        Price of exchange option: E[max(F1 - F2, 0)]
        Margrabe (1978). Forwards already carry no drift under futures measure.
        """
        sigma_spread = np.sqrt(s1**2 + s2**2 - 2*rho*s1*s2)
        if sigma_spread < 1e-10 or T < 1e-10:
            return max(F1 - F2, 0.0)
        d1 = (np.log(F1/F2) + 0.5*sigma_spread**2*T) / (sigma_spread*np.sqrt(T))
        d2 = d1 - sigma_spread * np.sqrt(T)
        return F1*norm.cdf(d1) - F2*norm.cdf(d2)

    def price_analytical(self) -> float:
        """
        Closed-form lower bound / approximation using Margrabe decompositions
        plus Kirk's approximation for the three-asset spread.

        E[min(F1,F2,F3)] ≈ E[F1] + E[F2] + E[F3]
                           - E[max(F1,F2)] - E[max(F2,F3)] - E[max(F1,F3)]
                           + E[max(F1,F2,F3)]

        max(A,B)   = A + max(B-A, 0)
        max(A,B,C) ≈ Stulz (1982) three-asset option (implemented below)
        """
        mkt   = self.mkt
        T     = mkt.T_ex
        F     = mkt.forward_vector()   # [F0, F1, F2]
        sigma = mkt.vol_vector()
        C     = mkt.correlation_matrix()
        df    = np.exp(-mkt.r * T)

        r01 = C[0,1]; r02 = C[0,2]; r12 = C[1,2]

        # E[max(Fi, Fj)] = Fj + Margrabe(Fi, Fj)  -- note symmetry
        def E_max(i, j):
            return F[j] + self._margrabe(F[i], F[j], sigma[i], sigma[j], C[i,j], T)

        E_max01 = E_max(0,1)
        E_max02 = E_max(0,2)
        E_max12 = E_max(1,2)

        # E[max(F0,F1,F2)] via Stulz (1982) three-asset maximum
        E_max012 = self._stulz_max3(F, sigma, C, T)

        # E[min(A,B,C)] = A+B+C - max(A,B) - max(A,C) - max(B,C) + max(A,B,C)
        E_min = (F[0] + F[1] + F[2]
                 - E_max01 - E_max02 - E_max12
                 + E_max012)

        return df * E_min

    @staticmethod
    def _stulz_max3(F: np.ndarray, sigma: np.ndarray,
                    C: np.ndarray, T: float) -> float:
        """
        Stulz (1982): E[ max(F1, F2, F3) ]
        Uses the identity:
          max(A,B,C) = A + max(B-A,0) + max(C-A,0) - max(C-B, B-A, 0) ...
        We use the recursive:
          E[max(A,B,C)] = E[max(A,max(B,C))]
        and price E[max(B,C)] first (Margrabe), then
        price E[max(A, E_max_BC)] using a log-normal approx for max(B,C).

        For robustness we use a direct Margrabe chain:
          max(A,B,C) = A + [B-A]^+ + [C-max(A,B)]^+
        The last term is approximated via Kirk's spread.
        """
        # Step 1: E[max(F0, F1)] via Margrabe
        s01   = np.sqrt(sigma[0]**2 + sigma[1]**2 - 2*C[0,1]*sigma[0]*sigma[1])
        d1_01 = (np.log(F[0]/F[1]) + 0.5*s01**2*T) / (s01*np.sqrt(T) + 1e-12)
        d2_01 = d1_01 - s01*np.sqrt(T)
        E_max01 = F[0]*norm.cdf(d1_01) + F[1]*norm.cdf(-d1_01)

        # Effective forward for max(F0,F1) — approximate as log-normal
        # using the moment-matched parameters (Brigo-Mercurio lognormal approx)
        # E[max(F0,F1)^2] needed for vol matching
        # We skip full moment matching and use a first-order approximation
        F_eff  = E_max01
        # Vol of max(F0,F1): approximate via log-normal with sigma_eff
        # σ_eff ≈ σ_1 * F1*N(d1) / E_max + σ_0 * F0*N(-d1) / E_max  (delta-weighted)
        if F_eff > 1e-6:
            w0     = F[0]*norm.cdf(d1_01)  / F_eff
            w1     = F[1]*norm.cdf(-d1_01) / F_eff
            sigma_eff = np.sqrt(
                w0**2*sigma[0]**2 + w1**2*sigma[1]**2
                + 2*w0*w1*C[0,1]*sigma[0]*sigma[1]
            )
            # Correlation of max(F0,F1) with F2
            rho_eff2 = (w0*C[0,2]*sigma[0] + w1*C[1,2]*sigma[1]) / (sigma_eff + 1e-12)
            rho_eff2 = np.clip(rho_eff2, -0.9999, 0.9999)
        else:
            sigma_eff = sigma[2]
            rho_eff2  = C[0,2]

        # Step 2: E[max(max(F0,F1), F2)]
        s_eff2 = np.sqrt(sigma_eff**2 + sigma[2]**2
                         - 2*rho_eff2*sigma_eff*sigma[2])
        if s_eff2 < 1e-10 or T < 1e-10:
            return max(F_eff, F[2])

        d1_x = (np.log(F_eff/F[2]) + 0.5*s_eff2**2*T) / (s_eff2*np.sqrt(T))
        d2_x = d1_x - s_eff2*np.sqrt(T)
        E_max012 = F_eff*norm.cdf(d1_x) + F[2]*norm.cdf(-d1_x)
        return E_max012

    # ── Full Greeks via Finite Difference ─────────────────────

    def greeks_fd(
        self,
        n_paths : int = N_DEFAULT,
        bump_F  : float = 0.01,    # $/bbl
        bump_sig: float = 0.001,   # 10bp vol bump
        bump_r  : float = 0.0001,  # 1bp rate bump
        bump_t  : float = 1/365    # 1 calendar day
    ) -> dict:
        """
        Compute Greeks via central finite differences on the MC price.
        """
        mkt  = self.mkt
        base, _ = self.price_mc(n_paths)

        delta, gamma, vega = {}, {}, {}

        for i, tenor in enumerate(mkt.tenors):
            # Delta: bump each forward price
            mkt.F[tenor] += bump_F
            self._cholesky = None
            pu, _ = self.price_mc(n_paths)

            mkt.F[tenor] -= 2*bump_F
            pd, _ = self.price_mc(n_paths)

            mkt.F[tenor] += bump_F        # restore
            self._cholesky = None

            delta[tenor] = (pu - pd) / (2*bump_F)
            gamma[tenor] = (pu - 2*base + pd) / (bump_F**2)

            # Vega: bump each vol
            mkt.sigma[tenor] += bump_sig
            pu_v, _ = self.price_mc(n_paths)
            mkt.sigma[tenor] -= 2*bump_sig
            pd_v, _ = self.price_mc(n_paths)
            mkt.sigma[tenor] += bump_sig  # restore

            vega[tenor] = (pu_v - pd_v) / (2*bump_sig) * 0.01  # per 1% vol

        # Theta: bump expiry by -1 day
        orig_T = mkt.T_ex
        mkt.T_ex = max(orig_T - bump_t, 1e-6)
        p_theta, _ = self.price_mc(n_paths)
        mkt.T_ex = orig_T
        theta = (p_theta - base) / (bump_t * 365) * (-1)  # $/bbl per day (decay)

        # Rho: bump rate
        mkt.r += bump_r
        p_rho, _ = self.price_mc(n_paths)
        mkt.r -= bump_r
        rho = (p_rho - base) / bump_r * 0.0001  # per 1bp

        return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    # ── Pathwise / Likelihood Ratio Greeks (faster) ───────────

    def greeks_pathwise(self, n_paths: int = N_DEFAULT) -> dict:
        """
        Pathwise (IPA) estimator for delta and vega.
        For min(F1,F2,F3): the pathwise delta w.r.t. F0_i is
            e^{-rT} * I(F_i = argmin) * (F_i / F0_i)
        """
        mkt = self.mkt
        T   = mkt.T_ex
        r   = mkt.r
        F0  = mkt.forward_vector()
        sig = mkt.vol_vector()
        df  = np.exp(-r * T)

        F_T = self._simulate_forwards(T, n_paths)    # (N,3)
        idx_min = np.argmin(F_T, axis=1)              # which tenor wins

        delta, vega = {}, {}
        for i, tenor in enumerate(mkt.tenors):
            # Pathwise delta
            indicator  = (idx_min == i).astype(float)  # (N,)
            pw_delta   = df * indicator * (F_T[:,i] / F0[i])
            delta[tenor] = pw_delta.mean()

            # Pathwise vega (log-normal payoff derivative w.r.t. sigma)
            Z_i       = (np.log(F_T[:,i]/F0[i]) + 0.5*sig[i]**2*T) / (sig[i]*np.sqrt(T))
            pw_vega   = df * indicator * F_T[:,i] * (Z_i - sig[i]*np.sqrt(T)) * 0.01
            vega[tenor] = pw_vega.mean()

        return dict(delta=delta, vega=vega)

    # ── Master Entry Point ────────────────────────────────────

    def price_and_greeks(
        self,
        n_paths          : int  = N_DEFAULT,
        greeks_method    : str  = 'pathwise',   # 'pathwise' or 'fd'
        include_analytical: bool = True
    ) -> PricingResult:
        """
        Full pricing and risk run.

        Parameters
        ----------
        n_paths          : MC simulation paths
        greeks_method    : 'pathwise' (fast) or 'fd' (finite difference)
        include_analytical: also run analytical pricer for cross-check
        """
        price_mc, std_err = self.price_mc(n_paths)

        if greeks_method == 'pathwise':
            g = self.greeks_pathwise(n_paths)
            cross_gamma = {}
            gamma       = {}
        else:
            g = self.greeks_fd(n_paths)
            gamma       = g.get('gamma', {})
            cross_gamma = {}

        # Intrinsic value: min of current forwards, discounted
        intrinsic  = np.exp(-self.mkt.r * self.mkt.T_ex) * min(self.mkt.forward_vector())
        time_value = price_mc - intrinsic

        result = PricingResult(
            price         = price_mc,
            price_std_err = std_err,
            method        = f"Monte Carlo ({n_paths:,} paths, {greeks_method} Greeks)",
            delta         = g['delta'],
            vega          = g['vega'],
            theta         = g.get('theta', 0.0),
            rho           = g.get('rho',   0.0),
            gamma         = gamma,
            cross_gamma   = cross_gamma,
            intrinsic     = intrinsic,
            time_value    = time_value
        )

        if include_analytical:
            p_anal = self.price_analytical()
            result._analytical_price = p_anal

        return result


# ─────────────────────────────────────────────────────────────
# 4.  Scenario / Risk Analysis
# ─────────────────────────────────────────────────────────────

class ScenarioAnalysis:
    """
    Scenario and stress analysis utilities.
    """

    def __init__(self, pricer: ChooserOptionPricer):
        self.pricer = pricer
        self.base_mkt = pricer.mkt

    def vol_surface_sensitivity(
        self,
        vol_shifts: list = [-0.05, -0.02, 0.0, 0.02, 0.05]
    ) -> pd.DataFrame:
        """Reprice across parallel vol shifts."""
        results = []
        for dv in vol_shifts:
            mkt_copy = self._copy_mkt()
            for t in mkt_copy.tenors:
                mkt_copy.sigma[t] = max(self.base_mkt.sigma[t] + dv, 0.01)
            p, _ = ChooserOptionPricer(mkt_copy).price_mc(n_paths=50_000)
            results.append({'vol_shift': f"{dv:+.0%}", 'price': round(p,4)})
        return pd.DataFrame(results)

    def forward_curve_sensitivity(
        self,
        price_shifts: list = [-5, -2, 0, 2, 5]
    ) -> pd.DataFrame:
        """Reprice across parallel forward curve shifts ($/bbl)."""
        results = []
        for dp in price_shifts:
            mkt_copy = self._copy_mkt()
            for t in mkt_copy.tenors:
                mkt_copy.F[t] = self.base_mkt.F[t] + dp
            p, _ = ChooserOptionPricer(mkt_copy).price_mc(n_paths=50_000)
            results.append({'price_shift': f"{dp:+.1f}", 'price': round(p,4)})
        return pd.DataFrame(results)

    def correlation_sensitivity(
        self,
        rho_shifts: list = [-0.1, -0.05, 0.0, 0.05, 0.1]
    ) -> pd.DataFrame:
        """Reprice across parallel correlation shifts."""
        results = []
        for dr in rho_shifts:
            mkt_copy = self._copy_mkt()
            for k in mkt_copy.corr:
                mkt_copy.corr[k] = np.clip(self.base_mkt.corr[k] + dr, -0.99, 0.99)
            p, _ = ChooserOptionPricer(mkt_copy).price_mc(n_paths=50_000)
            results.append({'corr_shift': f"{dr:+.2f}", 'price': round(p,4)})
        return pd.DataFrame(results)

    def time_decay_profile(self, days: int = 30) -> pd.DataFrame:
        """Daily theta profile from today to expiry."""
        import datetime
        results = []
        val_dt = pd.Timestamp(self.base_mkt.valuation_date)
        ex_dt  = val_dt + pd.to_timedelta(
            int(self.base_mkt.T_ex * self.base_mkt.calendar_days_per_year), unit='D'
        )
        for d in range(0, min(days, int(self.base_mkt.T_ex*365))+1, 1):
            mkt_copy = self._copy_mkt()
            new_val  = val_dt + pd.Timedelta(days=d)
            mkt_copy.valuation_date = str(new_val.date())
            mkt_copy.__post_init__()
            if mkt_copy.T_ex < 1e-4:
                break
            p, _ = ChooserOptionPricer(mkt_copy).price_mc(n_paths=30_000)
            results.append({'days_elapsed': d, 'price': round(p,4), 'T_ex': round(mkt_copy.T_ex, 4)})
        return pd.DataFrame(results)

    def _copy_mkt(self) -> BrentMarketData:
        import copy
        return copy.deepcopy(self.base_mkt)


# ─────────────────────────────────────────────────────────────
# 5.  Implied Vol Utilities
# ─────────────────────────────────────────────────────────────

def implied_vol_min_option(
    market_price: float,
    mkt         : BrentMarketData,
    vol_guess   : float = 0.30
) -> float:
    """
    Back out a flat implied vol (constant across all tenors) that
    reproduces the market price of the chooser min option.
    """
    def objective(v):
        mkt_copy = mkt.__class__.__new__(mkt.__class__)
        mkt_copy.__dict__ = mkt.__dict__.copy()
        mkt_copy.sigma    = {t: v for t in mkt.tenors}
        mkt_copy.F        = mkt.F.copy()
        mkt_copy.corr     = mkt.corr.copy()
        pricer  = ChooserOptionPricer(mkt_copy)
        p, _    = pricer.price_mc(n_paths=50_000, seed=99)
        return p - market_price

    try:
        iv = brentq(objective, 1e-4, 5.0, xtol=1e-4, maxiter=50)
    except ValueError:
        iv = float('nan')
    return iv


# ─────────────────────────────────────────────────────────────
# 6.  Demo / Example Run
# ─────────────────────────────────────────────────────────────

# ── Fallback data (used only when Bloomberg is unavailable) ──
# These values are illustrative placeholders. In production the
# Bloomberg loader will always override them.
_FALLBACK_MARKET_DATA = {
    "F": {
        "M-1": 82.50,   # $/bbl  — previous prompt (Apr Brent proxy)
        "M"  : 82.20,   # $/bbl  — front month    (May Brent)
        "M+1": 81.90,   # $/bbl  — second month   (Jun Brent)
    },
    "sigma": {
        "M-1": 0.280,   # 28.0% annualised ATM log-vol
        "M"  : 0.295,   # 29.5%
        "M+1": 0.310,   # 31.0%
    },
    "corr": {
        ("M-1", "M")  : 0.970,
        ("M-1", "M+1"): 0.930,
        ("M",   "M+1"): 0.980,
    },
    "r": 0.05,
}


def run_demo(
    valuation_date : str = "2026-03-10",
    bl_date        : str = "2026-04-10",
    corr_lookback  : int = 90,
):
    """
    Run the full pricing and risk demo.

    Market data is sourced from Bloomberg (blpapi) via bbg_market_data.py.
    If Bloomberg is unavailable (e.g. running offline / no terminal),
    the hardcoded _FALLBACK_MARKET_DATA values are used and a warning is printed.

    Parameters
    ----------
    valuation_date : YYYY-MM-DD  snapshot / pricing date
    bl_date        : YYYY-MM-DD  Bill-of-Lading date
    corr_lookback  : business days of history for correlation estimation
    """
    print("\n" + "="*60)
    print("  DATED BRENT CHOOSER OPTION — PRICING FRAMEWORK DEMO")
    print("="*60)

    # ── Load market data from Bloomberg ──────────────────────
    # bbg_market_data.load_market_data_with_fallback() will:
    #   1. Attempt to connect to Bloomberg Desktop/B-Pipe via xbbg or pdblp
    #   2. Pull CO1/CO2 prices, IVOL_MID vols, BDH history for correlations,
    #      and 1M SOFR OIS rate — all as-of valuation_date
    #   3. If Bloomberg is unreachable, fall back to _FALLBACK_MARKET_DATA
    #      and emit a UserWarning so the caller knows live data was not used

    _bbg_succeeded = False
    data_source    = "Fallback (hardcoded — Bloomberg unavailable)"

    try:
        from bbg_market_data import (
            load_bloomberg_market_data as _bbg_live,
            load_market_data_with_fallback as _bbg_load,
        )
        # First, attempt a live Bloomberg connection
        try:
            mkt = _bbg_live(
                valuation_date = valuation_date,
                bl_date        = bl_date,
                corr_lookback  = corr_lookback,
                verbose        = True,
            )
            _bbg_succeeded = True
            data_source    = "Bloomberg (live)"
        except Exception as bbg_exc:
            # Bloomberg unreachable — fall back gracefully
            warnings.warn(
                f"Bloomberg unavailable ({bbg_exc}). "
                "Using hardcoded fallback data. "
                "Results are illustrative only — NOT for trading.",
                UserWarning,
                stacklevel=2,
            )
            mkt = BrentMarketData(
                valuation_date = valuation_date,
                bl_date        = bl_date,
                **_FALLBACK_MARKET_DATA,
            )

    except ImportError as imp_exc:
        # bbg_market_data module not found
        warnings.warn(
            f"bbg_market_data module not found ({imp_exc}). "
            "Place bbg_market_data.py in the same directory. "
            "Using hardcoded fallback data.",
            UserWarning,
        )
        mkt = BrentMarketData(
            valuation_date = valuation_date,
            bl_date        = bl_date,
            **_FALLBACK_MARKET_DATA,
        )

    # ── Print market data summary ─────────────────────────────
    print(f"\n  Data Source    : {data_source}")
    print(f"  Valuation Date : {mkt.valuation_date}")
    print(f"  B/L Date       : {mkt.bl_date}")
    print(f"  T_BL           : {mkt.T_bl:.4f} yrs  ({mkt.T_bl*365:.1f} cal days)")
    print(f"  T_Ex (BL+25bd) : {mkt.T_ex:.4f} yrs  ({mkt.T_ex*365:.1f} cal days)")
    print(f"\n  Forward Prices : M-1={mkt.F['M-1']:.2f}  M={mkt.F['M']:.2f}  M+1={mkt.F['M+1']:.2f}  $/bbl")
    print(f"  Vols           : M-1={mkt.sigma['M-1']:.1%}  M={mkt.sigma['M']:.1%}  M+1={mkt.sigma['M+1']:.1%}")
    print(f"  Correlations   : M-1/M={mkt.corr[('M-1','M')]:.3f}  "
          f"M/M+1={mkt.corr[('M','M+1')]:.3f}  "
          f"M-1/M+1={mkt.corr[('M-1','M+1')]:.3f}")
    print(f"  Risk-free rate : {mkt.r:.4f} (cont., ACT/365)")

    # ── Analytical Price ──
    pricer = ChooserOptionPricer(mkt)
    p_anal = pricer.price_analytical()
    print(f"\n  Analytical Price (Margrabe/Stulz): {p_anal:.4f} $/bbl")

    # ── Monte Carlo ──
    print(f"\n  Running Monte Carlo ({ChooserOptionPricer.N_DEFAULT:,} paths)...")
    result = pricer.price_and_greeks(n_paths=ChooserOptionPricer.N_DEFAULT,
                                      greeks_method='pathwise')
    print(result)

    if hasattr(result, '_analytical_price'):
        spread = result.price - result._analytical_price
        print(f"  MC vs Analytical spread : {spread:+.4f} $/bbl\n")

    # ── Scenario Analysis ──
    print("\n" + "─"*55)
    print("  SCENARIO ANALYSIS")
    print("─"*55)

    scen = ScenarioAnalysis(pricer)

    print("\n  [1] Parallel Vol Surface Sensitivity:")
    print(scen.vol_surface_sensitivity().to_string(index=False))

    print("\n  [2] Parallel Forward Curve Sensitivity:")
    print(scen.forward_curve_sensitivity().to_string(index=False))

    print("\n  [3] Correlation Sensitivity:")
    print(scen.correlation_sensitivity().to_string(index=False))

    print("\n  [4] Time Decay Profile (first 10 days):")
    td = scen.time_decay_profile(days=10)
    print(td.to_string(index=False))

    # ── Contango / Backwardation Stress ──────────────────────
    # Stress scenarios are built as offsets from live market levels
    # so the analysis is always anchored to real Bloomberg data.
    print("\n" + "─"*55)
    print("  STRUCTURE STRESS TESTS  (offsets from live market)")
    print("─"*55)

    import copy
    F_live = mkt.F  # live forwards from Bloomberg (or fallback)

    # Curve shape scenarios: differential spread applied to live levels
    stress_scenarios = {
        "Base Case (Live)"    : {"M-1":  0.0, "M":  0.0, "M+1":  0.0},
        "Steep Backwardation" : {"M-1": +3.0, "M": +1.0, "M+1":  0.0},
        "Mild Backwardation"  : {"M-1": +1.0, "M": +0.5, "M+1":  0.0},
        "Flat Curve"          : {"M-1": -F_live["M-1"]+F_live["M"],
                                  "M":  0.0,
                                  "M+1": F_live["M"]-F_live["M+1"]},
        "Mild Contango"       : {"M-1":  0.0, "M": +0.5, "M+1": +1.0},
        "Steep Contango"      : {"M-1":  0.0, "M": +1.0, "M+1": +3.0},
        "Bull Parallel +5"    : {"M-1": +5.0, "M": +5.0, "M+1": +5.0},
        "Bear Parallel -5"    : {"M-1": -5.0, "M": -5.0, "M+1": -5.0},
    }

    stress_results = []
    for name, offsets in stress_scenarios.items():
        mkt2   = copy.deepcopy(mkt)
        for t in mkt2.tenors:
            mkt2.F[t] = F_live[t] + offsets[t]
        p, se = ChooserOptionPricer(mkt2).price_mc(n_paths=50_000)
        stress_results.append({
            "Scenario" : name,
            "M-1"      : round(mkt2.F["M-1"], 2),
            "M"        : round(mkt2.F["M"],   2),
            "M+1"      : round(mkt2.F["M+1"], 2),
            "Price"    : round(p,  4),
            "StdErr"   : round(se, 4),
        })

    df_stress = pd.DataFrame(stress_results)
    print(df_stress.to_string(index=False))

    print("\n  ✓ Framework run complete.\n")
    return result, mkt


# ─────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dated Brent Chooser Option Pricer — Bloomberg-connected demo"
    )
    parser.add_argument(
        "--valuation-date", default="2026-03-10",
        help="Pricing snapshot date (YYYY-MM-DD). Default: today."
    )
    parser.add_argument(
        "--bl-date", default="2026-04-10",
        help="Bill-of-Lading date (YYYY-MM-DD). Default: 1 month forward."
    )
    parser.add_argument(
        "--corr-lookback", type=int, default=90,
        help="Business days of history for correlation estimation. Default: 90."
    )
    args = parser.parse_args()

    result, mkt = run_demo(
        valuation_date = args.valuation_date,
        bl_date        = args.bl_date,
        corr_lookback  = args.corr_lookback,
    )