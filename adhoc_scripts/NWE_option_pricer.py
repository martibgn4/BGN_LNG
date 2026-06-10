"""
Egypt LNG Cancellation Option Valuation
========================================
Monte Carlo pricer for 4 monthly cancellable cargoes sold DES Egypt at TTF + 0.80.
Cancellation decision driven by: TTF level, NWE basis, freight, BOG cost, financing cost.

Author: Marti
"""

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

# ============================================================================
# 1. CONTRACT & MARKET PARAMETERS
# ============================================================================

VALUATION_DATE = date(2026, 5, 21)


freight_seasonals = [0.0099, -0.013125,	-0.033625,	-0.035125,	-0.039125,	-0.032125,	-0.028625,	-0.021625,	0.005375,	0.038375,	0.059875,	0.089875]
nwe_seasonals = [-0.004166667,	-0.014166667,	-0.024166667,	-0.004166667,	0.000833333,	0.000833333,	0.000833333,	0.000833333,	0.005833333,	0.005833333,	0.010833333,	0.020833333]
ttf_seasonals = [3.357637453,	2.971882698,	2.406109843,	-0.117583694,	-0.881892501,	-1.104773025,	-1.125689505,	-1.125688982,	-1.090714408,	-1.149006237,	-1.130147116,	-1.010134526]



# Cargo schedule: one per quarter in 2027
CARGOES = [
    {"month": "Jan-27", "delivery": date(2027, 1, 15), "ttf_seasonal": ttf_seasonals[0], "nwe_seasonal": nwe_seasonals[0], "freight_seasonal": freight_seasonals[0]},
    {"month": "Feb-27", "delivery": date(2027, 2, 15), "ttf_seasonal": ttf_seasonals[1], "nwe_seasonal": nwe_seasonals[1], "freight_seasonal": freight_seasonals[1]},
    {"month": "Mar-27", "delivery": date(2027, 3, 15), "ttf_seasonal": ttf_seasonals[2], "nwe_seasonal": nwe_seasonals[2], "freight_seasonal": freight_seasonals[2]},
    {"month": "Apr-27", "delivery": date(2027, 4, 15), "ttf_seasonal": ttf_seasonals[3], "nwe_seasonal": nwe_seasonals[3], "freight_seasonal": freight_seasonals[3]},
    {"month": "May-27", "delivery": date(2027, 5, 15), "ttf_seasonal": ttf_seasonals[4], "nwe_seasonal": nwe_seasonals[4], "freight_seasonal": freight_seasonals[4]},
    {"month": "Jun-27", "delivery": date(2027, 6, 15), "ttf_seasonal": ttf_seasonals[5], "nwe_seasonal": nwe_seasonals[5], "freight_seasonal": freight_seasonals[5]},
    {"month": "Jul-27", "delivery": date(2027, 7, 15), "ttf_seasonal": ttf_seasonals[6], "nwe_seasonal": nwe_seasonals[6], "freight_seasonal": freight_seasonals[6]},
    {"month": "Aug-27", "delivery": date(2027, 8, 15), "ttf_seasonal": ttf_seasonals[7], "nwe_seasonal": nwe_seasonals[7], "freight_seasonal": freight_seasonals[7]},
    {"month": "Sep-27", "delivery": date(2027, 9, 15), "ttf_seasonal": ttf_seasonals[8], "nwe_seasonal": nwe_seasonals[8], "freight_seasonal": freight_seasonals[8]},
    {"month": "Oct-27", "delivery": date(2027, 10, 15), "ttf_seasonal": ttf_seasonals[9], "nwe_seasonal": nwe_seasonals[9], "freight_seasonal": freight_seasonals[9]},
    {"month": "Nov-27", "delivery": date(2027, 11, 15), "ttf_seasonal": ttf_seasonals[10], "nwe_seasonal": nwe_seasonals[10], "freight_seasonal": freight_seasonals[10]},
    {"month": "Dec-27", "delivery": date(2027, 12, 15), "ttf_seasonal": ttf_seasonals[11], "nwe_seasonal": nwe_seasonals[11], "freight_seasonal": freight_seasonals[11]},
]

freight_seasonals28 = [ i + 0.01 for i in [0.0099, -0.013125,	-0.033625,	-0.035125,	-0.039125,	-0.032125,	-0.028625,	-0.021625,	0.005375,	0.038375,	0.059875,	0.089875]]
nwe_seasonals28 = [-0.004166667,	-0.014166667,	-0.024166667,	-0.004166667,	0.000833333,	0.000833333,	0.000833333,	0.000833333,	0.005833333,	0.005833333,	0.010833333,	0.020833333]
ttf_seasonals28 = [i - 4 for i in [3.357637453,	2.971882698,	2.406109843,	-0.117583694,	-0.881892501,	-1.104773025,	-1.125689505,	-1.125688982,	-1.090714408,	-1.149006237,	-1.130147116,	-1.010134526]]

#
# CARGOES = [
#     {"month": "Jan-28", "delivery": date(2028, 1, 15), "ttf_seasonal": ttf_seasonals28[0], "nwe_seasonal": nwe_seasonals28[0], "freight_seasonal": freight_seasonals28[0]},
#     {"month": "Feb-28", "delivery": date(2028, 2, 15), "ttf_seasonal": ttf_seasonals28[1], "nwe_seasonal": nwe_seasonals28[1], "freight_seasonal": freight_seasonals28[1]},
#     {"month": "Mar-28", "delivery": date(2028, 3, 15), "ttf_seasonal": ttf_seasonals28[2], "nwe_seasonal": nwe_seasonals28[2], "freight_seasonal": freight_seasonals28[2]},
#     {"month": "Apr-28", "delivery": date(2028, 4, 15), "ttf_seasonal": ttf_seasonals28[3], "nwe_seasonal": nwe_seasonals28[3], "freight_seasonal": freight_seasonals28[3]},
#     {"month": "May-28", "delivery": date(2028, 5, 15), "ttf_seasonal": ttf_seasonals28[4], "nwe_seasonal": nwe_seasonals28[4], "freight_seasonal": freight_seasonals28[4]},
#     {"month": "Jun-28", "delivery": date(2028, 6, 15), "ttf_seasonal": ttf_seasonals28[5], "nwe_seasonal": nwe_seasonals28[5], "freight_seasonal": freight_seasonals28[5]},
#     {"month": "Jul-28", "delivery": date(2028, 7, 15), "ttf_seasonal": ttf_seasonals28[6], "nwe_seasonal": nwe_seasonals28[6], "freight_seasonal": freight_seasonals28[6]},
#     {"month": "Aug-28", "delivery": date(2028, 8, 15), "ttf_seasonal": ttf_seasonals28[7], "nwe_seasonal": nwe_seasonals28[7], "freight_seasonal": freight_seasonals28[7]},
#     {"month": "Sep-28", "delivery": date(2028, 9, 15), "ttf_seasonal": ttf_seasonals28[8], "nwe_seasonal": nwe_seasonals28[8], "freight_seasonal": freight_seasonals28[8]},
#     {"month": "Oct-28", "delivery": date(2028, 10, 15), "ttf_seasonal": ttf_seasonals28[9], "nwe_seasonal": nwe_seasonals28[9], "freight_seasonal": freight_seasonals28[9]},
#     {"month": "Nov-28", "delivery": date(2028, 11, 15), "ttf_seasonal": ttf_seasonals28[10], "nwe_seasonal": nwe_seasonals28[10], "freight_seasonal": freight_seasonals28[10]},
#     {"month": "Dec-28", "delivery": date(2028, 12, 15), "ttf_seasonal": ttf_seasonals28[11], "nwe_seasonal": nwe_seasonals28[11], "freight_seasonal": freight_seasonals28[11]},
# ]

CARGO_MMBTU = 3_500_000  # 174k m3 TFDE standard delivered volume
FEE = 0.80  # USD/MMBtu, the deal margin over TTF
NOTICE_DAYS = 30  # cancellation notice before delivery

# TTF dynamics (GBM)
TTF_FORWARD = 12.92  # USD/MMBtu, flat Cal-27 forward
TTF_VOL = 0.50  # 50% annualised vol

# NWE basis to TTF (OU mean-reverting, negative = NWE discount)
NWE_MEAN = -0.505  # long-run mean basis
NWE_VOL = 0.35  # annualised stdev
NWE_KAPPA = 3.0  # mean reversion speed (half-life ~3M)
NWE_INIT = -0.505  # starting point

# Freight differential NWE->Egypt vs NWE->NWE drop, in USD/MMBtu of cargo
FREIGHT_MEAN = 0.12  # 55,000 usd/day * 7 days / 3_500_000


FREIGHT_VOL = 0.10
FREIGHT_KAPPA = 2.0
FREIGHT_INIT = 0.12

# Egypt reload premium (treated as deterministic)
EGYPT_PREMIUM = 0.05  # give 175_000 USD to seller

# Correlations
RHO_TTF_NWE = -0.40  # high TTF -> NWE less discounted
RHO_TTF_FREIGHT = +0.45  # winter spike correlation
RHO_NWE_FREIGHT = -0.25  # tight market: NWE discount + high freight

# BOG
BOG_LOSS_PCT = 0.007  # 0.10%/day * 7 days

# Financing (LC-wrapped EGAS receivable)
FINANCING_DAYS = 180
FINANCING_RATE = 0.055  # SOFR + LC fee + capital charge

# Discounting
RISK_FREE = 0.045

# Monte Carlo
N_PATHS = 500_000
SEED = 42


# ============================================================================
# 2. SIMULATION ENGINE
# ============================================================================

def simulate_drivers(T_years: float, n_paths: int, rng: np.random.Generator):
    """
    Simulate (TTF, NWE_basis, Freight) at time T using correlated shocks.
    - TTF: Geometric Brownian Motion under risk-neutral measure
    - NWE_basis: Ornstein-Uhlenbeck mean-reverting
    - Freight: Ornstein-Uhlenbeck mean-reverting
    Returns three arrays of length n_paths.
    """
    # Correlation matrix
    corr = np.array([
        [1.0, RHO_TTF_NWE, RHO_TTF_FREIGHT],
        [RHO_TTF_NWE, 1.0, RHO_NWE_FREIGHT],
        [RHO_TTF_FREIGHT, RHO_NWE_FREIGHT, 1.0],
    ])
    L = np.linalg.cholesky(corr)

    # Independent standard normals
    Z = rng.standard_normal(size=(3, n_paths))
    # Correlated shocks
    W = L @ Z  # shape (3, n_paths)

    # 1. TTF via GBM (drift = 0 under risk-neutral, since forward = E[S_T])
    #    S_T = F * exp(-0.5*sigma^2*T + sigma*sqrt(T)*W1)
    sigma_t = TTF_VOL
    ttf = TTF_FORWARD * np.exp(-0.5 * sigma_t ** 2 * T_years + sigma_t * np.sqrt(T_years) * W[0])

    # import matplotlib.pyplot as plt
    # plt.hist(ttf, density=True, bins=30)  # density=False would make counts
    # plt.ylabel('TTF')
    # plt.xlabel('USD/mmbtu');
    # plt.show()

    # 2. NWE basis via OU - exact discretisation
    #    X_T = X0*exp(-kappa*T) + mean*(1-exp(-kappa*T)) + sigma*sqrt((1-exp(-2kT))/2k)*W2
    decay = np.exp(-NWE_KAPPA * T_years)
    nwe_std = NWE_VOL * np.sqrt((1 - np.exp(-2 * NWE_KAPPA * T_years)) / (2 * NWE_KAPPA))
    nwe = NWE_INIT * decay + NWE_MEAN * (1 - decay) + nwe_std * W[1]

    # plt.hist(nwe, density=True, bins=30)  # density=False would make counts
    # plt.ylabel('NWE discount')
    # plt.xlabel('USD/mmbtu');
    # plt.show()

    # 3. Freight via OU
    decay_f = np.exp(-FREIGHT_KAPPA * T_years)
    fr_std = FREIGHT_VOL * np.sqrt((1 - np.exp(-2 * FREIGHT_KAPPA * T_years)) / (2 * FREIGHT_KAPPA))
    freight = FREIGHT_INIT * decay_f + FREIGHT_MEAN * (1 - decay_f) + fr_std * W[2]
    # Floor freight at 0 (negative freight is unphysical)
    freight = np.maximum(freight, 0.0)

    # plt.hist(freight, density=True, bins=30)  # density=False would make counts
    # plt.ylabel('Freight')
    # plt.xlabel('USD/mmbtu');
    # plt.show()


    return ttf, nwe, freight


def cargo_payoff(ttf, nwe, freight, ttf_seasonal_adj, nwe_seasonal_adj, freight_seasonal_adj):
    """
    For each path, compute the cancellation-aware payoff per MMBtu.
    Margin = 0.80 - NWE_disc - Freight_diff - Egypt_prem - BOG - Financing
    where NWE_disc = -nwe (positive number when NWE is discounted)

    Apply seasonal adjustment to TTF for delivery month.
    """
    ttf_effective = ttf + ttf_seasonal_adj
    ttf_effective = np.maximum(ttf_effective, 0.1)  # floor for safety

    nwe_discount = -nwe  + nwe_seasonal_adj  # positive = bad
    freight_cost = freight + freight_seasonal_adj   # already $/MMBtu
    egypt_prem = EGYPT_PREMIUM
    bog_cost = BOG_LOSS_PCT * ttf_effective
    fin_cost = (ttf_effective + FEE) * FINANCING_RATE * (FINANCING_DAYS / 360)

    margin = FEE - nwe_discount - freight_cost - egypt_prem - bog_cost - fin_cost

    # European option: deliver if margin > 0, else cancel (payoff = 0)
    payoff = np.maximum(margin, 0.0)

    return payoff, margin


def value_cargo(cargo, rng):
    """Run MC for a single cargo and return diagnostics."""
    delivery = cargo["delivery"]
    T_years = (delivery - VALUATION_DATE).days / 365.0

    # Notice period: decision made 60 days before delivery
    decision_date = delivery - relativedelta(days=NOTICE_DAYS)
    T_decision = (decision_date - VALUATION_DATE).days / 365.0

    # Simulate at the decision date (when exercise happens)
    ttf, nwe, freight = simulate_drivers(T_decision, N_PATHS, rng)

    payoff_per_mmbtu, margin = cargo_payoff(ttf, nwe, freight, cargo["ttf_seasonal"], cargo["nwe_seasonal"], cargo["freight_seasonal"])

    # Total cargo payoff in USD
    payoff_usd = payoff_per_mmbtu * CARGO_MMBTU

    # Discount to today (use delivery date for cashflow timing)
    discount = np.exp(-RISK_FREE * T_years)
    pv_per_path = payoff_usd * discount

    pv_mean = pv_per_path.mean()
    pv_stderr = pv_per_path.std() / np.sqrt(N_PATHS)
    cancel_prob = (margin <= 0).mean()

    # Conditional stats
    delivered_margin = margin[margin > 0]
    avg_margin_when_delivered = delivered_margin.mean() if len(delivered_margin) > 0 else 0

    return {
        "month": cargo["month"],
        "T_decision_yrs": round(T_decision, 3),
        "cancel_prob": cancel_prob,
        "avg_margin_deliv": avg_margin_when_delivered,
        "pv_usd": pv_mean,
        "pv_stderr_usd": pv_stderr,
        "pv_per_mmbtu": pv_mean / CARGO_MMBTU,
        # Driver stats for diagnostics
        "mean_ttf": ttf.mean() + cargo["ttf_seasonal"],
        "mean_nwe": nwe.mean() + cargo["nwe_seasonal"],
        "mean_freight": freight.mean() + cargo["freight_seasonal"],
        "mean_margin": margin.mean()
    }


# ============================================================================
# 3. RUN VALUATION
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    results = [value_cargo(c, rng) for c in CARGOES]
    df = pd.DataFrame(results)

    total_pv = df["pv_usd"].sum()
    total_se = np.sqrt((df["pv_stderr_usd"] ** 2).sum())

    print("=" * 78)
    print("EGYPT CANCELLATION OPTION VALUATION - BASE CASE")
    print("=" * 78)
    print(f"Valuation date:     {VALUATION_DATE}")
    print(f"MC paths:           {N_PATHS:,}")
    print(f"Financing rate:     {FINANCING_RATE:.2%} (LC-wrapped)")
    print(f"TTF Cal-27 fwd:     ${TTF_FORWARD:.2f}/MMBtu, vol {TTF_VOL:.0%}")
    print(f"NWE basis:          ${NWE_MEAN:.2f}/MMBtu (mean), vol ${NWE_VOL:.2f}")
    print(f"Freight diff:       ${FREIGHT_MEAN:.2f}/MMBtu (mean), vol ${FREIGHT_VOL:.2f}")
    print(f"BOG:                {BOG_LOSS_PCT:.1%} of cargo")
    print()
    print(df.to_string(
        index=False,
        formatters={
            "T_decision_yrs": "{:.3f}".format,
            "cancel_prob": "{:.1%}".format,
            "avg_margin_deliv": "${:.3f}".format,
            "pv_usd": "${:,.0f}".format,
            "pv_stderr_usd": "${:,.0f}".format,
            "pv_per_mmbtu": "${:.3f}".format,
            "mean_ttf": "${:.2f}".format,
            "mean_nwe": "${:.2f}".format,
            "mean_freight": "${:.2f}".format,
            "mean_margin": "${:.3f}".format,
        },
    ))
    print()
    print(f"TOTAL OPTION VALUE (PV):  ${total_pv:,.0f}  ± ${1.96 * total_se:,.0f} (95% CI)")
    print(f"Per cargo average:        ${total_pv / 12:,.0f}")
    print(f"Per MMBtu average:        ${total_pv / (12 * CARGO_MMBTU):.3f}/MMBtu")
    print()


    # ============================================================================
    # 4. GREEKS - finite difference
    # ============================================================================

    def revalue_all(override=None):
        """Re-run valuation with one parameter shifted. override is dict of globals to set."""
        if override:
            saved = {k: globals()[k] for k in override}
            globals().update(override)
        rng_local = np.random.default_rng(SEED)  # same seed for variance reduction
        res = [value_cargo(c, rng_local) for c in CARGOES]
        tot = sum(r["pv_usd"] for r in res)
        if override:
            globals().update(saved)
        return tot


    base_pv = total_pv

    # Delta to TTF forward: bump +$1
    pv_ttf_up = revalue_all({"TTF_FORWARD": TTF_FORWARD + 1.0})
    delta_ttf = pv_ttf_up - base_pv

    # Vega TTF: bump vol +1%
    pv_vol_up = revalue_all({"TTF_VOL": TTF_VOL + 0.01})
    vega_ttf = pv_vol_up - base_pv

    # Vega NWE basis: bump basis vol +$0.10
    pv_nwevol_up = revalue_all({"NWE_VOL": NWE_VOL + 0.10})
    vega_nwe = pv_nwevol_up - base_pv

    # Vega Freight: bump freight vol +$0.05
    pv_frvol_up = revalue_all({"FREIGHT_VOL": FREIGHT_VOL + 0.05})
    vega_fr = pv_frvol_up - base_pv

    # Sensitivity to mean NWE: widen discount by $0.10 (more negative)
    pv_nwe_down = revalue_all({"NWE_MEAN": NWE_MEAN - 0.10, "NWE_INIT": NWE_INIT - 0.10})
    nwe_delta = pv_nwe_down - base_pv

    # Sensitivity to mean freight: bump up $0.05
    pv_fr_up = revalue_all({"FREIGHT_MEAN": FREIGHT_MEAN + 0.05, "FREIGHT_INIT": FREIGHT_INIT + 0.05})
    freight_delta = pv_fr_up - base_pv

    # Sensitivity to financing rate: drop to 5%, raise to 7%
    pv_fin_low = revalue_all({"FINANCING_RATE": 0.050})
    pv_fin_high = revalue_all({"FINANCING_RATE": 0.070})

    print("=" * 78)
    print("GREEKS & SENSITIVITIES")
    print("=" * 78)
    print(f"Base PV:                              ${base_pv:,.0f}")
    print()
    print(f"Delta TTF (+$1 fwd):                  ${delta_ttf:+,.0f}")
    print(f"Vega TTF (+1% vol):                   ${vega_ttf:+,.0f}")
    print(f"Vega NWE basis (+$0.10 vol):          ${vega_nwe:+,.0f}")
    print(f"Vega Freight (+$0.05 vol):            ${vega_fr:+,.0f}")
    print(f"NWE basis widen $0.10 (more disc):    ${nwe_delta:+,.0f}")
    print(f"Freight mean +$0.05:                  ${freight_delta:+,.0f}")
    print(f"Financing rate 5.00%:                 ${pv_fin_low:,.0f}  ({pv_fin_low - base_pv:+,.0f})")
    print(f"Financing rate 7.00%:                 ${pv_fin_high:,.0f}  ({pv_fin_high - base_pv:+,.0f})")
    print()

    # ============================================================================
    # 5. SCENARIO ANALYSIS
    # ============================================================================

    print("=" * 78)
    print("SCENARIO ANALYSIS")
    print("=" * 78)

    scenarios = {
        "Base case": {},
        "Loose EU (NWE -$0.75)": {"NWE_MEAN": -0.75, "NWE_INIT": -0.75},
        "Tight EU (NWE +$0.10)": {"NWE_MEAN": 0.10, "NWE_INIT": 0.10},
        "High freight ($0.45)": {"FREIGHT_MEAN": 0.45, "FREIGHT_INIT": 0.45},
        "Low freight ($0.10)": {"FREIGHT_MEAN": 0.10, "FREIGHT_INIT": 0.10},
        "TTF spike ($20)": {"TTF_FORWARD": 20.0},
        "TTF crash ($5)": {"TTF_FORWARD": 5.0},
        "High vol (TTF 70%)": {"TTF_VOL": 0.70},
        "No LC (r=10%)": {"FINANCING_RATE": 0.10},
    }

    scen_results = []
    for name, override in scenarios.items():
        pv = revalue_all(override)
        scen_results.append({"scenario": name, "pv_usd": pv, "pv_per_mmbtu": pv / (4 * CARGO_MMBTU)})

    scen_df = pd.DataFrame(scen_results)
    print(scen_df.to_string(
        index=False,
        formatters={
            "pv_usd": "${:,.0f}".format,
            "pv_per_mmbtu": "${:.3f}".format,
        }
    ))
