import numpy as np
from datetime import date

def price_basket_option_mc(vol_des, vol_brent301, vol_freight,
                           rho_des_brent, rho_des_freight, rho_brent_freight,
                           slope, r, T, route_days, mmbtu_start, mmbtu_end,
                           S0_des, S0_brent301, S0_freight, N_PATHS, antithetic=True,
                           extra_costs = 0.0):
    # 1. Initialize Price Arrays
    # Initialize paths with starting forward prices
    # des_paths = np.full((N_PATHS, 1), S0_des)
    # brent_paths = np.full((N_PATHS, 1), S0_brent301)
    # freight_paths = np.full((N_PATHS, 1), S0_freight)


    # Time vector for path simulation
    # t_values = np.linspace(0.0, T, N_STEPS + 1)

    # 2. Generate Correlated Random Numbers
    # Generate standard normal random variables
    corr_matrix = np.array([
        [1.0, rho_des_brent, rho_des_freight],
        [rho_des_brent, 1.0, rho_brent_freight],
        [rho_des_freight, rho_brent_freight, 1.0]
    ])

    L = np.linalg.cholesky(corr_matrix)

    # Determine total paths accounting for antithetic variates
    m = N_PATHS if not antithetic else (N_PATHS // 2)
    if antithetic and N_PATHS % 2 != 0:
        # make even for clean pairing
        N_PATHS += 1
        m = N_PATHS // 2

    # Precompute drifts and vol*sqrt(T)
    mu_des = (r - 0.5 * vol_des ** 2) * T
    mu_brent = (r - 0.5 * vol_brent301 ** 2) * T
    mu_freight = (r - 0.5 * vol_freight ** 2) * T
    sig_sqrtT_des = vol_des * np.sqrt(T)
    sig_sqrtT_brent = vol_brent301 * np.sqrt(T)
    sig_sqrtT_freight = vol_freight * np.sqrt(T)

    # Draw independent standard normals and correlate
    Z = rng.standard_normal(size=(m, 3))
    Zc = Z @ L.T  # correlated

    # Antithetic counterpart
    if antithetic:
        Zc_anti = (-Z) @ L.T
        Zc_full = np.vstack([Zc, Zc_anti])
    else:
        Zc_full = Zc

    # Simulate terminal prices (vectorized)
    # S_T[i] = S0[i] * exp(mu[i] + sig_sqrtT[i] * Zc_full[:, i])
    expo_des = mu_des + sig_sqrtT_des * Zc_full[:,0]
    expo_brent = mu_brent + sig_sqrtT_brent * Zc_full[:,1]
    expo_freight = mu_freight + sig_sqrtT_freight * Zc_full[:,2]
    ST_des = S0_des * np.exp(expo_des)   # in USD/MMBtu
    ST_brent = S0_brent301 * np.exp(expo_brent)  # in USD/bbl
    ST_freight = S0_freight * np.exp(expo_freight)  # in USD/day

    # Payoff
    forward_payoff = mmbtu_end * ST_des - slope * mmbtu_start * ST_brent - ST_freight * route_days - extra_costs
    payoff = np.maximum(forward_payoff, 0.0)  # directly in USD

    # Discount & statistics
    disc_payoff = np.exp(-r * T) * payoff
    price = np.mean(disc_payoff)
    # Use sample standard deviation and SE = sd / sqrt(n)
    sd = np.std(disc_payoff, ddof=1)
    stderr = sd / np.sqrt(disc_payoff.size)
    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    disc_payoff_forward = np.exp(-r * T) * forward_payoff
    price_forward = np.mean(disc_payoff_forward)
    # Use sample standard deviation and SE = sd / sqrt(n)
    sd_forward = np.std(disc_payoff_forward, ddof=1)
    stderr_forward = sd_forward / np.sqrt(disc_payoff_forward.size)
    ci95_forward = (price_forward - 1.96 * stderr_forward, price_forward + 1.96 * stderr_forward)

    return (float(price), float(stderr), (float(ci95[0]), float(ci95[1])),
            float(price_forward), float(stderr_forward), (float(ci95_forward[0]), float(ci95_forward[1])))

if __name__ == "__main__":
    vol_des = 0.4
    vol_brent301 = 0.25
    vol_freight = 0.5
    rho_des_brent = 0.4
    rho_des_freight = 0.2
    rho_brent_freight = 0.3

    r = 0.02
    route_days = 20
    today_date = date(2025, 12, 11)
    fob_delivery_date = date(2026, 5, 15)

    potential_delivery_dates = [date(2026, m, 15) for m in [1, 5, 6, 10]]

    T = (fob_delivery_date - today_date).days / 365.0

    N_PATHS = 1000000
    S0_des = 9.0  # USD/MMBtu
    S0_brent301 = 65  # USD/bbl
    S0_freight = 20000  # USD/day
    mmbtu_start = 3500000  # MMBtu
    boiloff_rate = 0.001  # %MMBtu / day
    mmbtu_end = mmbtu_start * (1.0 - route_days * boiloff_rate)
    slope = 0.13
    extra_costs = 0.0  # In USD
    intrinsic_value = np.exp(-r*T) * (mmbtu_end * S0_des - slope * mmbtu_start * S0_brent301 - S0_freight * route_days - extra_costs)
    print(f"Current ATM, in mUSD: {intrinsic_value/1e6:.3f}")

    for seed in range(3):
        rng = np.random.default_rng(seed=seed)

        option_price, std_option, ci_option, forward_price, std_forward, ci_forward = price_basket_option_mc(
            vol_des, vol_brent301, vol_freight, rho_des_brent, rho_des_freight, rho_brent_freight,
            slope, r, T, route_days, mmbtu_start, mmbtu_end,
            S0_des, S0_brent301, S0_freight,
            N_PATHS, antithetic=True,
            extra_costs=extra_costs
        )


        print(f"Seed: {seed}, MC option price, in mUSD: {option_price/1e6:.3f}  (SE: {std_option/1e6:.3f})  95% CI: [{ci_option[0]/1e6:.3f}, {ci_option[1]/1e6:.3f}]")
        print(f"Seed: {seed}, MC forward price, in mUSD: {forward_price/1e6:.3f}  (SE: {std_forward/1e6:.3f})  95% CI: [{ci_forward[0]/1e6:.3f}, {ci_forward[1]/1e6:.3f}]")
