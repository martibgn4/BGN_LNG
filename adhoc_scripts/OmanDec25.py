import numpy as np
from datetime import date

from Utils.utils_maths import jump_rv


def price_basket_option_mc(vol_des, vol_brent301, vol_freight,
                           rho_des_brent, rho_des_freight, rho_brent_freight,
                           slope, r, T, route_days, mmbtu_start, mmbtu_end,
                           S0_des, S0_brent301, S0_freight, N_PATHS, antithetic=True, mc_seed=1,
                           extra_costs=0.0,
                           big_jump_yearly_probability=0.0, big_jump_size=0.0,
                           small_jump_yearly_probability=0.0, small_jump_size=0.0,
                           report_simulated_prices=False):
    # 1. Initialize Price Arrays
    # Initialize paths with starting forward prices
    # des_paths = np.full((N_PATHS, 1), S0_des)
    # brent_paths = np.full((N_PATHS, 1), S0_brent301)
    # freight_paths = np.full((N_PATHS, 1), S0_freight)

    # Time vector for path simulation
    # t_values = np.linspace(0.0, T, N_STEPS + 1)
    rng = np.random.default_rng(seed=mc_seed)
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
    expo_des = mu_des + sig_sqrtT_des * Zc_full[:, 0]
    expo_brent = mu_brent + sig_sqrtT_brent * Zc_full[:, 1]
    expo_freight = mu_freight + sig_sqrtT_freight * Zc_full[:, 2]
    ST_des = S0_des * np.exp(expo_des)  # in USD/MMBtu
    ST_brent = S0_brent301 * np.exp(expo_brent)  # in USD/bbl
    ST_freight = S0_freight * np.exp(expo_freight)  # in USD/day


    if big_jump_size != 0.0:
        # Rescale yearly probabsize so that it has denoted yearly occurence on average
        actual_jump_yearly_probability = 1 - (1-big_jump_yearly_probability)**T

        jump = jump_rv(N_PATHS, p=actual_jump_yearly_probability, mu=big_jump_size, sigma=0.001)
        ST_des += jump

    if small_jump_size != 0.0:
        # Rescale probab
        actual_jump_yearly_probability = 1 - (1-small_jump_yearly_probability)**T

        jump = jump_rv(N_PATHS, p=actual_jump_yearly_probability, mu=small_jump_size, sigma=0.001)
        ST_des += jump

    sim_spread = ST_des - ST_brent

    # Payoff
    forward_payoff = mmbtu_end * ST_des - slope * mmbtu_start * ST_brent - ST_freight * route_days - extra_costs
    payoff = np.maximum(forward_payoff, 0.0)  # directly in USD

    intrinsic = mmbtu_end * S0_des - slope * mmbtu_start * S0_brent301 - S0_freight * route_days - extra_costs
    intrinsic = np.exp(-r * T) * intrinsic

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

    if report_simulated_prices:
        return sim_spread
    else:
        return (float(price), float(stderr),
                float(price_forward), float(stderr_forward),
                intrinsic)


def price_long_basket_option_mc(vol_u1, vol_u2, vol_u3,
                                rho_u1u2, rho_u1u3, rho_u2u3,
                                weight_u1, weight_u2, weight_u3,
                                r, T,
                                price_u1, price_u2, price_u3,
                                N_PATHS, antithetic=True, mc_seed=1,
                                extra_costs=0.0):
    # 1. Initialize Price Arrays
    # Initialize paths with starting forward prices
    # des_paths = np.full((N_PATHS, 1), S0_des)
    # brent_paths = np.full((N_PATHS, 1), S0_brent301)
    # freight_paths = np.full((N_PATHS, 1), S0_freight)

    # Time vector for path simulation
    # t_values = np.linspace(0.0, T, N_STEPS + 1)
    rng = np.random.default_rng(seed=mc_seed)
    # 2. Generate Correlated Random Numbers
    # Generate standard normal random variables
    corr_matrix = np.array([
        [1.0, rho_u1u2, rho_u1u3],
        [rho_u1u2, 1.0, rho_u2u3],
        [rho_u1u3, rho_u2u3, 1.0]
    ])

    L = np.linalg.cholesky(corr_matrix)

    # Determine total paths accounting for antithetic variates
    m = N_PATHS if not antithetic else (N_PATHS // 2)
    if antithetic and N_PATHS % 2 != 0:
        # make even for clean pairing
        N_PATHS += 1
        m = N_PATHS // 2

    # Precompute drifts and vol*sqrt(T)
    mu_1 = (r - 0.5 * vol_u1 ** 2) * T
    mu_2 = (r - 0.5 * vol_u2 ** 2) * T
    mu_3 = (r - 0.5 * vol_u3 ** 2) * T
    sig_sqrtT_u1 = vol_u1 * np.sqrt(T)
    sig_sqrtT_u2 = vol_u2 * np.sqrt(T)
    sig_sqrtT_u3 = vol_u3 * np.sqrt(T)

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
    expo_1 = mu_1 + sig_sqrtT_u1 * Zc_full[:, 0]
    expo_2 = mu_2 + sig_sqrtT_u2 * Zc_full[:, 1]
    expo_3 = mu_3 + sig_sqrtT_u3 * Zc_full[:, 2]
    ST_1 = price_u1 * np.exp(expo_1)  # in USD/MMBtu
    ST_2 = price_u2 * np.exp(expo_2)  # in USD/MMBtu
    ST_3 = price_u3 * np.exp(expo_3)  # in USD/MMbtu

    # Payoff
    forward_payoff = weight_u1 * ST_1 - weight_u2 * ST_2 - weight_u3 * ST_3 - extra_costs
    payoff = np.maximum(forward_payoff, 0.0)  # directly in USD

    intrinsic = weight_u1 * price_u1 - weight_u2 * price_u2 - weight_u3 * price_u3 - extra_costs
    intrinsic = np.exp(-r * T) * intrinsic

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

    return (float(price), float(stderr),
            float(price_forward), float(stderr_forward),
            intrinsic)


if __name__ == "__main__":
    vol_jkm = 0.2
    vol_ttf = 0.22
    vol_freight = 0.001
    rho_jkm_ttf = 0.995
    rho_jkm_freight = 0.99
    rho_ttf_freight = 0.99

    r = 0.02
    route_days = 20
    today_date = date(2026, 3, 17)
    fob_delivery_date = date(2027, 5, 15)

    # potential_delivery_dates = [date(2026, m, 15) for m in [1, 5, 6, 10]]

    T = (fob_delivery_date - today_date).days / 365.0

    N_PATHS = 1000000
    S0_jkm = 10.0  # USD/MMBtu
    S0_ttf = 9.9 # USD/bbl
    S0_freight = 20000  # USD/day
    mmbtu_start = 3500000  # MMBtu
    boiloff_rate = 0.00085  # %MMBtu / day
    mmbtu_end = mmbtu_start * (1.0 - route_days * boiloff_rate)
    slope = 1.0
    extra_costs = 0.0  # In USD
    intrinsic_value = np.exp(-r * T) * (
                mmbtu_end * S0_jkm - slope * mmbtu_start * S0_ttf - S0_freight * route_days - extra_costs)
    print(f"Current ATM, in mUSD: {intrinsic_value / 1e6:.3f}")

    rng = np.random.default_rng(seed=1)

    sim_spread = price_basket_option_mc(
        vol_jkm, vol_ttf, vol_freight, rho_jkm_ttf, rho_jkm_freight, rho_ttf_freight,
        slope, r, T, route_days, mmbtu_start, mmbtu_end,
        S0_jkm, S0_ttf, S0_freight,
        N_PATHS, antithetic=True,
        extra_costs=extra_costs,
        big_jump_yearly_probability=0.01, big_jump_size=3.0,
        small_jump_yearly_probability=0.03, small_jump_size=1.5,
        report_simulated_prices=True
    )

    import matplotlib.pyplot as plt
    fig2, ax2 = plt.subplots(figsize=(4, 2))
    ax2.hist(sim_spread, bins=40)
    ax2.set_title("Simulated JKM - TFU spread");
    plt.show()
