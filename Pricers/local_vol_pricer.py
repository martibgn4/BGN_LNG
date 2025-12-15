import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def generate_local_vol_surface(tenors, strikes, ivol_matrix, interpolation_method,
                               todays_date, spot, r=0.0, div_rate=0.0):
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_counter = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = todays_date
    # Term Structures for Discounting
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, r, day_counter)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, div_rate, day_counter)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

    # 2. Market Data (Implied Volatility Surface)
    # Maturities in years
    expiries = [todays_date + ql.Period(int(t * 365.25), ql.Days) for t in tenors]

    # Strikes
    # Implied Volatility Matrix (Example Data)
    # Rows: Strikes, Columns: Maturities (T1, T2, T3, T4)
    implied_vols = ql.Matrix(len(strikes), len(expiries))
    for i in range(len(strikes)):
        for j in range(len(expiries)):
            implied_vols[i][j] = ivol_matrix[i][j]

    # 3. Construct the Black Variance Surface (Input)
    black_var_surface = ql.BlackVarianceSurface(
        todays_date,
        calendar,
        expiries,
        strikes.tolist(),
        implied_vols,
        day_counter
    )
    black_var_surface.setInterpolation(interpolation_method)
    # 4. Construct the Local Volatility Surface (Output)
    local_vol_surface = ql.LocalVolSurface(
        ql.BlackVolTermStructureHandle(black_var_surface),
        flat_ts,
        dividend_ts,
        spot_handle
    )
    local_vol_surface.enableExtrapolation()

    return local_vol_surface


if __name__ == "__main__":
    plot_VOL_GRAPHS = False

    # 1. Setup (Dates, Rates, Spot)
    interpolation_method = "bilinear"
    # def generate_local_vol_surface(underyling):
    today = ql.Date(9, 12, 2025)

    ttf_forward_price = 20.0
    tenors = [0.25, 0.50, 0.75, 1.0]
    ttf_strikes = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    ttf_ivol_matrix = np.zeros((len(ttf_strikes), len(tenors)))
    ttf_ivol_matrix[0][0] = 0.35; ttf_ivol_matrix[0][1] = 0.30; ttf_ivol_matrix[0][2] = 0.25; ttf_ivol_matrix[0][3] = 0.20
    ttf_ivol_matrix[1][0] = 0.30; ttf_ivol_matrix[1][1] = 0.28; ttf_ivol_matrix[1][2] = 0.23; ttf_ivol_matrix[1][3] = 0.20
    ttf_ivol_matrix[2][0] = 0.28; ttf_ivol_matrix[2][1] = 0.25; ttf_ivol_matrix[2][2] = 0.22; ttf_ivol_matrix[2][3] = 0.20  # ATM Vol
    ttf_ivol_matrix[3][0] = 0.30; ttf_ivol_matrix[3][1] = 0.28; ttf_ivol_matrix[3][2] = 0.23; ttf_ivol_matrix[3][3] = 0.20
    ttf_ivol_matrix[4][0] = 0.35; ttf_ivol_matrix[4][1] = 0.30; ttf_ivol_matrix[4][2] = 0.25; ttf_ivol_matrix[4][3] = 0.20

    local_vol_surface_ttf = generate_local_vol_surface(tenors, ttf_strikes, ttf_ivol_matrix, interpolation_method,
                               today, ttf_forward_price, r=0.0, div_rate=0.0)
    # local_vol_surface_ttf, tenors_ttf, strikes_ttf, implied_vols_ttf = generate_local_vol_surface("TTF")

    brent_forward_price = 60.0
    tenors = [0.25, 0.50, 0.75, 1.0]
    brent_strikes = np.array([40.0, 50.0, 60.0, 70.0, 80.0])
    brent_ivol_matrix = np.zeros((len(brent_strikes), len(tenors)))
    brent_ivol_matrix[0][0] = 0.35; brent_ivol_matrix[0][1] = 0.30; brent_ivol_matrix[0][2] = 0.25; brent_ivol_matrix[0][3] = 0.20
    brent_ivol_matrix[1][0] = 0.30; brent_ivol_matrix[1][1] = 0.28; brent_ivol_matrix[1][2] = 0.23; brent_ivol_matrix[1][3] = 0.20
    brent_ivol_matrix[2][0] = 0.28; brent_ivol_matrix[2][1] = 0.25; brent_ivol_matrix[2][2] = 0.22; brent_ivol_matrix[2][3] = 0.20  # ATM Vol
    brent_ivol_matrix[3][0] = 0.30; brent_ivol_matrix[3][1] = 0.28; brent_ivol_matrix[3][2] = 0.23; brent_ivol_matrix[3][3] = 0.20
    brent_ivol_matrix[4][0] = 0.35; brent_ivol_matrix[4][1] = 0.30; brent_ivol_matrix[4][2] = 0.25; brent_ivol_matrix[4][3] = 0.20

    local_vol_surface_brent = generate_local_vol_surface(tenors, brent_strikes, brent_ivol_matrix, interpolation_method,
                                                       today, brent_forward_price, r=0.0, div_rate=0.0)


    # --- 2. DATA PREPARATION FOR PLOTTING ---
    if plot_VOL_GRAPHS:
        # 2.1. Prepare Grid for the Fitted Surface
        # Define a denser grid for a smooth visualization (e.g., 50x50 points)
        N_TIME = 50
        N_STRIKE = 50

        time_grid = np.linspace(min(tenors), max(tenors), N_TIME)
        strike_grid = np.linspace(min(ttf_strikes), max(ttf_strikes), N_STRIKE)

        T, K = np.meshgrid(time_grid, strike_grid)

        # 2.2. Calculate Fitted Volatilities (Z_fitted)
        Z_fitted = np.zeros_like(T)
        for i in range(N_STRIKE):
            for j in range(N_TIME):
                # Use the BlackVarianceSurface's blackVol method to get the interpolated value
                Z_fitted[i, j] = local_vol_surface_ttf.localVol(T[i, j], K[i, j], False)

        # 2.3. Prepare Market Data (Z_market)
        # Convert QuantLib Matrix back to a NumPy array for easy plotting
        Z_market = ttf_ivol_matrix

        # --- 3. VISUALIZATION ---

        fig = plt.figure(figsize=(15, 7))

        # --- Subplot 1: Market Data Points (Sparse) ---
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_wireframe(T, K, Z_fitted, color='lightgray', alpha=0.5)  # Background fitted surface
        # Scatter plot the original input points (Z_market)
        X_market, Y_market = np.meshgrid(tenors, ttf_strikes)
        ax1.scatter(X_market, Y_market, Z_market, color='red', marker='o', s=50, label='Market Data Points')

        ax1.set_title('Market Implied Volatility (Input Points)', fontsize=14)
        ax1.set_xlabel('Time to Expiry (Years)')
        ax1.set_ylabel('Strike Price (K)')
        ax1.set_zlabel('Implied Volatility')
        ax1.legend()

        # --- Subplot 2: Fitted Volatility Surface (Smooth) ---
        ax2 = fig.add_subplot(122, projection='3d')
        # Plot the interpolated surface using a color map
        surf = ax2.plot_surface(T, K, Z_fitted, cmap=plt.cm.viridis,
                                linewidth=0, antialiased=False, alpha=0.9)

        ax2.set_title(f'QuantLib Fitted Volatility Surface ({interpolation_method.upper()})', fontsize=14)
        ax2.set_xlabel('Time to Expiry (Years)')
        ax2.set_ylabel('Strike Price (K)')
        ax2.set_zlabel('Implied Volatility')
        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Volatility')

        plt.tight_layout()
        plt.show()

    # 3 Contract Specs & Market Parameters
    K = 2.0  # Spread option strike (TTF - Brent)
    T = 0.1  # Option maturity in years
    slope = 0.15
    r = 0.0  # Risk-free rate (assumed constant)
    historical_rho = 0.99  # Historical (or implied) correlation between TTF and Brent M12, TO BE CALIBRATED
    S0_TTF = 20.0  # Current TTF Forward Price (e.g., F(0, T))
    S0_Brent = 60.0  # Current Brent Forward Price (e.g., F(0, T))

    # Monte Carlo Parameters
    N_PATHS = 50000  # Number of simulation paths
    N_STEPS = int(252*T)  # Number of time steps (e.g., trading days)
    dt = 1.0 / 252  # Time step size


    # --- B. MONTE CARLO PRICER FUNCTION ---

    def price_spread_option_mc(lv_ttfa, lv_brenta, K, T, r, rho, S0_TTF, S0_Brent, N_PATHS, N_STEPS):

        # 1. Initialize Price Arrays
        # Initialize paths with starting forward prices
        S1_paths = np.full((N_PATHS, N_STEPS + 1), S0_TTF)
        S2_paths = np.full((N_PATHS, N_STEPS + 1), S0_Brent)

        # Time vector for path simulation
        t_values = np.linspace(0.0, T, N_STEPS + 1)

        # 2. Generate Correlated Random Numbers
        # Generate standard normal random variables
        Z1 = np.random.standard_normal((N_PATHS, N_STEPS))
        Z2_uncorr = np.random.standard_normal((N_PATHS, N_STEPS))

        # Apply Cholesky decomposition to get correlated variables
        # dW2 = rho*dW1 + sqrt(1-rho^2)*dZ2_uncorr
        Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2_uncorr

        # 3. Path Simulation (Euler Discretization)
        MAX_VOL = 0.5

        def get_ttf_vol(t, s):
            try:
                return lv_ttfa.localVol(t, s, False)
            except:
                # print(f"Max Vol capped TTF at {t} on {s}, vol = 0.5")
                return MAX_VOL

        def get_brent_vol(t, s):
            try:
                return lv_brenta.localVol(t, s, False)
            except:
                # print(f"Max Vol capped BRENT at {t} on {s}, vol = 0.5")
                return MAX_VOL

        for i in range(N_STEPS):
            if i%30 == 0:
                print(f"Step {i} out of {N_STEPS}")
            t = t_values[i]

            # Current asset prices (S_t)
            S1_t = S1_paths[:, i]
            S2_t = S2_paths[:, i]

            # Fetch Local Volatilities from Surfaces (using S_t as the strike argument)
            # Note: QuantLib requires a single call for each price/time pair;
            # using a simple loop structure for clarity, though vectorization is possible

            # Vectorized call to get Local Vols for all paths
            sigma1_t = np.array([get_ttf_vol(t, s1) for s1 in S1_t])
            sigma2_t = np.array([get_brent_vol(t, s2) for s2 in S2_t])

            # Wiener process increments
            dW1 = np.sqrt(dt) * Z1[:, i]
            dW2 = np.sqrt(dt) * Z2[:, i]

            # Euler Step for S1 (TTF)
            # dS/S = r*dt + sigma*dW -> S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*dW)
            S1_paths[:, i + 1] = S1_t * np.exp(
                (r - 0.5 * sigma1_t ** 2) * dt + sigma1_t * dW1
            )

            # Euler Step for S2 (Brent)
            S2_paths[:, i + 1] = S2_t * np.exp(
                (r - 0.5 * sigma2_t ** 2) * dt + sigma2_t * dW2
            )

        # 4. Calculate Payoffs

        # Final prices at maturity T
        S1_T = S1_paths[:, -1]
        S2_T = S2_paths[:, -1]

        # Payoff: max(S1_T - S2_T - K, 0)
        payoffs = np.maximum(S1_T - slope * S2_T - K, 0.0)

        # 5. Discount and Average
        discount_factor = np.exp(-r * T)
        price = discount_factor * np.mean(payoffs)

        # Calculate Standard Error
        std_err = discount_factor * np.std(payoffs) / np.sqrt(N_PATHS)

        return price, std_err


    # --- C. EXECUTION ---

    start_time = time.time()

    # Assuming lv_surface_TTF and lv_surface_Brent are defined and calibrated
    price, std_err = price_spread_option_mc(
        local_vol_surface_ttf, local_vol_surface_brent, K, T, r, historical_rho,
        S0_TTF, S0_Brent, N_PATHS, N_STEPS
    )

    end_time = time.time()

    ## Results
    print(f"--- MC Pricing Results ---")
    print(f"Asset 1 (TTF) Forward: {S0_TTF:.2f}")
    print(f"Asset 2 (Brent) Forward: {S0_Brent:.2f}")
    print(f"Strike K: {K:.2f}")
    print(f"Correlation (rho): {historical_rho}")
    print(f"Option Price (Call Spread): {price:.4f}")
    print(f"Monte Carlo Standard Error: +/- {std_err:.4f}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")