import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

if __name__ == "__main__":
    plot_VOL_GRAPHS = False

    # 1. Setup (Dates, Rates, Spot)
    interpolation_method = "bilinear"
    def generate_local_vol_surface(underyling):
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        day_counter = ql.Actual365Fixed()
        todays_date = ql.Date(9, 12, 2025)
        ql.Settings.instance().evaluationDate = todays_date

        spot = 100.0
        risk_free_rate = 0.02
        dividend_rate = 0.00

        # Term Structures for Discounting
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(todays_date, risk_free_rate, day_counter)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(todays_date, dividend_rate, day_counter)
        )
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        # 2. Market Data (Implied Volatility Surface)
        # Maturities in years
        tenors = [0.25, 0.50, 1.0, 2.0]
        expiries = [todays_date + ql.Period(int(t * 365.25), ql.Days) for t in tenors]

        # Strikes
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

        # Implied Volatility Matrix (Example Data)
        # Rows: Strikes, Columns: Maturities (T1, T2, T3, T4)
        implied_vols = ql.Matrix(len(strikes), len(expiries))
        implied_vols[0][0] = 0.35; implied_vols[0][1] = 0.30; implied_vols[0][2] = 0.25; implied_vols[0][3] = 0.20
        implied_vols[1][0] = 0.30; implied_vols[1][1] = 0.28; implied_vols[1][2] = 0.23; implied_vols[1][3] = 0.20
        implied_vols[2][0] = 0.28; implied_vols[2][1] = 0.25; implied_vols[2][2] = 0.22; implied_vols[2][3] = 0.20 # ATM Vol
        implied_vols[3][0] = 0.30; implied_vols[3][1] = 0.28; implied_vols[3][2] = 0.23; implied_vols[3][3] = 0.20
        implied_vols[4][0] = 0.35; implied_vols[4][1] = 0.30; implied_vols[4][2] = 0.25; implied_vols[4][3] = 0.20

        implied_vols_mat = np.array([[implied_vols[i][j] for j in range(len(expiries))] for i in range(len(strikes))])
        # 3. Construct the Black Variance Surface (Input)
        black_var_surface = ql.BlackVarianceSurface(
            todays_date,
            calendar,
            expiries,
            strikes.tolist(),
            implied_vols,
            day_counter
        )

        # Optional: Set interpolation for smoother derivatives (crucial step)

        # interpolation_method = "bilinear"
        black_var_surface.setInterpolation(interpolation_method)
        print("Interpolation set to 'bilinear' successfully.")

        # 4. Construct the Local Volatility Surface (Output)
        local_vol_surface = ql.LocalVolSurface(
            ql.BlackVolTermStructureHandle(black_var_surface),
            flat_ts,
            dividend_ts,
            spot_handle
        )
        local_vol_surface.enableExtrapolation()

        # 5. Extract a Local Volatility Value
        time_to_expiry = tenors[2] # 1.0 year
        strike_price = 95.0

        # The localVol method applies the Dupire formula to the interpolated surface
        local_vol = local_vol_surface.localVol(time_to_expiry, strike_price, True)

        print(f"Local Volatility at T={time_to_expiry} and K={strike_price}: {local_vol:.4f}")

        return local_vol_surface, tenors, strikes, implied_vols_mat

    local_vol_surface_ttf, tenors_ttf, strikes_ttf, implied_vols_ttf = generate_local_vol_surface("TTF")
    local_vol_surface_brent, tenors_brent, strikes_brent, implied_vols_brent = generate_local_vol_surface("BRENT")


    # --- 2. DATA PREPARATION FOR PLOTTING ---
    if plot_VOL_GRAPHS:
        # 2.1. Prepare Grid for the Fitted Surface
        # Define a denser grid for a smooth visualization (e.g., 50x50 points)
        N_TIME = 50
        N_STRIKE = 50

        time_grid = np.linspace(min(tenors_ttf), max(tenors_ttf), N_TIME)
        strike_grid = np.linspace(min(strikes_ttf), max(strikes_ttf), N_STRIKE)

        T, K = np.meshgrid(time_grid, strike_grid)

        # 2.2. Calculate Fitted Volatilities (Z_fitted)
        Z_fitted = np.zeros_like(T)
        for i in range(N_STRIKE):
            for j in range(N_TIME):
                # Use the BlackVarianceSurface's blackVol method to get the interpolated value
                Z_fitted[i, j] = local_vol_surface_ttf.localVol(T[i, j], K[i, j], False)

        # 2.3. Prepare Market Data (Z_market)
        # Convert QuantLib Matrix back to a NumPy array for easy plotting
        Z_market = implied_vols_ttf

        # --- 3. VISUALIZATION ---

        fig = plt.figure(figsize=(15, 7))

        # --- Subplot 1: Market Data Points (Sparse) ---
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_wireframe(T, K, Z_fitted, color='lightgray', alpha=0.5)  # Background fitted surface
        # Scatter plot the original input points (Z_market)
        X_market, Y_market = np.meshgrid(tenors_ttf, strikes_ttf)
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
    K = 20.0  # Spread option strike (TTF - Brent)
    T = 0.1  # Option maturity in years
    r = 0.02  # Risk-free rate (assumed constant)
    historical_rho = 0.99  # Historical (or implied) correlation between TTF and Brent M12, TO BE CALIBRATED
    S0_TTF = 100.0  # Current TTF Forward Price (e.g., F(0, T))
    S0_Brent = 80.0  # Current Brent Forward Price (e.g., F(0, T))

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
        payoffs = np.maximum(S1_T - S2_T - K, 0.0)

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