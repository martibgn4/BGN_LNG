import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SpreadOptionPricer:
    def __init__(self, ttf_series, brent_series, eurusd_series, expiry_years, risk_free_rate_eur):
        """
        Inputs:
        - ttf_series: pd.Series of historical TTF Forward prices (EUR/MWh)
        - brent_series: pd.Series of historical Brent Forward prices (USD/bbl)
        - eurusd_series: pd.Series of historical EURUSD rates (USD per EUR)
        - expiry_years: Time to maturity in years (e.g., 1.5)
        - risk_free_rate_eur: EUR risk-free rate for discounting
        """
        self.ttf = ttf_series
        self.brent = brent_series
        self.fx = eurusd_series
        self.T = expiry_years
        self.r = risk_free_rate_eur

        # Energy Conversion Factor: MWh per bbl
        # 1 bbl of oil is approx 1.7 MWh (thermal).
        # Adjust this based on specific contract specs (e.g. if it's a heat-rate spread).
        self.conversion_factor = 1.7

        self._calibrate()

    def _calibrate(self):
        """
        Calibrate Volatilities and Correlation Matrix from historical log returns.
        """
        # 1. Combine into a DataFrame
        df = pd.DataFrame({
            'Gas': self.ttf,
            'Oil': self.brent,
            'FX': self.fx
        }).dropna()

        # 2. Calculate Log Returns
        log_rets = np.log(df / df.shift(1)).dropna()

        # 3. Annualized Volatility (assuming 252 trading days)
        self.vol = log_rets.std() * np.sqrt(252)

        # 4. Correlation Matrix
        self.corr_matrix = log_rets.corr()

        print("--- Calibration Results ---")
        print(f"Volatilities:\n{self.vol}\n")
        print(f"Correlation Matrix:\n{self.corr_matrix}\n")

    def price_monte_carlo(self, K, num_sims=100000):
        """
        Price the spread option using Monte Carlo with correlated GBM.
        Payoff = max(Gas_EUR - (Oil_USD / FX_EURUSD * Conversion) - K, 0)
        """
        dt = self.T
        n_assets = 3

        # Current Start Prices
        S0_gas = self.ttf.iloc[-1]
        S0_oil = self.brent.iloc[-1]
        S0_fx = self.fx.iloc[-1]

        # Cholesky Decomposition for Correlated Random Normals
        L = np.linalg.cholesky(self.corr_matrix.values)

        # Generate Uncorrelated Random Normals (Z)
        Z_uncorrelated = np.random.normal(0, 1, (n_assets, num_sims))

        # Generate Correlated Random Normals (W)
        # W = L * Z
        W_correlated = np.dot(L, Z_uncorrelated)

        # Extract individual Brownian motions
        W_gas = W_correlated[0, :]
        W_oil = W_correlated[1, :]
        W_fx = W_correlated[2, :]

        # --- Simulate Prices at Expiry T ---
        # Note: Forwards drift is 0 (Martingale), FX drift is neglected here for simplicity
        # or assumed embedded in forward FX if available. We use standard Black-76 dynamics.

        # Gas Forward Path
        # F_T = F_0 * exp(-0.5*sigma^2*T + sigma*sqrt(T)*Z)
        F_T_gas = S0_gas * np.exp(-0.5 * self.vol['Gas'] ** 2 * dt +
                                  self.vol['Gas'] * np.sqrt(dt) * W_gas)

        # Oil Forward Path
        F_T_oil = S0_oil * np.exp(-0.5 * self.vol['Oil'] ** 2 * dt +
                                  self.vol['Oil'] * np.sqrt(dt) * W_oil)

        # FX Path
        # Assuming drift = 0 for simplicity (or modeling Forward FX directly)
        F_T_fx = S0_fx * np.exp(-0.5 * self.vol['FX'] ** 2 * dt +
                                self.vol['FX'] * np.sqrt(dt) * W_fx)

        # --- Calculate Payoff ---
        # Convert Oil to EUR: (USD/bbl) / (USD/EUR) = EUR/bbl
        oil_price_eur = F_T_oil / F_T_fx

        # Convert EUR/bbl to EUR/MWh using conversion factor
        oil_price_eur_mwh = oil_price_eur * self.conversion_factor

        # Spread Payoff
        payoffs = np.maximum(F_T_gas - oil_price_eur_mwh - K, 0)

        # Discount back to present (using EUR rate)
        discount_factor = np.exp(-self.r * self.T)
        price = discount_factor * np.mean(payoffs)

        # Standard Error calculation to judge convergence
        std_error = discount_factor * np.std(payoffs) / np.sqrt(num_sims)

        return price, std_error


# ==========================================
# 4. Usage Simulation
# ==========================================

# Generate Synthetic Data for demonstration
# In reality, you would load your CSVs here
np.random.seed(42)
days = 252
dates = pd.date_range(start='2024-01-01', periods=days, freq='B')

# Synthetic Trends
# Gas: High vol (60%), starting 40 EUR/MWh
gas_path = 40 * np.cumprod(np.exp(0.6 * np.sqrt(1 / 252) * np.random.normal(0, 1, days) - 0.5 * 0.6 ** 2 / 252))
# Oil: Medium vol (30%), starting 80 USD/bbl
oil_path = 80 * np.cumprod(np.exp(0.3 * np.sqrt(1 / 252) * np.random.normal(0, 1, days) - 0.5 * 0.3 ** 2 / 252))
# FX: Low vol (10%), starting 1.10
fx_path = 1.1 * np.cumprod(np.exp(0.1 * np.sqrt(1 / 252) * np.random.normal(0, 1, days) - 0.5 * 0.1 ** 2 / 252))

# Create Series
ttf_fwd = pd.Series(gas_path, index=dates)
brent_fwd = pd.Series(oil_path, index=dates)
eurusd = pd.Series(fx_path, index=dates)

# Initialize Pricer
# Expiry June 2026 is approx 1.5 years away
pricer = SpreadOptionPricer(ttf_fwd, brent_fwd, eurusd, expiry_years=1.5, risk_free_rate_eur=0.03)

# Define Strike K (Spread Strike in EUR/MWh)
# E.g. We want to protect if Gas becomes 5 EUR/MWh more expensive than Oil equivalent
strike = 5.0

price, error = pricer.price_monte_carlo(K=strike)

print(f"--- Pricing Result ---")
print(f"Strike: {strike} EUR/MWh")
print(f"Option Price: {price:.4f} EUR")
print(f"Std Error: {error:.4f}")