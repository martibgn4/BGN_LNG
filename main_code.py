from Pricers.european_option import black76_price, black76_greeks
from Data.vol_surface import VolSurface

# Example forward and market data
F_TTF = 40.0  # EUR/MWh
K = 48.0
T = 0.5  # 6 months
r = 0.02

# Example volatility surface (strike vs tenor)
vol_surface = {
    35: {0.25: 0.45, 0.5: 0.48, 1.0: 0.50},
    38: {0.25: 0.47, 0.5: 0.50, 1.0: 0.52},
    40: {0.25: 0.49, 0.5: 0.52, 1.0: 0.54}
}

# Interpolate volatility for given strike and tenor
sigma = VolSurface(vol_surface).vol(strike=K, tenor=T)

# Price vanilla option
call_price = black76_price(F_TTF, K, sigma, T, r, option_type="call")
put_price = black76_price(F_TTF, K, sigma, T, r, option_type="put")

# Compute Greeks
call_delta = black76_greeks(F_TTF, K, sigma, T, r, "delta", option_type="call")
call_gamma = black76_greeks(F_TTF, K, sigma, T, r, "gamma", option_type="call")
call_vega = black76_greeks(F_TTF, K, sigma, T, r, "vega", option_type="call")
call_theta = black76_greeks(F_TTF, K, sigma, T, r, "theta", option_type="call")

print("Interpolated Volatility:", sigma)
print("Vanilla Call Price:", call_price)
print("Vanilla Put Price:", put_price)
print("Call Delta:", call_delta)
print("Call Gamma:", call_gamma)
print("Call Vega:", call_vega)
print("Call Theta", call_theta)
