
import numpy as np
import pandas as pd
from datetime import date

def simulate_bounded_forward_curve(
    pricing_date: date,
    simulation_dates: list,
    F0: pd.Series,
    mu: float,
    sigma: float,
    alpha: float,
    # n_steps: int = 252,
    n_paths: int = 1000,
    lower_bound: float = 0.5,
    upper_bound: float = 10.0,
    seed: int = 42,
    is_spread=False
):
    rng = np.random.default_rng(seed)

    tenors = list(F0.index)
    n_ten = len(tenors)

    # Initial log prices
    if is_spread:
        Y0 = F0.values
    else:
        Y0 = np.log(F0.values.astype(float))

    n_steps = len(simulation_dates)+1
    # Pre-allocate arrays
    Y_paths = np.zeros((n_paths, n_steps, n_ten))
    F_paths = np.zeros((n_paths, n_steps, n_ten))
    Y_paths[:, 0, :] = Y0
    F_paths[:, 0, :] = F0.values

    def get_maturities_on_date(_tenors, _date):
        maturities = [(_d - _date).days/365.0 for _d in _tenors ]
        return maturities

    tau_vec = np.array(get_maturities_on_date(tenors, pricing_date), dtype=float)
    front_loadings = sigma * np.exp(-alpha * tau_vec)  # per tenor

    prev_sim_date = pricing_date
    for t in range(1, n_steps):
        # Common factor shocks shared across tenors
        Z = rng.standard_normal(size=(n_paths, 1))       # independent normals
        # E = Z @ chol.T                                   # correlated (eps1, eps2)
        # eps1 = E[:, 0]                                   # level shock
        # eps2 = E[:, 1]                                   # front shock

        # # Optional idiosyncratic shocks per tenor/path
        # if eta_idio > 0.0:
        #     eps_idio = rng.standard_normal(size=(n_paths, n_ten))
        # else:
        #     eps_idio = 0.0

        # dY for each tenor: broadcast eps1, eps2 across tenors
        # dY = sqrt_dt * (mu * eps1 + front_loadings * eps2) + sqrt_dt * eta_idio * eps_idio
        sim_date = simulation_dates[t-1]
        dt = (sim_date - prev_sim_date).days/365.0
        sqrt_dt = np.sqrt(dt)

        v = 0.5*(mu + front_loadings)

        a = -0.5*(v**2)

        if is_spread:
            dY = v * Z * sqrt_dt
        else:
            dY = a * dt + v * Z * sqrt_dt
        # if isinstance(eps_idio, np.ndarray):
        #     dY += eta_idio * sqrt_dt * eps_idio
        future_tenors = tau_vec > 0

        # Update log prices and map to prices
        Y_paths[:, t, future_tenors] = Y_paths[:, t - 1, future_tenors] + dY[:, future_tenors]
        if is_spread:
            F_paths[:, t, future_tenors] = Y_paths[:, t, future_tenors]
        else:
            F_paths[:, t, future_tenors] = np.exp(Y_paths[:, t, future_tenors])
        F_paths[:, t, ~future_tenors] = F_paths[:, t-1, ~future_tenors]
        # Enforce hard bounds
        F_paths[:, t, future_tenors] = np.clip(F_paths[:, t, future_tenors], lower_bound, upper_bound)


        tau_vec = np.array(get_maturities_on_date(tenors, sim_date), dtype=float)
        front_loadings = sigma * np.exp(-alpha * tau_vec)  # per tenor
        prev_sim_date = sim_date

    return F_paths



_dict_fitted_params = {
    "HH": (0.2988,0.5883,2.0000), # (mu, sigma, alpha)
    "TFU_HH": (0.4099,0.3352,0.5000),  # (mu, sigma, alpha)
    "JKM_HH": (0.7767,0.0766,2.0000),  # (mu, sigma, alpha)
}

_dict_fitted_params_normal = {
    "HH": (1.2590,1.4887,2.0000), # (mu, sigma, alpha)
    "TFU_HH": (2.0279,4.5812,0.5000),  # (mu, sigma, alpha)
    "JKM_HH": (6.1970,2.1198,2.0000),  # (mu, sigma, alpha)
    "JKM_TFU": (4.7782,0.0000,0.5000)  # (mu, sigma, alpha)
}

_dict_bounds = {
    "HH": (0.5,10.0), # (lower_bound, upper_bound)
    "TFU_HH": (-2.0, 10.0),  # (lower_bound, upper_bound)
    "JKM_HH": (-1.0, 14.0),  # (lower_bound, upper_bound)
    "JKM_TFU": (-2.0, 5.0),  # (lower_bound, upper_bound)
}

xl_file_path = "C:\\Users\\marti.fernandezreal\\OneDrive - BAYEGAN DIS TIC. A.S\\Marti\\BloombergData\\bbg_data_extractor.xlsx"

def read_df_from_xl_database(und):
    df = pd.read_excel(
            xl_file_path,
            sheet_name=und+"_BBG",
            usecols="A:Y",
            skiprows=1,
            index_col=0)
    df.index.name = None
    df.index = [i.date() for i in df.index]
    return df

def get_df_for_und(und):
    if und.split("_")[0] == und:  # Single underlying, go ahead
        df = read_df_from_xl_database(und)
    else:
        und1, und2 = und.split("_")
        df1 = read_df_from_xl_database(und1)
        df2 = read_df_from_xl_database(und2)
        common_idx = df1.index.intersection(df2.index)
        common_cols = df1.columns.intersection(df2.columns)

        df = df1.loc[common_idx, common_cols] - df2.loc[common_idx, common_cols]
    return df

def read_forward_curve_on_date(und, _day):
    df = get_df_for_und(und)
    try:
        curve_prices = df.loc[_day]
    except KeyError:
        raise KeyError(f"Day {_day} not available in xl database for {und}")
    return curve_prices

def get_tenor_date_from_month_code_and_pricing_date(_pricing_date, m_code):
    months_int = int(m_code[1:])
    final_month = _pricing_date.month + months_int
    final_year = final_month//12
    final_month = final_month%12
    if final_month==0:
        final_year = final_year-1
        final_month = 12
    return date(_pricing_date.year +final_year,
                final_month,
                _pricing_date.day)


if __name__ == "__main__":
    pricing_date = date(2025, 12, 12)
    und = "JKM_TFU"

    F0 = read_forward_curve_on_date(und, pricing_date)
    F0.index = [
        get_tenor_date_from_month_code_and_pricing_date(pricing_date, i)
        for i in F0.index
    ]
    simulation_dates = [date(2026, i, 15) for i in range(1, 13)]

    # F0 = pd.Series({d: 5.0 for d in simulation_dates})

    is_spread = "_" in und

    if is_spread:
        mu, sigma, alpha = _dict_fitted_params_normal[und]
    else:
        mu, sigma, alpha = _dict_fitted_params[und]

    n_paths = 1000
    lower_bound, upper_bound = _dict_bounds[und]
    F_simulated = simulate_bounded_forward_curve(
        pricing_date,
        simulation_dates,
        F0,
        mu, sigma, alpha,
        n_paths,
        lower_bound,
        upper_bound,
        seed = 42,
        is_spread=True
    )

    import matplotlib.pyplot as plt

    n_rows=3
    n_cols=4
    # Create the subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, constrained_layout=True)
    for i, expiry in enumerate(simulation_dates):
        ax=axes[i//n_cols,i%n_cols]
        ax.hist(F_simulated[:, -1, i])
        ax.set_title(f"Expiry {expiry}")
        ax.set_xlabel("Price [$ / MMbtu]")
        ax.set_ylabel("Frequency")
    plt.show()

    print("Succeeded")
