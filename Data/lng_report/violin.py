"""Standalone violin plot of the historical monthly TTF-NBP spread distribution."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Data.forward_curve_interpolator import ForwardCurve
from Utils.datetime_utils import month_string_from_int

from Data.lng_report.config import bbg_dict_code_to_month
from Data.lng_report.bloomberg import (
    BloombergExtractor,
    get_historical_prices_for_underlyings,
)


def violin_plots_ttf_nbp_spread(start_date, back_periods=400):
    periods_historical = back_periods
    initial_date = start_date

    dates = pd.date_range(end=initial_date, periods=periods_historical)

    y = initial_date.year % 2000  # - 20
    bbg_tenor_codes = ["J" + str(y), "K" + str(y), "M" + str(y), "N" + str(y), "Q" + str(y), "U" + str(y),
                       "V" + str(y), "X" + str(y), "Z" + str(y),
                       "F" + str(y + 1), "G" + str(y + 1), "H" + str(y + 1)]

    ttf_codes = ["CO" + t + " Comdty" for t in bbg_tenor_codes]
    nbp_codes = []
    nbp_codes_clean = ["FN" + t for t in bbg_tenor_codes]
    fx_codes = ["EURGBP BGN Curncy"] + ["EURGBP" + str(m) + "M BGN Curncy" for m in range(1, 20)]

    df = get_historical_prices_for_underlyings(
        ttf_codes + nbp_codes + fx_codes,
        dates[0],
        dates[-1],
        price_label="px_last"
    )

    new_columns = [c[0].replace(" Comdty", "").replace(" BGN Curncy", "") for c in df.columns]
    df.columns = new_columns
    df.dropna(inplace=True)

    for c in df.columns:
        if c[-1] == "M":  # If its FX tenor, add spot to it due to its FX format from bbg
            df[c] = df[c] / 10000 + df["EURGBP"]

    def convert_nbp_from_full_row(row):
        current_date = row.name
        month_code = month_string_from_int(current_date.month)
        year_code = current_date.year % 2000
        tenors_fx, raw_tenors_fx = BloombergExtractor("EURGBP", start_year=year_code,
                                                      start_month=month_code).get_monthly_tickers()

        tenors_fx = [t.replace(" BGN Curncy", "") for t in tenors_fx]
        dict_tenors_fx_to_raw_tenors_fx = dict(zip(tenors_fx, raw_tenors_fx))
        tenors_in_df = [t for t in df.columns if t in tenors_fx]
        dict_tenor_to_price = {dict_tenors_fx_to_raw_tenors_fx[t]: row[t] for t in tenors_in_df}
        eurgbp_curve = ForwardCurve(dict_tenor_to_price, current_date)
        nbp_transformed = []
        for _nbp_tenor in nbp_codes_clean:
            month_code = _nbp_tenor[-3]
            year_code = _nbp_tenor[-2:]
            current_tenor = bbg_dict_code_to_month[month_code] + "_" + year_code
            nbp_in_eurmwh = row[_nbp_tenor] / eurgbp_curve.get_value_on_month(current_tenor) * 0.01 / 0.0293001
            nbp_transformed.append(nbp_in_eurmwh)
        return pd.Series(nbp_transformed)

    df[nbp_codes_clean] = df.apply(convert_nbp_from_full_row, axis=1)

    spread_codes = ["TTFNBP_Spread_" + bbg_dict_code_to_month[bbg_code[0]] + "_" + bbg_code[-2:] for bbg_code in
                    bbg_tenor_codes]

    for spread_code, bbg_code in zip(spread_codes, bbg_tenor_codes):
        df[spread_code] = df["TZT" + bbg_code] - df["FN" + bbg_code]

    df_to_plot = df[spread_codes]
    df_to_plot.columns = [c.replace("TTFNBP_Spread_", "") for c in df_to_plot.columns]
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=df_to_plot, cut=0)
    last_values = df_to_plot.loc[dates[-1].date()]
    sns.stripplot(x=last_values.keys(), y=last_values.values,
                  edgecolor='black', linewidth=1, s=10, ax=ax)
    plt.xticks(rotation=30)
    plt.ylabel("EUR/MWh", fontsize=14)
    plt.xlabel("")
    plt.title(f"TTF-NBP spread from {dates[0].date()} to {dates[-1].date()}")

    plt.show()
