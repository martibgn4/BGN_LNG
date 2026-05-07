import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from xbbg import blp
import numpy as np

from Data.forward_curve_interpolator import ForwardCurve
from Utils.datetime_utils import date_tenor_from_label, parse_month, label_from_date_tenor, month_int_from_string, \
    month_string_from_int
from Utils.utils_spark import retrieve_credentials, get_access_token, fetch_cargo_prices, fetch_freight_prices, \
    list_contracts

dict_fx_to_tickers = {
    "GBPUSD": "GBP",
    "EURUSD": "EUR",
    "EURGBP": "EURGBP"
}
dict_comm_to_tickers = {
    "HenryHub": "NG",
    "HH": "NG",
    "JKM": "JKL",
    "TTF": ("TZT", "QZT", "QQT", "QTT"),
    "TFU": ("TMR", "TQR", "TSR", "TYR"),
    "Brent": "CO",
    "NBP": ("FN", "QR", "SA", "YAA"),
    "THE": ("NCG", "NCG", "NCG", "NCG"),
    "PEG": ("PNG", "PNG", "PNG", "PNG"),
    "PSV": ("PSR", "PSR", "PSR", "PSR"),
    "PVB": ("PXB", "PXB", "PXB", "PXB"),
    "VTP": ("CEG", "CEG", "CEG", "CEG"),
}

bbg_months_available_fx = ["SP", "1M", "2M", "3M", "4M", "5M", "6M", "9M", "1Y", "15M", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y"]
bbg_months_available_fx_int = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120]
dict_bbg_months_available_fx_to_int = {k: v for k, v in zip(bbg_months_available_fx, bbg_months_available_fx_int)}


bbg_month_codes = ["F", "G", "H", "J", "K", "M",
               "N", "Q", "U", "V", "X", "Z"]
bbg_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

bbg_dict = {m: c for m, c in zip(bbg_months, bbg_month_codes)}
bbg_dict_code_to_month = {c: m for m, c in zip(bbg_months, bbg_month_codes)}

add_suffix_underlyings = ["THE", "PEG", "PSV", "PVB", "VTP"]


class BloombergExtractor:
    def __init__(self, commodity, months=120, quarters=12, seasons=10, years=5,
                 start_year = 26, start_month="Feb"):
        if commodity in dict_fx_to_tickers:
            self.commodity_code = dict_fx_to_tickers[commodity]
            self.is_fx = True
        elif commodity in dict_comm_to_tickers:
            self.commodity_codes = dict_comm_to_tickers[commodity]
            self.q_ticker, self.s_ticker, self.y_ticker = None, None, None
            if isinstance(self.commodity_codes, tuple):
                self.commodity_code, self.q_ticker, self.s_ticker, self.y_ticker = self.commodity_codes
            else:
                self.commodity_code = self.commodity_codes
            self.is_fx = False
        else:
            raise ValueError(f"Invalid commodity {commodity}, not supported yet")
        self.start_year = start_year
        self.start_month = start_month

        self.months = months
        self.quarters = quarters
        self.seasons = seasons
        self.years = years
        self.commodity = commodity

    def get_monthly_tickers(self):
        if not self.is_fx:
            tickers = [self.commodity_code + code + str(year)
                       for year in range(self.start_year, self.start_year + self.years)
                       for code in bbg_month_codes]
            raw_tickers = [code + "_" + str(year)[-2:]
                           for year in range(self.start_year, self.start_year + self.years)
                           for code in bbg_months]
            if self.commodity in add_suffix_underlyings:
                # Add suffix to specific commodities only
                tickers = [t+"M" for t in tickers]
            return tickers, raw_tickers
        else:
            label_curncy = " BGN Curncy"
            tickers = [self.commodity_code + label_curncy]
            raw_tickers = [self.start_month + "_" + str(self.start_year)]
            current_month_int = month_int_from_string(self.start_month)
            for s_month, i_month in zip(bbg_months_available_fx[1:], bbg_months_available_fx_int[1:]):
                tickers.append(self.commodity_code + s_month + label_curncy)
                _month_int = current_month_int + i_month
                one_to_twelve_month, years_to_add = _month_int % 12, _month_int // 12
                if one_to_twelve_month == 0:
                    one_to_twelve_month = 12
                    years_to_add -= 1
                year = self.start_year + years_to_add
                month = month_string_from_int(one_to_twelve_month)
                raw_tickers.append(month + "_" + str(year)[-2:])
            return tickers, raw_tickers


    def get_quarterly_tickers(self):
        if self.q_ticker is None:
            return [], []
        q_codes = ["J", "N", "V", "F"]
        _q_codes = ["Q2", "Q3", "Q4", "Q1"]

        tickers = [self.q_ticker + code + str(year)
                   for year in range(self.start_year, self.start_year + self.years)
                   for code in q_codes]
        if self.commodity in add_suffix_underlyings:
            # Add suffix to specific commodities only
            tickers = [t+"Q" for t in tickers]

        raw_tickers = [_code + "_" + str(year)[-2:]
                   for year in range(self.start_year, self.start_year + self.years)
                   for _code in _q_codes]
        return tickers, raw_tickers

    def get_season_tickers(self):
        if self.s_ticker is None:
            return [], []
        q_codes = ["J", "V"]
        _q_codes = ["SS", "WS"]

        tickers = [self.s_ticker + code + str(year)
                   for year in range(self.start_year, self.start_year + self.years)
                   for code in q_codes]
        if self.commodity in add_suffix_underlyings:
            # Add suffix to specific commodities only
            tickers = [t+"S" for t in tickers]

        raw_tickers = [_code + "_" + str(year)[-2:]
                   for year in range(self.start_year, self.start_year + self.years)
                   for _code in _q_codes]

        return tickers, raw_tickers

    def get_yearly_tickers(self):
        if self.y_ticker is None:
            return [], []

        tickers = [
            self.y_ticker + "F" + str(year)
            for year in range(self.start_year, self.start_year + self.years)
        ]
        if self.commodity in add_suffix_underlyings:
            # Add suffix to specific commodities only
            tickers = [t+"Y" for t in tickers]

        raw_tickers = [
            "Y" + "_" + str(year)[-2:]
            for year in range(self.start_year, self.start_year + self.years)
        ]

        return tickers, raw_tickers

    def get_all_tickers(self):
        monthly_tickers, raw_monthly_tickers = self.get_monthly_tickers()
        quarterly_tickers, raw_quarterly_tickers = self.get_quarterly_tickers()
        season_tickers, raw_season_tickers = self.get_season_tickers()
        yearly_tickers, raw_yearly_tickers = self.get_yearly_tickers()
        bbg_tickers = monthly_tickers + quarterly_tickers + season_tickers + yearly_tickers
        raw_tickers = raw_monthly_tickers + raw_quarterly_tickers + raw_season_tickers + raw_yearly_tickers
        return bbg_tickers, raw_tickers

    def retrieve_latest_bbg_values(self, bbg_price_label="Last_Price"):
        if not self.is_fx:
            bbg_tickers, raw_tickers = self.get_all_tickers()
            _bbg_tickers = [t + " Comdty" for t in bbg_tickers]
            dict_tickers = {bbg_tick: raw_tick for bbg_tick, raw_tick in zip(_bbg_tickers, raw_tickers)}
            all_quotes = blp.bdp(
                tickers= _bbg_tickers,
                flds=[bbg_price_label, "volume", "time"]
            )

            all_quotes["raw_tickers"] = [dict_tickers[t] for t in all_quotes.index]

            if self.start_year < 15:
                # Adapt it to make it compatible across all underlyings
                def convert_to_2_digit_year(code):
                    month, year = code.split("_")
                    if int(year) < 6:
                        twoyear_digit = "3" + year
                    else:
                        twoyear_digit = "2" + year
                    return month + "_" + twoyear_digit
                all_quotes["raw_tickers"] = all_quotes["raw_tickers"].apply(convert_to_2_digit_year)
            all_quotes = all_quotes.set_index("raw_tickers")
            return all_quotes

        else:  # is_fx is True
            bbg_tickers, raw_tickers = self.get_monthly_tickers()
            dict_tickers = {bbg_tick: raw_tick for bbg_tick, raw_tick in zip(bbg_tickers, raw_tickers)}
            all_quotes = blp.bdp(
                tickers=bbg_tickers,
                flds=["px_last"]
            )

            all_quotes["raw_tickers"] = [dict_tickers[t] for t in all_quotes.index]
            all_quotes = all_quotes.set_index("raw_tickers")
            spot_price = all_quotes.loc[raw_tickers[0]].px_last
            all_quotes = spot_price + all_quotes/10000
            return all_quotes

def plot_hh_tfu_jkm(hide_show=False):
    tfu_quotes = BloombergExtractor(
        "TFU",
        start_year=26
    ).retrieve_latest_bbg_values()

    hh_quotes = BloombergExtractor(
        "HH",
        start_year=26
    ).retrieve_latest_bbg_values()

    jkm_quotes = BloombergExtractor(
        "JKM",
        start_year=26
    ).retrieve_latest_bbg_values()

    common_tenors = [t for t in hh_quotes.index if t in tfu_quotes.index]

    date_tenors = [date_tenor_from_label(d) for d in common_tenors]

    us_nwe_spread = tfu_quotes["last_price"] - 1.15 * hh_quotes["last_price"]
    jkm_tfu_spread = jkm_quotes["last_price"] - tfu_quotes["last_price"]

    us_nwe_spread = us_nwe_spread[common_tenors]
    us_nwe_spread.index = date_tenors

    jkm_tfu_spread = jkm_tfu_spread[common_tenors]
    jkm_tfu_spread.index = date_tenors

    # Plot
    plt.figure(figsize=(14, 9))

    # Base line
    plt.plot(us_nwe_spread.index, us_nwe_spread.values, color="tab:blue", linewidth=2, label="TTF - 115% HH")
    plt.plot(jkm_tfu_spread.index, jkm_tfu_spread.values, color="tab:orange", linewidth=2, label="JKM - TTF")

    # Formatting to match style
    plt.title("Spread Forward Curves", fontsize=26, loc='left', pad=25)
    plt.xlabel("Tenor", fontsize=26)
    plt.xticks(fontsize=26)
    plt.ylabel("USD/MMBtu", fontsize=26)
    plt.yticks(fontsize=26)

    # plt.xticks(years)
    plt.legend(loc="upper right", fontsize=26)
    plt.grid(True)

    plt.tight_layout()
    if not hide_show:
        plt.show()

def plot_ttf_and_eu_spreads(years, today_date, plot_spreads=True, hide_show=False):
    # today_date = date(2026, 1, 26)
    bbg_price_label="px_settle"
    ttf_quotes = BloombergExtractor(
        "TTF", start_year=26, years=years
    ).retrieve_latest_bbg_values(bbg_price_label="px_settle")

    dict_tenor_to_price_ttf = {m: ttf_quotes.loc[m][bbg_price_label] for m in ttf_quotes.index}
    ttf_curve = ForwardCurve(dict_tenor_to_price_ttf, today_date)
    ttf_monthly_quotes = ttf_curve.monthly_quotes

    # Plot
    fig = plt.figure(figsize=(14, 9))
    ax = plt.subplot(111)
    # plt.figure(figsize=(6, 3))
    for c in ["NBP", "THE", "PEG", "PSV", "PVB", "VTP"]:
        _start_year = 26 if c not in add_suffix_underlyings else 6
        c_quotes = BloombergExtractor(commodity=c, start_year=_start_year,years=years).retrieve_latest_bbg_values(bbg_price_label)

        dict_tenor_to_price_c = {m: c_quotes.loc[m][bbg_price_label] for m in c_quotes.index}
        c_curve = ForwardCurve(dict_tenor_to_price_c, today_date)
        c_monthly_quotes = c_curve.monthly_quotes

        common_tenors = [t for t in ttf_monthly_quotes if t in c_monthly_quotes]

        ttf_c_spread = {}
        ttf_nbp_threshold = {}
        if c != "NBP":
            for monthly_t in common_tenors:
                if plot_spreads:
                    ttf_c_spread[monthly_t] = ttf_curve.get_value_on_month(monthly_t) - c_curve.get_value_on_month(monthly_t)
                else:
                    ttf_c_spread[monthly_t] = c_curve.get_value_on_month(monthly_t)
        else:
            eurgbp_quotes = BloombergExtractor(commodity="EURGBP", start_year=_start_year, years=years).retrieve_latest_bbg_values()
            dict_tenor_to_price_eurgbp = {m: eurgbp_quotes.loc[m]["px_last"] for m in eurgbp_quotes.index}
            eurgbp_curve = ForwardCurve(dict_tenor_to_price_eurgbp, today_date)

            for monthly_t in common_tenors:
                eurgbp_rate = eurgbp_curve.get_value_on_month(monthly_t)
                nbp_transformed = c_curve.get_value_on_month(monthly_t) * 0.01 / eurgbp_rate / 0.0293001
                if plot_spreads:
                    ttf_c_spread[monthly_t] = ttf_curve.get_value_on_month(monthly_t) - nbp_transformed
                else:
                    ttf_c_spread[monthly_t] = nbp_transformed
                    ttf_nbp_threshold[monthly_t] = nbp_transformed*0.985 - 1.24  # TTF limit from below for buying TTF from UK

        dates = [date_tenor_from_label(k) for k in ttf_c_spread.keys()]
        # Plot TTF-c spread
        label = "TTF - " + c if plot_spreads else c
        ax.plot(dates, ttf_c_spread.values(), linewidth=3, label=label)
        if c == "NBP":
            ax.plot(dates, ttf_nbp_threshold.values(), linewidth=3, label="98.5%NBP - 1.24")

    # Formatting to match style
    if not plot_spreads:
        # Plot TTF separately
        ttf_quotes = {}
        for monthly_t in ttf_monthly_quotes.keys():
            ttf_quotes[monthly_t] = ttf_curve.get_value_on_month(monthly_t)
        dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
        ax.plot(dates, ttf_quotes.values(), linewidth=5, label="TTF")

    if plot_spreads:
        title = f"European Spreads as of COB {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "
    else:
        title = f"European curves as of COB {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "
    plt.title(title, fontsize=26, loc='left', pad=25)
    plt.xlabel("Tenor", fontsize=26)
    plt.xticks(rotation=30, fontsize=26)
    plt.ylabel("EUR/MWh", fontsize=26)
    plt.yticks(fontsize=26)

    # plt.xticks(years)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 1, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=26)
    plt.grid(True)

    plt.tight_layout()
    if not hide_show:
        plt.show()


def plot_nwe_vs_nbp(years, today_date, hide_show=False):
    bbg_price_label="px_settle"
    ttf_quotes = BloombergExtractor(
        "TTF", start_year=26, years=years
    ).retrieve_latest_bbg_values(bbg_price_label="px_settle")

    dict_tenor_to_price_ttf = {m: ttf_quotes.loc[m][bbg_price_label] for m in ttf_quotes.index}
    ttf_curve = ForwardCurve(dict_tenor_to_price_ttf, today_date)
    ttf_monthly_quotes = ttf_curve.monthly_quotes

    nwe_discounts = extract_spark_quotes(quote_type='NWEDiscountsFinancial', latest_only=True, limit=90, cal_month=None, print_available_contacts=True)
    nwe_discounts["Period Start"] = nwe_discounts["Period Start"].apply(label_from_date_tenor)
    nwe_discounts.set_index("Period Start", inplace=True)
    common_indeces = [i for i in nwe_discounts.index if i in ttf_quotes.index]
    dict_tenor_to_price_discounts = {m: nwe_discounts.loc[m]["Price"] for m in common_indeces}
    nwediscounts_curve = ForwardCurve(dict_tenor_to_price_discounts, today_date)

    eurusd_quotes = BloombergExtractor(commodity="EURUSD", start_year=26,
                                       years=years).retrieve_latest_bbg_values("")
    dict_tenor_to_price_eurusd = {m: eurusd_quotes.loc[m]["px_last"] for m in eurusd_quotes.index}
    eurusd_curve = ForwardCurve(dict_tenor_to_price_eurusd, today_date)

    # Plot
    ax = plt.subplot(111)
    for c in ["NBP"]:
        _start_year = 26 if c not in add_suffix_underlyings else 6
        c_quotes = BloombergExtractor(commodity=c, start_year=_start_year,years=years).retrieve_latest_bbg_values(bbg_price_label)

        dict_tenor_to_price_c = {m: c_quotes.loc[m][bbg_price_label] for m in c_quotes.index}
        c_curve = ForwardCurve(dict_tenor_to_price_c, today_date)
        c_monthly_quotes = c_curve.monthly_quotes

        common_tenors = [t for t in ttf_monthly_quotes if t in c_monthly_quotes]

        ttf_nbp_threshold = {}
        gbpusd_quotes = BloombergExtractor(commodity="GBPUSD", start_year=_start_year, years=years).retrieve_latest_bbg_values("")
        dict_tenor_to_price_gbpusd = {m: gbpusd_quotes.loc[m]["px_last"] for m in gbpusd_quotes.index}
        gbpusd_curve = ForwardCurve(dict_tenor_to_price_gbpusd, today_date)

        for monthly_t in common_tenors:
            gbpusd_rate = gbpusd_curve.get_value_on_month(monthly_t)
            eurusd_rate = eurusd_curve.get_value_on_month(monthly_t)
            eurgbp_rate = gbpusd_rate / eurusd_rate
            nbp_transformed = c_curve.get_value_on_month(monthly_t) * 0.01 * eurgbp_rate / 0.0293001
            eurusd = eurusd_curve.get_value_on_month(monthly_t)
            ttf_nbp_threshold[monthly_t] = nbp_transformed*0.985 - 0.45 / eurusd * 3.412142   # TTF limit from below for buying TTF from UK
            # ttf_nbp_threshold[monthly_t] = nbp_transformed

        dates = [date_tenor_from_label(k) for k in ttf_nbp_threshold.keys()]
        # Plot TTF-c spread

        ax.plot(dates, ttf_nbp_threshold.values(), linewidth=2, label="98.5%NBP - 0.45 $/mmbtu")

    # Plot TTF separately
    ttf_quotes = {}
    for monthly_t in ttf_monthly_quotes.keys():
        ttf_quote = ttf_curve.get_value_on_month(monthly_t)
        nwe_discount = nwediscounts_curve.get_value_on_month(monthly_t) / eurusd_curve.get_value_on_month(monthly_t) * 3.412142
        ttf_quotes[monthly_t] = ttf_quote + nwe_discount

    dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
    ax.plot(dates, ttf_quotes.values(), linewidth=2, label="NWE")

    title = f"NWE vs NBP. COB: {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "
    plt.title(title, fontsize=16, loc='left', pad=25)
    plt.xlabel("Tenor", fontsize=16)
    plt.xticks(rotation=30, fontsize=16)
    plt.ylabel("EUR/MWh", fontsize=16)
    plt.yticks(fontsize=16)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    if not hide_show:
        plt.show()


def extract_spark_quotes(quote_type="Cargo", latest_only=True, limit=90, cal_month="Apr-2026", print_available_contacts=False):
    client_id, client_secret = retrieve_credentials(file_path="..\\adhoc_scripts\\client_credentials.csv")
    access_token = get_access_token(client_id, client_secret)
    if print_available_contacts:
        print(list_contracts(access_token))

    if quote_type == "Cargo":
        full_df = fetch_cargo_prices(access_token, 'sparknwe', limit, latest_only=latest_only, cal_month=cal_month)
    elif quote_type == "NWEDiscountsFinancial":
        full_df = fetch_cargo_prices(access_token, 'sparknwe-fin-monthly', limit, latest_only=latest_only, cal_month=cal_month)
    elif quote_type == "FreightSpark30":
        full_df = fetch_freight_prices(access_token, 'spark30fo', limit, my_vessel='174-2stroke', latest_only=latest_only, cal_month=cal_month)
    elif quote_type == "FreightSpark30Spot":
        full_df = fetch_freight_prices(access_token, 'spark30s', limit, my_vessel='174-2stroke', latest_only=latest_only, cal_month=None)
    elif quote_type == "FreightSpark30FFA":
        full_df = fetch_freight_prices(access_token, 'spark30ffa-monthly', limit, my_vessel='174-2stroke', latest_only=latest_only, cal_month=cal_month)
    else:
        raise ValueError(f"Spark type {quote_type} not recognized")
    return full_df


def get_historical_prices_for_underlyings(underlyings, start_date, end_date, price_label="px_last"):
    df_h = blp.bdh(
        underlyings,
        price_label,
        start_date,
        end_date,
    )
    return df_h


def plot_nicely_historical_prices(ttf_tick, jkm_tick, hh_tick, month, brent_tick=None, back_periods=180, hide_show=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime


    # 2. GENERATE SAMPLE DATA (JKM, TTF, Henry Hub)
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings([ttf_tick, jkm_tick, hh_tick, brent_tick], dates[0], dates[-1])
    df.dropna(inplace=True)

    df['JKM-TTF'] = df[jkm_tick] - df[ttf_tick]
    df['JKM-HH'] = df[jkm_tick] - df[hh_tick]

    # 3. CREATE THE PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})

    # --- TOP PANEL: ABSOLUTE PRICES ---
    ax1.plot(df.index, df[jkm_tick], color='#00D4FF', linewidth=2.5, label='JKM')
    ax1.plot(df.index, df[ttf_tick], color='#FF0055', linewidth=2.5, label='TTF')
    ax1.plot(df.index, df[hh_tick], color='#FFFFFF', linewidth=1.5, linestyle='--', alpha=0.7, label='Henry Hub')

    # Fill the gap between JKM and TTF to highlight the East/West spread visually
    # ax1.fill_between(df.index, df[jkm_tick].values, df[hh_tick].values, color='#00D4FF', alpha=0.05)

    ax1.set_ylabel(r'$USD/MMBtu$', fontsize=20)
    # ax1.set_yticklabels([3.5 + i*0.5 for i in range(1, 15)])
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.set_title(str(back_periods) + " days " + r' $\mathbf{HISTORICAL\ PERFORMANCE\ \ GAS \ BENCHMARK: } $' + month, loc='left', fontsize=20, pad=25, color='white')
    ax1.grid(True, which='major', linestyle=':', alpha=1.0)

    ax1_right = ax1.twinx()
    ax1_right.plot(df.index, df[brent_tick], color='#FFB800', linewidth=1.5, label='Brent')
    ax1_right.set_ylabel(r'USD/bbl', fontsize=20)
    ax1_right.yaxis.set_tick_params(labelsize=20)

    # ax1.legend(loc='best', frameon=True, facecolor='#0D1117', edgecolor='#30363D', fontsize=20)

    legend_elements = [
        Line2D([0], [0], color='#00D4FF', lw=4, label=f'JKM'),
        Line2D([0], [0], color='#FF0055', lw=4, label=f'TTF'),
        Line2D([0], [0], color='#FFFFFF', lw=4, linestyle='--', alpha=0.7, label=f'HH'),
        Line2D([0], [0], color='#FFB800', lw=4, label='Brent')
    ]
    ax1.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=18)

    # --- BOTTOM PANEL: SPREADS (THE "ARB") ---
    ax2.fill_between(df.index, df['JKM-TTF'], 0, where=(df['JKM-TTF'] >= 0), color='#00D4FF', alpha=0.3, label='JKM-TTF')
    ax2.plot(df.index, df['JKM-TTF'], color='#00D4FF', linewidth=1, alpha=0.8)

    ax2.fill_between(df.index, df['JKM-HH'], 0, color='#FFB800', alpha=0.1, label='JKM-HH')
    ax2.plot(df.index, df['JKM-HH'], color='#FFB800', linewidth=1, alpha=0.6)

    ax2.axhline(0, color='white', linewidth=0.5)
    ax2.set_ylabel(r'$Spread\ (USD/MMBtu)$', fontsize=20)
    ax2.set_title(r'$\mathbf{REGIONAL\ ARBITRAGE\ WINDOWS}$', loc='left', fontsize=14, color='#8B949E')
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=20)
    ax2.grid(True, which='major', linestyle=':', alpha=1.0)
    ax2.xaxis.set_tick_params(labelsize=20, rotation=15)
    ax2.yaxis.set_tick_params(labelsize=20)

    # 4. FINAL FORMATTING
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    # plt.figtext(0.1, 0.02, 'Data: Market Feeds | BGN LNG Desk Analytics', fontsize=14, color='#484F58')
    # plt.figtext(0.9, 0.02, 'CONFIDENTIAL', fontsize=10, color='#F85149', weight='bold', ha='right')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # plt.savefig('lng_spreads.png', dpi=300)
    if not hide_show:
        plt.show()


def plot_spark_quotes(hide_show=False):
    df_nwe_discounts = extract_spark_quotes(quote_type="Cargo", latest_only=False, limit=2, cal_month=None)
    df_spark30_phys = extract_spark_quotes(quote_type="FreightSpark30", latest_only=False, limit=2, cal_month=None)
    df_spark30_ffa = extract_spark_quotes(quote_type="FreightSpark30FFA", latest_only=False, limit=2, cal_month=None)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    # 2. DATA PREPARATION
    months = [m for m in reversed(sorted(df_nwe_discounts["Period Start"].unique()))]
    n_months = len(months)

    nwe_days = sorted(df_nwe_discounts["Release Date"].unique())[0:2]
    spark30_phys_days = sorted(df_spark30_phys["Release Date"].unique())[0:2]
    spark30_ffa_days = sorted(df_spark30_ffa["Release Date"].unique())[0:2]

    def get_price_for_month_and_date_on_df(_month, _r_date, _df, is_freight=False):
        filtered_dates = _df[_df["Release Date"]==str(_r_date.date())]
        month_price = filtered_dates[filtered_dates["Period Start"] == _month]
        if not is_freight:
            return month_price.Price.values[0] if len(month_price) > 0 else 0
        else:
            return month_price.USDperday.values[0]/1000 if len(month_price) > 0 else 0

    nwe_prev = [get_price_for_month_and_date_on_df(m, nwe_days[0], df_nwe_discounts) for m in months]
    nwe_curr = [get_price_for_month_and_date_on_df(m, nwe_days[1], df_nwe_discounts) for m in months]

    phys_prev = [get_price_for_month_and_date_on_df(m, spark30_phys_days[0], df_spark30_phys, is_freight=True) for m in months]
    phys_curr = [get_price_for_month_and_date_on_df(m, spark30_phys_days[1], df_spark30_phys, is_freight=True) for m in months]

    ffa_prev = [get_price_for_month_and_date_on_df(m, spark30_ffa_days[0], df_spark30_ffa, is_freight=True) for m in months]
    ffa_curr = [get_price_for_month_and_date_on_df(m, spark30_ffa_days[1], df_spark30_ffa, is_freight=True) for m in months]


    data = {
        'Month': [datetime.strptime(m, '%Y-%m-%d').strftime("%b-%Y") for m in months],
        'NWE_Curr': nwe_curr,
        'NWE_Prev': nwe_prev,
        'Phys_Curr': phys_curr,
        'Phys_Prev': phys_prev,
        'FFA_Curr': ffa_curr,
        'FFA_Prev': ffa_prev,
    }
    df = pd.DataFrame(data)

    # 3. PLOT CONSTRUCTION
    fig, ax_right = plt.subplots(figsize=(14, 9))
    ax_left = ax_right.twiny()  # Create the dual axis for the top/left side

    y = np.arange(n_months)
    alpha = 0.5
    height = 0.25  # width of individual bars
    right_height = 0.125
    # --- RIGHT SIDE: FREIGHT (FFA & Physical) ---
    # FFA Freight (Top of the group)
    ax_right.barh(y + height, df['FFA_Curr'], right_height, color='#00D4FF', label='FFA Current', edgecolor='white', linewidth=0.3)
    ax_right.barh(y + right_height, df['FFA_Prev'], right_height, color='#00D4FF', alpha=alpha, label='FFA Prev')

    # Physical Freight (Bottom of the group)
    ax_right.barh(y - right_height, df['Phys_Curr'], right_height, color='#3399FF', label='Phys Current', edgecolor='white', linewidth=0.3)
    ax_right.barh(y - height, df['Phys_Prev'], right_height, color='#3399FF', alpha=alpha, label='Phys Prev')

    # --- LEFT SIDE: NWE DISCOUNT ---
    # We use negative values to push the bars to the left
    ax_left.barh(y+ right_height, df['NWE_Curr'], height, color='#FF0055', label='NWE Current', edgecolor='white', linewidth=0.3)
    ax_left.barh(y-right_height, df['NWE_Prev'], height, color='#FF0055', alpha=alpha, label='NWE Prev')

    # 4. SCALE ALIGNMENT (Crucial Step)
    ax_right.set_xlim(-38, 114)
    ax_left.set_xlim(-1.5, 4.5)

    # 5. FORMATTING
    ax_right.axvline(0, color='white', linewidth=1.5)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(df['Month'], fontsize=12, fontweight='bold')

    # Labels and Ticks
    ax_right.set_xlabel(f'Freight Rates (kUSD/day): Latest on {str(spark30_phys_days[-1].date())} (Spark30 174k)', color='#3399FF', loc='right', fontweight='bold', fontsize=20)
    ax_left.set_xlabel(f'NWE Discount (USD/MMBtu): Latest on {str(nwe_days[-1].date())} (Spark)', color='#FF0055', loc='left', fontweight='bold', fontsize=20)

    # Clean up ticks to only show positive values
    ax_right.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    ax_right.xaxis.set_tick_params(labelsize=20)
    ax_right.yaxis.set_tick_params(labelsize=20)
    ax_right.grid(True, which='major', linestyle=':', alpha=1.0)
    ax_left.set_xticks([0, -0.5, -1.0, -1.5])
    ax_left.xaxis.set_tick_params(labelsize=20)
    ax_left.yaxis.set_tick_params(labelsize=20)
    ax_left.grid(True, which='major', linestyle=':', alpha=1.0)
    # ax_left.set_xticklabels(['0', '-0.5', '1.0', '1.5'])

    # ax_right.set_title('BGN LNG Desk: NWE Discounts & Freight Dashboard from SPARK', loc='left', fontsize=20, pad=50)

    # Branding / Legend
    legend_elements = [
        Line2D([0], [0], color='#00D4FF', lw=6, label=f'FFA Freight (kUSD/day)'),
        Line2D([0], [0], color='#3399FF', lw=6, label=f'Phys Freight (kUSD/day)'),
        Line2D([0], [0], color='#FF0055', lw=6, label=f'NWE Discount (USD/MMBtu)'),
        Line2D([0], [0], color='gray', alpha=alpha, lw=6, label='Prev Day Value')
    ]
    ax_right.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=20)

    plt.tight_layout()
    if not hide_show:
        plt.show()


def create_bgn_lng_report_grid(report_date):
    import io
    from PIL import Image
    def capture_function_plot():
        # func()
        active_fig = plt.gcf()
        buf = io.BytesIO()
        active_fig.savefig(buf, format='png', bbox_inches='tight', dpi=450)
        plt.close(active_fig)
        buf.seek(0)
        return Image.open(buf)
    # Create the container figure (The Report Canvas)
    # We turn off the 'constrained_layout' here because we are placing images, not plots
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.facecolor": "#0D1117",
        "axes.linewidth": "3",
        "axes.edgecolor": "#30363D",
        "figure.facecolor": "#0D1117",
        "grid.color": "#30363D",
        "grid.linewidth": "3",
        "text.color": "#C9D1D9",
        "axes.labelcolor": "#8B949E",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'BGN LNG Desk: {report_date.strftime('%d-%b-%Y')} Daily Report ', fontsize=20, weight='bold')

    # Flatten axes for easy iteration (0,0 -> 0,1 -> 1,0 -> 1,1)
    ax_flat = axes.flatten()

    plot_spark_quotes(hide_show=True)
    img = capture_function_plot()
    ax_flat[0].imshow(img)
    ax_flat[0].axis('off')

    plot_nicely_historical_prices(
        ttf_tick="TYRF27 Comdty", jkm_tick="AJKMY 27 BCFV Index", hh_tick="FFHHY 27 Index", brent_tick="FSBTY 27 Index",
        month = "Cal27 Contracts",
        back_periods=180,
        hide_show=True
    )
    img = capture_function_plot()
    ax_flat[1].imshow(img)
    ax_flat[1].axis('off')

    pricing_date = datetime.now().date()
    plot_ttf_and_eu_spreads(years=3, today_date=pricing_date, plot_spreads=False, hide_show=True)
    img = capture_function_plot()
    ax_flat[2].imshow(img)
    ax_flat[2].axis('off')

    plot_hh_tfu_jkm(hide_show=True)
    img = capture_function_plot()
    ax_flat[3].imshow(img)
    ax_flat[3].axis('off')


    plt.tight_layout()
    fig.savefig('final_stitched_report.png', dpi=450)
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    plot_nwe_vs_nbp(2, pricing_date, hide_show=False)
    plt.show()


def violin_plots_ttf_nbp_spread(start_date, back_periods=400):
    import seaborn as sns

    periods_historical = back_periods
    initial_date = start_date

    dates = pd.date_range(end=initial_date, periods=periods_historical)

    y = initial_date.year % 2000  # - 20
    bbg_tenor_codes = ["J" + str(y), "K" + str(y), "M" + str(y), "N" + str(y), "Q" + str(y), "U" + str(y), "V" + str(y), "X" + str(y), "Z" + str(y),
                       "F" + str(y + 1), "G" + str(y + 1), "H" + str(y + 1)]


    ttf_codes = ["CO" + t + " Comdty" for t in bbg_tenor_codes]
    # nbp_codes = ["FN" + t + " Comdty" for t in bbg_tenor_codes]
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

    # tenors_fx, raw_tenors_fx = BloombergExtractor("EURGBP", start_year=26, start_month="Feb").get_monthly_tickers()

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


if __name__ == "__main__":

    from datetime import datetime, timedelta, date
    import pandas as pd
    pricing_date = datetime.now().date()
    # create_bgn_lng_report_grid(pricing_date)




    # fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    # plot_nwe_vs_nbp(4, pricing_date, hide_show=False)
    # plt.show()

    # start_date = datetime(2026, 2, 25)
    # violin_plots_ttf_nbp_spread(start_date, back_periods=800)
    #

    ticker = "NG"

    df = get_historical_prices_for_underlyings(
        [ticker+str(i)+ " Comdty" for i in range(1, 36)],
        start_date=date(2022,1,1),
        end_date=date(2026,5,4),
        price_label=["px_settle", "px_open", "px_close", "px_high", "px_low", "volume", "open_int"]
    )

    roll_dates = blp.bdp(
        tickers=[ticker + str(i) + str(year)[-2:] + " Comdty" for i in bbg_month_codes for year in range(2022, 2030)],
        flds="last_tradeable_dt"
    ).sort_values(by="last_tradeable_dt")

    def get_roll_date(current_date):
        idx = np.where(roll_dates >=current_date)[0][0]
        return roll_dates.iloc[idx].last_tradeable_dt


    # Apply to the index
    df['roll_date'] = df.index.map(get_roll_date)

    a = 1
    #
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda x: '/'.join([str(i) for i in x]))

    df.to_csv("C:\\Marti\\HH_daily_OCHL_data.csv")

    # a =1
    # import pandas as pd
    #
    # big_df = pd.DataFrame()
    #
    #
    # for i in range(1, 37):
    #
    #     intraday = blp.bdib(
    #         ticker=f"NG{i} Comdty",
    #         start_datetime='2024-01-01 09:30:00',
    #         end_datetime='2026-04-21 17:30:00',
    #         interval=30,
    #         # ref ="FuturesEuropeICE",
    #         ref="NYME"
    #     )
    #     if isinstance(intraday.columns, pd.MultiIndex):
    #         intraday.columns = intraday.columns.map(lambda x: '/'.join([str(i) for i in x]))
    #     # intraday.to_csv("C:\\Marti\\TTF_intraday_data.csv")
    #
    #     big_df = pd.concat([big_df, intraday], axis=1)
    #
    # big_df.to_csv("C:\\Marti\\HH_intraday_data.csv")


    # intraday_tick_data = blp.bdtick(
    #     ticker="TZT1 Comdty",
    #     dt="2026-04-14",
    #     time_range=('09:30','17:30'),
    #     interval=15,
    #     ref ="FuturesFinancialsICE",
    #     types= ["TRADE", "AT_TRADE", "BID", "ASK", "MID_PRICE", "BID_BEST", "ASK_BEST", "BEST_BID", "BEST_ASK"]
    # )

    a = 1
