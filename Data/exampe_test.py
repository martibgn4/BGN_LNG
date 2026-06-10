# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xbbg import blp
import numpy as np

from Data.bloomberg_bsrch import get_bsrch_lng_figs
from forward_curve_interpolator import ForwardCurve
from Utils.datetime_utils import date_tenor_from_label, parse_month, label_from_date_tenor, month_int_from_string, \
    month_string_from_int
from Utils.utils_spark import retrieve_credentials, get_access_token, fetch_cargo_prices, fetch_freight_prices, \
    list_contracts


def _dark_layout(**overrides):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="sans-serif", color="#C9D1D9", size=14),
        xaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
    )
    layout.update(overrides)
    return layout


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
    "BLNG1": "IKA",
    "BLNG2": "IKD",
    "BLNG3": "IKI"
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

def plot_hh_tfu_jkm():
    tfu_quotes = BloombergExtractor(
        "TFU",
        start_year=26
    ).retrieve_latest_bbg_values("px_settle")

    hh_quotes = BloombergExtractor(
        "HH",
        start_year=26
    ).retrieve_latest_bbg_values("px_settle")

    jkm_quotes = BloombergExtractor(
        "JKM",
        start_year=26
    ).retrieve_latest_bbg_values()

    jkm_settles = pd.read_csv("C:\\Marti\\ClaudeProjects\\data\\raw\\jkm_settlements\\master.csv")
    last_date = jkm_settles.settlement_date.max()
    last_date_as_date = datetime.strptime(last_date, "%Y-%m-%d").date()

    settleimplied_tfu_date = datetime.strptime(tfu_quotes.time.mode().values[0], "%m/%d/%Y").date()

    if last_date_as_date != settleimplied_tfu_date:
        raise ValueError(f"Date mismatch when reading TFU and JKM settlements! Got TTF as {settleimplied_tfu_date} and JKM as {last_date_as_date}")

    jkm_last_settles = jkm_settles[jkm_settles.settlement_date == last_date]
    jkm_last_settles.strip_label = jkm_last_settles.strip_label.apply(lambda x: x[:3] + "_" + x[-2:])

    common_tenors = [t for t in hh_quotes.index if t in tfu_quotes.index]

    # date_tenors = [date_tenor_from_label(d) for d in common_tenors]

    us_nwe_spread = tfu_quotes["px_settle"] - 1.15 * hh_quotes["px_settle"]

    tfu_tenors = [t for t in tfu_quotes.index]
    jkm_tenors = [t for t in jkm_last_settles.strip_label]
    common_tenors_tfu_jkm = [t for t in tfu_tenors if t in jkm_tenors]

    spread_dict = {}
    for t in common_tenors_tfu_jkm:
        tfu_settle = tfu_quotes.loc[t].px_settle
        jkm_settle = jkm_last_settles[jkm_last_settles.strip_label == t].settlement.values[0]
        spread_dict[t] = jkm_settle - tfu_settle

    jkm_tfu_spread = pd.Series(spread_dict)

    common_tenors = [t for t in common_tenors if t in common_tenors_tfu_jkm]
    date_tenors = [date_tenor_from_label(d) for d in common_tenors]

    us_nwe_spread = us_nwe_spread[common_tenors]
    us_nwe_spread.index = date_tenors

    jkm_tfu_spread = jkm_tfu_spread[common_tenors]
    jkm_tfu_spread.index = date_tenors

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(us_nwe_spread.index), y=list(us_nwe_spread.values),
                             mode='lines', name='TTF - 115% HH',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=list(jkm_tfu_spread.index), y=list(jkm_tfu_spread.values),
                             mode='lines', name='JKM - TTF',
                             line=dict(color='#ff7f0e', width=2)))
    fig.update_layout(**_dark_layout(), height=500,
                      title=dict(text="Spread Forward Curves", x=0.01),
                      xaxis_title="Tenor", yaxis_title="USD/MMBtu",
                      legend=dict(x=0.75, y=0.98))
    return fig

def plot_ttf_and_eu_spreads(years, today_date, plot_spreads=True):
    # today_date = date(2026, 1, 26)
    bbg_price_label="px_settle"
    ttf_quotes = BloombergExtractor(
        "TTF", start_year=26, years=years
    ).retrieve_latest_bbg_values(bbg_price_label="px_settle")

    dict_tenor_to_price_ttf = {m: ttf_quotes.loc[m][bbg_price_label] for m in ttf_quotes.index}
    ttf_curve = ForwardCurve(dict_tenor_to_price_ttf, today_date)


    ttf_monthly_quotes = ttf_curve.monthly_quotes

    # # Plot
    # fig = plt.figure(figsize=(14, 9))
    # ax = plt.subplot(111)
    fig = go.Figure()
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
                    ttf_nbp_threshold[monthly_t] = nbp_transformed*0.985 - 1.24

        dates = [date_tenor_from_label(k) for k in ttf_c_spread.keys()]
        # Plot TTF-c spread
        label = "TTF - " + c if plot_spreads else c
        # ax.plot(dates, ttf_c_spread.values(), linewidth=3, label=label)
        fig.add_trace(go.Scatter(x=dates, y=list(ttf_c_spread.values()),
                                 mode='lines', name=label, line=dict(width=3)))
        if c == "NBP" and ttf_nbp_threshold:
            # ax.plot(dates, ttf_nbp_threshold.values(), linewidth=3, label="98.5%NBP - 1.24")
            fig.add_trace(go.Scatter(x=dates, y=list(ttf_nbp_threshold.values()),
                                     mode='lines', name="98.5%NBP - 1.24", line=dict(width=3)))

    # Formatting to match style
    if not plot_spreads:
        # Plot TTF separately
        ttf_quotes = {}
        for monthly_t in ttf_monthly_quotes.keys():
            ttf_quotes[monthly_t] = ttf_curve.get_value_on_month(monthly_t)
        dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
        # ax.plot(dates, ttf_quotes.values(), linewidth=5, label="TTF")
        fig.add_trace(go.Scatter(x=dates, y=list(ttf_quotes.values()),
                                 mode='lines', name='TTF', line=dict(width=5)))

        nwe_discounts = extract_spark_quotes(quote_type='NWEDiscountsFinancial', latest_only=True, cal_month=None)
        nwe_discounts["Period Start"] = nwe_discounts["Period Start"].apply(label_from_date_tenor)
        nwe_discounts.set_index("Period Start", inplace=True)
        common_indeces = [i for i in nwe_discounts.index if i in ttf_quotes.keys()]
        dict_tenor_to_price_discounts = {m: nwe_discounts.loc[m]["Price"] for m in common_indeces}
        nwediscounts_curve = ForwardCurve(dict_tenor_to_price_discounts, today_date)

        eurusd_quotes = BloombergExtractor(commodity="EURUSD", start_year=26,
                                           years=years).retrieve_latest_bbg_values("")
        dict_tenor_to_price_eurusd = {m: eurusd_quotes.loc[m]["px_last"] for m in eurusd_quotes.index}
        eurusd_curve = ForwardCurve(dict_tenor_to_price_eurusd, today_date)

        nwe_quotes = {}
        for monthly_t in ttf_monthly_quotes.keys():
            nwe_discount = nwediscounts_curve.get_value_on_month(monthly_t) / eurusd_curve.get_value_on_month(monthly_t) * 3.412142
            nwe_quotes[monthly_t] = ttf_curve.get_value_on_month(monthly_t) + nwe_discount

        fig.add_trace(go.Scatter(x=dates, y=list(nwe_quotes.values()),
                                 mode='lines', name='NWE', line=dict(width=5)))

    if plot_spreads:
        title = f"European Spreads as of COB {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "
    else:
        title = f"European curves as of COB {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "

    fig.update_layout(**_dark_layout(), height=500,
                      title=dict(text=title, x=0.01),
                      xaxis_title="Tenor", yaxis_title="EUR/MWh",
                      legend=dict(x=1.02, y=0.5, xanchor='left'))
    return fig


def plot_nwe_vs_nbp(years, today_date):
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

    # # Plot
    # ax = plt.subplot(111)
    fig = go.Figure()
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
        # ax.plot(dates, ttf_nbp_threshold.values(), linewidth=2, label="98.5%NBP - 0.45 $/mmbtu")  # PLOTLY
        fig.add_trace(go.Scatter(x=dates, y=list(ttf_nbp_threshold.values()),                        #
                                 mode='lines', name="98.5%NBP - 0.45 $/mmbtu", line=dict(width=2)))

    # Plot TTF separately
    ttf_quotes = {}
    for monthly_t in ttf_monthly_quotes.keys():
        ttf_quote = ttf_curve.get_value_on_month(monthly_t)
        nwe_discount = nwediscounts_curve.get_value_on_month(monthly_t) / eurusd_curve.get_value_on_month(monthly_t) * 3.412142
        ttf_quotes[monthly_t] = ttf_quote + nwe_discount

    dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
    # ax.plot(dates, ttf_quotes.values(), linewidth=2, label="NWE")
    fig.add_trace(go.Scatter(x=dates, y=list(ttf_quotes.values()),
                             mode='lines', name="NWE", line=dict(width=2)))

    title = f"NWE vs NBP. COB: {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "
    # plt.title(title, fontsize=16, loc='left', pad=25)
    # plt.xlabel("Tenor", fontsize=16)
    # plt.xticks(rotation=30, fontsize=16)
    # plt.ylabel("EUR/MWh", fontsize=16)
    # plt.yticks(fontsize=16)
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # if not hide_show:
    #     plt.show()

    fig.update_layout(**_dark_layout(), height=400,
                      title=dict(text=title, x=0.01),
                      xaxis_title="Tenor", yaxis_title="EUR/MWh",
                      legend=dict(x=1.02, y=0.5, xanchor='left'))
    return fig


def extract_spark_quotes(quote_type="Cargo", latest_only=True, limit=90, cal_month="Apr-2026", print_available_contacts=False):
    client_id, client_secret = retrieve_credentials(file_path="C:\\Marti\\python_tests_2\\BGN_LNG\\adhoc_scripts\\client_credentials.csv")
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


def plot_nicely_historical_prices(ttf_tick, jkm_tick, hh_tick, month, brent_tick=None, back_periods=180):
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd

    # 2. GENERATE SAMPLE DATA (JKM, TTF, Henry Hub)
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings([ttf_tick, jkm_tick, hh_tick, brent_tick], dates[0], dates[-1])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)

    df['JKM-TTF'] = df[jkm_tick] - df[ttf_tick]
    df['JKM-HH'] = df[jkm_tick] - df[hh_tick]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.08,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                        subplot_titles=[
                            f"{back_periods} days <b>HISTORICAL PERFORMANCE GAS BENCHMARK:</b> {month}",
                            "<b>REGIONAL ARBITRAGE WINDOWS</b>"])
    fig.add_trace(go.Scatter(x=df.index, y=df[jkm_tick], name='JKM',
                             line=dict(color='#00D4FF', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[ttf_tick], name='TTF',
                             line=dict(color='#FF0055', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[hh_tick], name='HH',
                             line=dict(color='#FFFFFF', width=1.5, dash='dash'),
                             opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[brent_tick], name='Brent',
                             line=dict(color='#FFB800', width=1.5)),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['JKM-TTF'], name='JKM-TTF',
                             line=dict(color='#00D4FF', width=1), opacity=0.8,
                             fill='tozeroy', fillcolor='rgba(0,212,255,0.3)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['JKM-HH'], name='JKM-HH',
                             line=dict(color='#FFB800', width=1), opacity=0.6,
                             fill='tozeroy', fillcolor='rgba(255,184,0,0.1)'), row=2, col=1)
    fig.add_hline(y=0, line_color='white', line_width=0.5, row=2, col=1)
    fig.update_layout(**_dark_layout(), height=500)
    fig.update_yaxes(title_text="USD/MMBtu", row=1, col=1)
    fig.update_yaxes(title_text="USD/bbl", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread (USD/MMBtu)", row=2, col=1)
    fig.update_xaxes(tickformat='%d %b %Y', row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(color='#C9D1D9', size=14)
    return fig


def plot_nicely_baltic_freight(blng1_ticker, blng2_ticker, blng3_ticker, spark30_ticker, back_periods=180):
    from datetime import datetime
    import pandas as pd
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings(
        [blng1_ticker, blng2_ticker, blng3_ticker, spark30_ticker],
        dates[0], dates[-1]
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    # df.dropna(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[blng1_ticker], name='BLNG1 - 174k Australia to Japan',
                             line=dict(color='#00D4FF', width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[blng2_ticker], name='BLNG2- 174k USGC to EU',
                             line=dict(color='#FF0055', width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[blng3_ticker], name='BLNG3- 174k USGC to Japan',
                             line=dict(color='#FFB800', width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[spark30_ticker], name='Spark30- 174k USGC to EU',
                             line=dict(color='#15B01A', width=2.5, dash='dash')))
    fig.update_layout(**_dark_layout(), height=500,
                      title=dict(text="LNG Freight History", x=0.01),
                      xaxis_title="Day", yaxis_title="USD/day",
                      legend=dict(x=0.0, y=0.98))
    return fig


def plot_forwards_baltic_freight(capped_tenors=24):
    blng1_quotes = BloombergExtractor(
        "BLNG1",
        start_year=26
    ).retrieve_latest_bbg_values()

    blng2_quotes = BloombergExtractor(
        "BLNG2",
        start_year=26
    ).retrieve_latest_bbg_values()

    blng3_quotes = BloombergExtractor(
        "BLNG3",
        start_year=26
    ).retrieve_latest_bbg_values()

    for df in [blng1_quotes, blng2_quotes, blng3_quotes]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(blng1_quotes.index)[:capped_tenors],
                             y=list(blng1_quotes.last_price.values)[:capped_tenors],
                             mode='lines', name='BLNG1',
                             line=dict(color='#00D4FF', width=2)))
    fig.add_trace(go.Scatter(x=list(blng2_quotes.index)[:capped_tenors],
                             y=list(blng2_quotes.last_price.values)[:capped_tenors],
                             mode='lines', name='BLNG2',
                             line=dict(color='#FF0055', width=2)))
    fig.add_trace(go.Scatter(x=list(blng3_quotes.index)[:capped_tenors],
                             y=list(blng3_quotes.last_price.values)[:capped_tenors],
                             mode='lines', name='BLNG3',
                             line=dict(color='#FFB800', width=2)))
    fig.update_layout(**_dark_layout(), height=500,
                      title=dict(text="Baltic LNG Forward Curves", x=0.01),
                      xaxis_title="Tenor", yaxis_title="USD/day",
                      legend=dict(x=0.75, y=0.98))
    return fig


def plot_nicely_lng_on_water_curves(lng_on_water_20d_count, lng_on_water_30d_count, lng_on_water_20d_vol, lng_on_water_30d_vol, back_periods=180):
    from datetime import datetime
    import pandas as pd
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings(
        [lng_on_water_20d_count, lng_on_water_30d_count, lng_on_water_20d_vol, lng_on_water_30d_vol],
        dates[0], dates[-1]
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df.index, y=df[lng_on_water_20d_count], name='LNG on Water 20D <br> (Vessel Count)',
                             line=dict(color='#00D4FF', width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[lng_on_water_30d_count], name='LNG on Water 30D <br> (Vessel Count)',
                             line=dict(color='#FF0055', width=2.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[lng_on_water_20d_vol], name='LNG on Water 20D <br> (Volume)',
                             line=dict(color='#FFB800', width=2.5, dash='dash')),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df[lng_on_water_30d_vol], name='LNG on Water 30D <br> (Volume)',
                             line=dict(color='#15B01A', width=2.5, dash='dash')),
                  secondary_y=True)
    fig.update_layout(**_dark_layout(), height=500,
                      title=dict(text=f"LNG on Water as of {datetime.today().date().strftime('%d-%b-%Y')}: >20 vs >30 days", x=0.01),
                      xaxis_title="Day", yaxis_title="USD/day",
                      legend=dict(x=1.1, y=0.98))
    fig.update_yaxes(title_text="Vessel Count")
    fig.update_yaxes(title_text="Volume [Metric Tonnes]", secondary_y=True)

    return fig


def plot_spark_quotes():
    df_nwe_discounts = extract_spark_quotes(quote_type="Cargo", latest_only=False, limit=2, cal_month=None)
    df_spark30_phys = extract_spark_quotes(quote_type="FreightSpark30", latest_only=False, limit=2, cal_month=None)
    df_spark30_ffa = extract_spark_quotes(quote_type="FreightSpark30FFA", latest_only=False, limit=2, cal_month=None)

    # import matplotlib.pyplot as plt
    # from matplotlib.lines import Line2D
    import pandas as pd

    # 2. DATA PREPARATION
    months = [m for m in reversed(sorted(df_nwe_discounts["Period Start"].unique()))]
    n_months = len(months)

    latest_nwe_date = max(df_nwe_discounts["Release Date"]).date()
    latest_spark30_date = max(df_spark30_phys["Release Date"]).date()

    def get_curr_and_prev_for_month_on_df(_month, _df, is_freight=False):
        month_prices = _df[_df["Period Start"] == _month]
        data_available = month_prices.shape[0]
        if data_available >=2:
            if not is_freight:
                curr_price = month_prices.Price.values[0]
                prev_price = month_prices.Price.values[1]
            else:
                curr_price = month_prices.USDperday.values[0]/1000
                prev_price = month_prices.USDperday.values[1]/1000
        else:
            if data_available == 0:
                curr_price, prev_price = 0, 0
            else: # data_available = 1
                year_data = int(month_prices["Period Start"].values[0][:4])
                if year_data > datetime.today().year:
                    curr_price = month_prices.Price.values[0] if not is_freight else month_prices.USDperday.values[0]/1000
                    prev_price = 0
                else:
                    prev_price = month_prices.Price.values[0] if not is_freight else month_prices.USDperday.values[
                                                                                         0] / 1000
                    curr_price = 0
        return curr_price, prev_price

    nwe_curr_prev = [get_curr_and_prev_for_month_on_df(m, df_nwe_discounts) for m in months]
    nwe_prev = [p for _, p in nwe_curr_prev]
    nwe_curr = [c for c, _ in nwe_curr_prev]

    phys_curr_prev = [get_curr_and_prev_for_month_on_df(m, df_spark30_phys, is_freight=True) for m in months]
    phys_prev = [p for _, p in phys_curr_prev]
    phys_curr = [c for c, _ in phys_curr_prev]

    ffa_curr_prev = [get_curr_and_prev_for_month_on_df(m, df_spark30_ffa, is_freight=True) for m in months]
    ffa_prev = [p for _, p in ffa_curr_prev]
    ffa_curr = [c for c, _ in ffa_curr_prev]


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

    # PLOTLY: replacement — numeric y with manual bar offsets (exact replica of matplotlib)
    fig = go.Figure()
    months_list = df['Month'].tolist()
    y = np.arange(n_months)
    alpha = 0.5
    height = 0.25
    right_height = 0.125

    # Freight on xaxis (bottom) — 4 bars at distinct y-offsets
    fig.add_trace(
        go.Bar(
            y=y + height, x=df['FFA_Curr'].tolist(), orientation='h',
            name='FFA Freight (kUSD/day)', marker_color='#00D4FF',
            marker_line=dict(color='white', width=0.3), width=right_height,
            hovertemplate='FFA %{x:k$.3f} k$/day <extra></extra>',
        )
    )
    fig.add_trace(
        go.Bar(
            y=y + right_height, x=df['FFA_Prev'].tolist(), orientation='h',
            marker_color='#00D4FF', opacity=alpha, showlegend=False, width=right_height,
            hovertemplate='Prev FFA %{x:k$.3f} k$/day <extra></extra>',
        )
    )
    fig.add_trace(
        go.Bar(
            y=y - right_height, x=df['Phys_Curr'].tolist(), orientation='h',
            name='Phys Freight (kUSD/day)', marker_color='#3399FF',
            marker_line=dict(color='white', width=0.3), width=right_height,
            hovertemplate='Phys %{x:k$.3f} k$/day <extra></extra>',
        )
    )
    fig.add_trace(
        go.Bar(
            y=y - height, x=df['Phys_Prev'].tolist(), orientation='h',
            marker_color='#3399FF', opacity=alpha, showlegend=False, width=right_height,
            hovertemplate='Prev Phys %{x:k$.3f} k$/day <extra></extra>',
        )
    )

    # NWE on xaxis2 (top) — overlapping freight at same y-offsets, different axis = no collision
    fig.add_trace(
        go.Bar(
            y=y + right_height, x=df['NWE_Curr'].tolist(), orientation='h',
            name='NWE Discount (USD/MMBtu)', marker_color='#FF0055',
            marker_line=dict(color='white', width=0.3), xaxis='x2', width=height,
            hovertemplate='NWE %{x:.3f} $/mmbtu <extra></extra>',
        )
    )
    fig.add_trace(
        go.Bar(
            y=y - right_height, x=df['NWE_Prev'].tolist(), orientation='h',
            marker_color='#FF0055', opacity=alpha, showlegend=False, xaxis='x2', width=height,
            hovertemplate='prev NWE %{x:.3f} $/mmbtu <extra></extra>',
        )
    )

    x_nwe_axis = min(min(df['NWE_Prev']), min(df['NWE_Curr'])) - 0.1

    # Dummy for legend
    fig.add_trace(go.Bar(y=[None], x=[None], orientation='h',
        name='Prev Day Value', marker_color='gray', opacity=alpha))

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      font=dict(family="sans-serif", color="#C9D1D9", size=14),
                      barmode='overlay', height=500, legend=dict(x=0.95, y=1.02))
    fig.update_layout(
        xaxis=dict(range=[-38, 114],
                   tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                   title=dict(text=f'Freight Rates (k$/day): Latest on {latest_spark30_date} (Spark30 174k)',
                              font=dict(color='#3399FF')),
                   gridcolor='#30363D', linecolor='#30363D'),
        xaxis2=dict(range=[x_nwe_axis, -3*x_nwe_axis], overlaying='x', side='top',
                    tickvals=[0, -0.5, -1.0],
                    title=dict(text=f'NWE Discount ($/MMBtu): Latest on {latest_nwe_date} (Spark)',
                               font=dict(color='#FF0055')),
                    gridcolor='#30363D', linecolor='#30363D'),
        yaxis=dict(tickvals=list(y), ticktext=months_list,
                   gridcolor='#30363D', linecolor='#30363D'),
    )
    fig.add_vline(x=0, line_color='white', line_width=1.5)
    return fig


def create_bgn_lng_report_grid(report_date):
    # # import io
    # # from PIL import Image
    # # def capture_function_plot(): ...
    # # plt.rcParams.update({...})
    # # fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # # ... imshow / savefig / plt.show() ...
    import plotly.io as pio

    pricing_date = datetime.now().date()

    fig10 = plot_nicely_lng_on_water_curves(
        lng_on_water_20d_count="LNGG20DC Index",
        lng_on_water_30d_count="LNGG30DC Index",
        lng_on_water_20d_vol="LNGG20DT Index",
        lng_on_water_30d_vol="LNGG30DT Index",
        back_periods=720
    )

    fig1 = plot_spark_quotes()
    fig2 = plot_nicely_historical_prices(
        ttf_tick="TYRF27 Comdty", jkm_tick="AJKMY 27 BCFV Index",
        hh_tick="FFHHY 27 Index", brent_tick="FSBTY 27 Index",
        month="Cal27 Contracts", back_periods=180
    )
    fig3 = plot_ttf_and_eu_spreads(years=3, today_date=pricing_date, plot_spreads=False)
    fig4 = plot_hh_tfu_jkm()
    fig5 = plot_nicely_baltic_freight(
        blng1_ticker="IKA1 Comdty",
        blng2_ticker="IKD1 Comdty",
        blng3_ticker="IKI1 Comdty",
        spark30_ticker="LBE1 Comdty",
        back_periods=720
    )
    # fig1, fig2, fig3, fig4, fig5 = [], [], [], [], []
    fig6 = plot_forwards_baltic_freight()
    fig7, fig8, html9 = get_bsrch_lng_figs()
    # fig5 = plot_nwe_vs_nbp(2, pricing_date)

    figs_to_htmlise = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig10]

    divs = [pio.to_html(f, full_html=False, include_plotlyjs=False) for f in figs_to_htmlise]
    divs = divs + [html9]
    # div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        body {{ background-color: #0D1117; color: #C9D1D9; font-family: sans-serif; margin: 10px; }}
        h1 {{ text-align: center; font-size: 22px; margin: 8px 0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
        .grid > div {{ background: #0D1117; border: 1px solid #30363D; border-radius: 6px; }}
        .full-width {{ margin-top: 10px; border: 1px solid #30363D; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>BGN LNG Desk: {report_date.strftime('%d-%b-%Y')} Daily Report</h1>
    <div class="grid">
        <div>{divs[0]}</div>
        <div>{divs[1]}</div>
        <div>{divs[2]}</div>
        <div>{divs[3]}</div>
        <div>{divs[4]}</div>
        <div>{divs[5]}</div>
        <div>{divs[6]}</div>
        <div>{divs[7]}</div>
        <div>{divs[8]}</div>
        <div>{divs[9]}</div>
    </div>
</body>
</html>"""

    out_path = f"lng_report_{pricing_date}.html"
    out_path_online = f"C:\\Users\\marti.fernandezreal\\BAYEGAN DIS TIC. A.S\\LNG Team - 01. Miscellaneous\\17. LNG BGN Reports\\lng_report_{pricing_date}.html"
    out_path_online_base = f"C:\\Users\\marti.fernandezreal\\BAYEGAN DIS TIC. A.S\\LNG Team - 01. Miscellaneous\\17. LNG BGN Reports\\lng_report.html"
    for save_path in [out_path, out_path_online, out_path_online_base]:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Interactive report saved to {save_path}")

    import base64

    def fig_to_base64_img(fig, width=600, height=400):
        """Convert a Plotly figure to a base64-embedded <img> tag."""
        img_bytes = fig.to_image(format="png", width=width, height=height)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:{width}px; display:block;">'

    img_tags = [fig_to_base64_img(fig) for fig in figs_to_htmlise] + [html9]

    simplified_html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body style="background-color: #0D1117; color: #C9D1D9; font-family: Arial, sans-serif; margin: 10px;">
        <h1 style="text-align: center; font-size: 22px; margin: 8px 0;">
            BGN LNG Desk: {report_date.strftime('%d-%b-%Y')} Daily Report
        </h1>
        <table cellpadding="3" cellspacing="0" border="0" style="width:100%; border-collapse: collapse;">
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[0]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[1]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[2]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[3]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[4]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[5]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[6]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[7]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[8]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[9]}</td>
            </tr>
        </table>
    </body>
    </html>"""

    send_email(
        simplified_html,
        html_out_path=out_path,
        # sending_to="lng@bgn-int.com; vasileios.giannoutsos@bgn-int.com"
        sending_to="marti.fernandezreal@bgn-int.com; vasileios.giannoutsos@bgn-int.com"
    )

def send_email(html, html_out_path, sending_to="marti.fernandezreal@bgn-int.com"):

    import win32com.client
    from pathlib import Path

    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # 0 = MailItem

    mail.To = sending_to
    # mail.CC = "boss@company.com"  # optional
    mail.Subject = f"LNG Report - {date.today().isoformat()}"
    report_path = Path(html_out_path)
    mail.Attachments.Add(str(report_path.absolute()))
    mail.HTMLBody = html  # for Outlook

    mail.Send()



def violin_plots_ttf_nbp_spread(start_date, back_periods=400):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

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


def send_test_email():
    simplified_html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="background-color: #0D1117; color: #C9D1D9; font-family: Arial, sans-serif; margin: 10px;">
            <h1 style="text-align: center; font-size: 22px; margin: 8px 0;">
                BGN LNG Desk: Daily Report
            </h1>
            <table cellpadding="3" cellspacing="0" border="0" style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="width:50%; border:1px solid #30363D; padding:5px;"></td>
                    <td style="width:50%; border:1px solid #30363D; padding:5px;"></td>
                </tr>
            </table>
        </body>
        </html>"""

    send_email(
        simplified_html,
        html_out_path="C:\\Marti\\lng_on_water.html",
        sending_to="marti.fernandezreal@bgn-int.com"
    )


if __name__ == "__main__":

    from datetime import datetime, timedelta, date
    import pandas as pd
    import plotly.io as pio

    pricing_date = datetime.now().date()
    create_bgn_lng_report_grid(pricing_date)

    # fig = plot_spark_quotes()
    # html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    #
    # html = f"""<!DOCTYPE html>
    # <html>
    # <head>
    #     <meta charset="utf-8">
    #     <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    #     <style>
    #         body {{ background-color: #0D1117; color: #C9D1D9; font-family: sans-serif; margin: 10px; }}
    #         h1 {{ text-align: center; font-size: 22px; margin: 8px 0; }}
    #         .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    #         .grid > div {{ background: #0D1117; border: 1px solid #30363D; border-radius: 6px; }}
    #         .full-width {{ margin-top: 10px; border: 1px solid #30363D; border-radius: 6px; }}
    #     </style>
    # </head>
    # <body>
    #     <h1>BGN LNG Desk: Daily Report</h1>
    #     <div class="grid">
    #         <div>{html}</div>
    #     </div>
    # </body>
    # </html>"""
    #
    # fig.write_image("C:\\Marti\\fig1.jpeg")
    # out_path = f"C:\\Marti\\fig1.html"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     f.write(html)
    #     print(f"Test html saved in {out_path}")

    # send_test_email()