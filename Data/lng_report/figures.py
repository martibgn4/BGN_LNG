"""Plotly figures that make up the BGN LNG daily report.

Every function here returns a ``plotly.graph_objects.Figure`` (except the
Bloomberg BSRCH panels, which come from ``Data.bloomberg_bsrch``). They are
assembled into the HTML report by ``report.create_bgn_lng_report_grid``.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Data.forward_curve_interpolator import ForwardCurve
from Utils.datetime_utils import date_tenor_from_label, label_from_date_tenor

from Data.lng_report.config import add_suffix_underlyings
from Data.lng_report.bloomberg import (
    BloombergExtractor,
    get_historical_prices_for_underlyings,
)
from Data.lng_report.spark import extract_spark_quotes
from Data.lng_report.styling import dark_layout


def plot_hh_tfu_jkm():
    tfu_quotes = BloombergExtractor(
        "TFU",
        start_year=26
    ).retrieve_latest_bbg_values("px_settle")

    hh_quotes = BloombergExtractor(
        "HH",
        start_year=26
    ).retrieve_latest_bbg_values("px_settle")

    # jkm_quotes = BloombergExtractor(
    #     "JKM",
    #     start_year=26
    # ).retrieve_latest_bbg_values()

    jkm_settles = pd.read_csv("C:\\Marti\\ClaudeProjects\\data\\raw\\jkm_settlements\\master.csv")
    last_date = jkm_settles.settlement_date.max()
    last_date_as_date = datetime.strptime(last_date, "%Y-%m-%d").date()

    settleimplied_tfu_date = datetime.strptime(tfu_quotes.time.mode().values[0], "%m/%d/%Y").date()

    if last_date_as_date != settleimplied_tfu_date:
        raise ValueError(f"Date mismatch when reading TFU and JKM settlements! Got TTF as {settleimplied_tfu_date} and JKM as {last_date_as_date}")

    jkm_last_settles = jkm_settles[jkm_settles.settlement_date == last_date]
    jkm_last_settles.strip_label = jkm_last_settles.strip_label.apply(lambda x: x[:3] + "_" + x[-2:])

    common_tenors = [t for t in hh_quotes.index if t in tfu_quotes.index]

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
    fig.update_layout(**dark_layout(), height=500,
                      title=dict(text="Spread Forward Curves", x=0.01),
                      xaxis_title="Tenor", yaxis_title="USD/MMBtu",
                      legend=dict(x=0.75, y=0.98))
    return fig


def plot_ttf_and_eu_spreads(years, today_date, plot_spreads=True):
    bbg_price_label = "px_settle"
    ttf_quotes = BloombergExtractor(
        "TTF", start_year=26, years=years
    ).retrieve_latest_bbg_values(bbg_price_label="px_settle")

    dict_tenor_to_price_ttf = {m: ttf_quotes.loc[m][bbg_price_label] for m in ttf_quotes.index}
    ttf_curve = ForwardCurve(dict_tenor_to_price_ttf, today_date)

    ttf_monthly_quotes = ttf_curve.monthly_quotes

    fig = go.Figure()
    for c in ["NBP", "THE", "PEG", "PSV", "PVB", "VTP"]:
        _start_year = 26 if c not in add_suffix_underlyings else 6
        c_quotes = BloombergExtractor(commodity=c, start_year=_start_year, years=years).retrieve_latest_bbg_values(bbg_price_label)

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
                    ttf_nbp_threshold[monthly_t] = nbp_transformed * 0.985 - 1.24

        dates = [date_tenor_from_label(k) for k in ttf_c_spread.keys()]
        label = "TTF - " + c if plot_spreads else c
        fig.add_trace(go.Scatter(x=dates, y=list(ttf_c_spread.values()),
                                 mode='lines', name=label, line=dict(width=3)))
        if c == "NBP" and ttf_nbp_threshold:
            fig.add_trace(go.Scatter(x=dates, y=list(ttf_nbp_threshold.values()),
                                     mode='lines', name="98.5%NBP - 1.24", line=dict(width=3)))

    if not plot_spreads:
        # Plot TTF separately
        ttf_quotes = {}
        for monthly_t in ttf_monthly_quotes.keys():
            ttf_quotes[monthly_t] = ttf_curve.get_value_on_month(monthly_t)
        dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
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

    fig.update_layout(**dark_layout(), height=500,
                      title=dict(text=title, x=0.01),
                      xaxis_title="Tenor", yaxis_title="EUR/MWh",
                      legend=dict(x=1.02, y=0.5, xanchor='left'))
    return fig


def plot_nwe_vs_nbp(years, today_date):
    bbg_price_label = "px_settle"
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

    fig = go.Figure()
    for c in ["NBP"]:
        _start_year = 26 if c not in add_suffix_underlyings else 6
        c_quotes = BloombergExtractor(commodity=c, start_year=_start_year, years=years).retrieve_latest_bbg_values(bbg_price_label)

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
            ttf_nbp_threshold[monthly_t] = nbp_transformed * 0.985 - 0.45 / eurusd * 3.412142   # TTF limit from below for buying TTF from UK

        dates = [date_tenor_from_label(k) for k in ttf_nbp_threshold.keys()]
        fig.add_trace(go.Scatter(x=dates, y=list(ttf_nbp_threshold.values()),
                                 mode='lines', name="98.5%NBP - 0.45 $/mmbtu", line=dict(width=2)))

    # Plot TTF separately
    ttf_quotes = {}
    for monthly_t in ttf_monthly_quotes.keys():
        ttf_quote = ttf_curve.get_value_on_month(monthly_t)
        nwe_discount = nwediscounts_curve.get_value_on_month(monthly_t) / eurusd_curve.get_value_on_month(monthly_t) * 3.412142
        ttf_quotes[monthly_t] = ttf_quote + nwe_discount

    dates = [date_tenor_from_label(k) for k in ttf_quotes.keys()]
    fig.add_trace(go.Scatter(x=dates, y=list(ttf_quotes.values()),
                             mode='lines', name="NWE", line=dict(width=2)))

    title = f"NWE vs NBP. COB: {(datetime.today() - timedelta(1)).strftime('%Y-%b-%d')} "

    fig.update_layout(**dark_layout(), height=400,
                      title=dict(text=title, x=0.01),
                      xaxis_title="Tenor", yaxis_title="EUR/MWh",
                      legend=dict(x=1.02, y=0.5, xanchor='left'))
    return fig


def plot_nicely_historical_prices(ttf_tick, jkm_tick, hh_tick, month, brent_tick=None, back_periods=180):
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings([ttf_tick, jkm_tick, hh_tick, brent_tick], dates[0], dates[-1])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)

    df['JKM-TTF'] = df[jkm_tick] - df[ttf_tick]
    # df['JKM-HH'] = df[jkm_tick] - df[hh_tick]

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
    fig.add_trace(go.Scatter(x=df.index, y= 0.17 * df[brent_tick], name='17% Brent',
                             line=dict(color='#FFB800', width=1.5, dash='dash'),
                             opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[brent_tick], name='Brent',
                             line=dict(color='#FFB800', width=1.5)),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['JKM-TTF'], name='JKM-TTF',
                             line=dict(color='#00D4FF', width=1), opacity=0.8,
                             fill='tozeroy', fillcolor='rgba(0,212,255,0.3)'), row=2, col=1)

    # fig.add_trace(go.Scatter(x=df.index, y=df['JKM-HH'], name='JKM-HH',
    #                          line=dict(color='#FFB800', width=1), opacity=0.6,
    #                          fill='tozeroy', fillcolor='rgba(255,184,0,0.1)'), row=2, col=1)

    fig.add_hline(y=0, line_color='white', line_width=0.5, row=2, col=1)
    fig.update_layout(**dark_layout(), height=500)
    fig.update_yaxes(title_text="USD/MMBtu", row=1, col=1)
    fig.update_yaxes(title_text="USD/bbl", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread (USD/MMBtu)", row=2, col=1)
    fig.update_xaxes(tickformat='%d %b %Y', row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(color='#C9D1D9', size=14)
    return fig


def plot_nicely_baltic_freight(blng1_ticker, blng2_ticker, blng3_ticker, spark30_ticker, back_periods=180):
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    df = get_historical_prices_for_underlyings(
        [blng1_ticker, blng2_ticker, blng3_ticker, spark30_ticker],
        dates[0], dates[-1]
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
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
    fig.update_layout(**dark_layout(), height=500,
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
    fig.update_layout(**dark_layout(), height=500,
                      title=dict(text="Baltic LNG Forward Curves", x=0.01),
                      xaxis_title="Tenor", yaxis_title="USD/day",
                      legend=dict(x=0.75, y=0.98))
    return fig


def plot_nicely_lng_on_water_curves(lng_on_water_20d_count, lng_on_water_30d_count,
                                    lng_on_water_20d_vol, lng_on_water_30d_vol, back_periods=180):
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
    fig.update_layout(**dark_layout(), height=500,
                      title=dict(text=f"LNG on Water as of {datetime.today().date().strftime('%d-%b-%Y')}: >20 vs >30 days", x=0.01),
                      xaxis_title="Day", yaxis_title="USD/day",
                      legend=dict(x=1.1, y=0.98))
    fig.update_yaxes(title_text="Vessel Count")
    fig.update_yaxes(title_text="Volume [Metric Tonnes]", secondary_y=True)

    return fig


def plot_nicely_lng_us_flows(back_periods=180):
    dates = pd.date_range(end=datetime.today(), periods=back_periods)

    dict_name_to_ticker = {
        "US-Total": "GSLIQTOT Index",
        "Golden Pass": "GSLIQGLF Index",
        "Freeport": "GSLIQFPT Index",
        "Sabine Pass": " GSLIQSPI Index",
        "Corpus Christi": "GSLIQCCH Index",
        "Plaquemines": "GSLIQPLQ Index",
        "Calcasieu Pass": "GSLIQCLC Index",
        "Cameron": "GSLIQCAM Index",
        "Cove Point": "GSLIQCOV Index",
        "Elba": "GSLIQELB Index",
        "Energia Costa Azul": "GSLIQECA Index",
    }

    dict_ticker_to_name = {v: k for k, v in dict_name_to_ticker.items()}

    df = get_historical_prices_for_underlyings(
        [k for k in dict_ticker_to_name],
        dates[0], dates[-1]
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)

    df = df * (-1037) * 1000 / 3_700_000 / 1_000_000  # Number of cargoes equivalent of LNG
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for name_terminal, ticker in dict_name_to_ticker.items():
        if name_terminal == "US-Total":
            fig.add_trace(go.Scatter(x=df.index, y=df[ticker], name=name_terminal,
                                     line=dict(color='#FF0055', width=3.5, dash='dash')),
                          secondary_y=True)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[ticker], name=name_terminal,
                                     line=dict(width=2.5)))

    fig.update_layout(**dark_layout(), height=500,
                      title=dict(text=f"US LNG flows latest on {datetime.today().date().strftime('%d-%b-%Y')}, standard cargo = 3.7 TBTu ",
                                 x=0.01),
                      xaxis_title="Day",
                      legend=dict(x=1.1, y=0.98))
    fig.update_yaxes(title_text="Number of Cargoes Equivalent per day")
    fig.update_yaxes(title_text="TOTAL Number of Cargoes Equivalent per day", secondary_y=True)

    return fig


def plot_spark_quotes():
    df_nwe_discounts = extract_spark_quotes(quote_type="Cargo", latest_only=False, limit=2, cal_month=None)
    df_spark30_phys = extract_spark_quotes(quote_type="FreightSpark30", latest_only=False, limit=2, cal_month=None)
    df_spark30_ffa = extract_spark_quotes(quote_type="FreightSpark30FFA", latest_only=False, limit=2, cal_month=None)

    months = [m for m in reversed(sorted(df_nwe_discounts["Period Start"].unique()))]
    n_months = len(months)

    latest_nwe_date = max(df_nwe_discounts["Release Date"]).date()
    latest_spark30_date = max(df_spark30_phys["Release Date"]).date()

    def get_curr_and_prev_for_month_on_df(_month, _df, is_freight=False):
        month_prices = _df[_df["Period Start"] == _month]
        data_available = month_prices.shape[0]
        if data_available >= 2:
            if not is_freight:
                curr_price = month_prices.Price.values[0]
                prev_price = month_prices.Price.values[1]
            else:
                curr_price = month_prices.USDperday.values[0] / 1000
                prev_price = month_prices.USDperday.values[1] / 1000
        else:
            if data_available == 0:
                curr_price, prev_price = 0, 0
            else:  # data_available = 1
                year_data = int(month_prices["Period Start"].values[0][:4])
                if year_data > datetime.today().year:
                    curr_price = month_prices.Price.values[0] if not is_freight else month_prices.USDperday.values[0] / 1000
                    prev_price = 0
                else:
                    prev_price = month_prices.Price.values[0] if not is_freight else month_prices.USDperday.values[0] / 1000
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

    # Numeric y with manual bar offsets (exact replica of the matplotlib chart).
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
        xaxis2=dict(range=[x_nwe_axis, -3 * x_nwe_axis], overlaying='x', side='top',
                    tickvals=[0, -0.5, -1.0],
                    title=dict(text=f'NWE Discount ($/MMBtu): Latest on {latest_nwe_date} (Spark)',
                               font=dict(color='#FF0055')),
                    gridcolor='#30363D', linecolor='#30363D'),
        yaxis=dict(tickvals=list(y), ticktext=months_list,
                   gridcolor='#30363D', linecolor='#30363D'),
    )
    fig.add_vline(x=0, line_color='white', line_width=1.5)
    return fig
