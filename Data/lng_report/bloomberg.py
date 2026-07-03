"""Bloomberg data access: forward-curve ticker generation and history pulls."""

from xbbg import blp

from Utils.datetime_utils import month_int_from_string, month_string_from_int

from Data.lng_report.config import (
    dict_fx_to_tickers,
    dict_comm_to_tickers,
    add_suffix_underlyings,
    bbg_months_available_fx,
    bbg_months_available_fx_int,
    bbg_month_codes,
    bbg_months,
)


class BloombergExtractor:
    """Build Bloomberg tickers for a curve and pull the latest quotes.

    Handles both FX crosses (BGN Curncy points, spot + forward pips) and
    commodity futures across monthly/quarterly/season/yearly strips.
    """

    def __init__(self, commodity, months=120, quarters=12, seasons=10, years=5,
                 start_year=26, start_month="Feb"):
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
                tickers = [t + "M" for t in tickers]
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
            tickers = [t + "Q" for t in tickers]

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
            tickers = [t + "S" for t in tickers]

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
            tickers = [t + "Y" for t in tickers]

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
                tickers=_bbg_tickers,
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
            all_quotes = spot_price + all_quotes / 10000
            return all_quotes


def get_historical_prices_for_underlyings(underlyings, start_date, end_date, price_label="px_last"):
    """Thin wrapper over ``blp.bdh`` for a list of tickers over a date range."""
    df_h = blp.bdh(
        underlyings,
        price_label,
        start_date,
        end_date,
    )
    return df_h
