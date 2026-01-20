from xbbg import blp

dict_comm_to_tickers = {
    "HenryHub": "NG",
    "HH": "NG",
    "TTF": ("TZT", "QZT", "QQT", "QTT"),
    "Brent": "CO",
    "NBP": ("FN", "QR", "SA", "YAA"),
    "THE": ("NCG", "NCG", "NCG", "NCG"),
    "PEG": ("PNG", "PNG", "PNG", "PNG"),
    "PSV": ("PSR", "PSR", "PSR", "PSR"),
    "PVB": ("PXB", "PXB", "PXB", "PXB"),
    "VTP": ("CEG", "CEG", "CEG", "CEG"),
}


bbg_month_codes = ["F", "G", "H", "J", "K", "M",
               "N", "Q", "U", "V", "X", "Z"]
bbg_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

add_suffix_underlyings = ["THE", "PEG", "PSV", "PVB", "VTP"]


class BloombergExtractor:
    def __init__(self, commodity, months=120, quarters=12, seasons=10, years=5,
                 start_year = 26):
        if commodity in dict_comm_to_tickers:
            self.commodity_codes = dict_comm_to_tickers[commodity]
            self.q_ticker, self.s_ticker, self.y_ticker = None, None, None
            if isinstance(self.commodity_codes, tuple):
                self.commodity_code, self.q_ticker, self.s_ticker, self.y_ticker = self.commodity_codes
            else:
                self.commodity_code = self.commodity_codes
        else:
            raise ValueError(f"Invalid commodity {commodity}, not supported yet")
        self.start_year = start_year

        self.months = months
        self.quarters = quarters
        self.seasons = seasons
        self.years = years
        self.commodity = commodity

    def get_monthly_tickers(self):
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

    def retrieve_latest_bbg_values(self):
        bbg_tickers, raw_tickers = self.get_all_tickers()
        _bbg_tickers = [t + " Comdty" for t in bbg_tickers]
        dict_tickers = {bbg_tick: raw_tick for bbg_tick, raw_tick in zip(_bbg_tickers, raw_tickers)}
        all_quotes = blp.bdp(
            tickers= _bbg_tickers,
            flds=['Last_Price', "volume", "time"]
        )

        all_quotes["raw_tickers"] = [dict_tickers[t] for t in all_quotes.index]
        all_quotes = all_quotes.set_index("raw_tickers")
        return all_quotes


if __name__ == "__main__":
    ttf_quotes = BloombergExtractor(
        "TTF",
        start_year=26
    ).retrieve_latest_bbg_values()

    hh_quotes = BloombergExtractor(
        "HH",
        start_year=26
    ).retrieve_latest_bbg_values()


    a = 1
