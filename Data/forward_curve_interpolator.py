
import numpy as np
from datetime import date
from scipy.interpolate import interp1d

from Utils.datetime_utils import time_between, month_int_from_string, is_tenor_month, infer_ratios_from_monthly_quotes, \
    tenor_to_monthly_strip, get_month_from_tenor, date_tenor_from_label


class ForwardCurve:
    base_periods = ["Q1", "Q2", "Q3", "Q4", "SS", "WS", "Y"]
    def __init__(self, dict_tenor_to_price, pricing_date):
        tenors = [t for t in dict_tenor_to_price.keys()]
        self.pricing_date = pricing_date
        self.monthly_quotes = {
            t: dict_tenor_to_price[t]
            for t in tenors
            if (is_tenor_month(t) and date_tenor_from_label(t)>pricing_date)
        }
        non_monthly_quotes ={t: dict_tenor_to_price[t]
                             for t in tenors
                             if not is_tenor_month(t)}

        self._populate_quoted_shape(self.monthly_quotes)
        self._infer_monthly_quotes(non_monthly_quotes)

        def get_maturity_for_tenor(t):
            if isinstance(t, float):
                return t
            else:
                return self._get_tenor_float_from_month_string(t)

        self.float_tenors_monthly = {get_maturity_for_tenor(t): p for t, p in self.monthly_quotes.items()}

        sorted_tenors = sorted(self.float_tenors_monthly)
        sorted_prices = [self.float_tenors_monthly[t] for t in sorted_tenors]

        self._interp = interp1d(sorted_tenors, sorted_prices, fill_value='extrapolate')

    def _populate_quoted_shape(self, monthly_tenors):
        self._ratio_dict = infer_ratios_from_monthly_quotes(
            monthly_tenors,
            self.base_periods
        )

    def _infer_monthly_quotes(self, non_monthly_quotes):
        for p_str in self.base_periods:
            provided_quotes = {q: non_monthly_quotes[q] for q in non_monthly_quotes if p_str in q}
            for q, avg_period in provided_quotes.items():
                months_inferred = tenor_to_monthly_strip(q)
                for m in months_inferred:
                    if m not in self.monthly_quotes and date_tenor_from_label(m)>self.pricing_date:
                        month_code = get_month_from_tenor(m)
                        ratio = self._ratio_dict[p_str][month_code]
                        self.monthly_quotes[m] = ratio * avg_period


    def _get_tenor_float_from_month_string(self, month_string):
        _month, _year = month_string.split('_')
        year = int("20" + _year)
        try:
            month = month_int_from_string(_month)
            tenor_date = date(year, month, 15)
            tenor_float = time_between(self.pricing_date, tenor_date)
            return tenor_float
        except KeyError:
            return 1000.0

    def get_value_on_month(self, month_str):
        # Assumes month is like "Mar_28"
        float_tenor = self._get_tenor_float_from_month_string(month_str)
        price = self._interp(float_tenor)
        return price
