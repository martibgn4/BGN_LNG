import math
from datetime import timedelta

from general_utils import time_between
from time_period import get_end, get_start, intersection, is_equivalent, get_load_shape, month_containing


class MissingPriceProcessError(Exception):
    """Raised from VolCorr if a price process is missing"""
    pass


class DatedVolCorrAdaptor:
    """Adaptor class which allows use of VolCorr functionality
    with swap rates and exercise dates instead of times measured in years"""

    def __init__(self, vol_corr, pricing_date):
        self.vol_corr = vol_corr
        self.vol_corr_start_date = pricing_date

        self._ideal_lsu_cache = {}

    def _ideal_lsu(self, underlying, load_shape):
        try:
            return self._ideal_lsu_cache[underlying, load_shape]
        except KeyError:
            ideal_lsu = underlying.vol_model_underlying.get_shaped(load_shape)
            if ideal_lsu not in self.vol_corr.lsus:
                ideal_lsu = None
            self._ideal_lsu_cache[underlying, load_shape] = ideal_lsu
            return ideal_lsu

    def get_lsu(self, swap_rate):
        # critical function. optimized here for performance
        underlying = swap_rate.underlying
        ideal_lsu = self._ideal_lsu(underlying, get_load_shape(swap_rate.time_period))
        if ideal_lsu is not None:
            return ideal_lsu

        for lsu in self.vol_corr.get_intersecting_lsus(underlying):
            if is_equivalent(intersection(lsu.load_shape, swap_rate.time_period), swap_rate.time_period):
                return lsu

        raise MissingPriceProcessError(f"{swap_rate} not in {self.vol_corr.lsus}")

    def calc_t_T_D(self, swap_rate, ex_date=None):
        """Calculate the exercise time, start of delivery time and duration of delivery time in years
        for use with vol corr"""
        time_period = swap_rate.time_period
        underlying = swap_rate.underlying
        if underlying.vol_model_month_offset != 0:
            assert time_period.is_month, f"Expected swap rate for month ({swap_rate})"
            time_period = time_period.get_subsequent(-underlying.vol_model_month_offset)

        if ex_date is None:  # Attempt to establish the last trading date, and probably exercise date
            try:
                post_trading_date = underlying.post_trading_date(time_period)
            except AssertionError:  # Oil underlyings only give a post trading date for Monthly contracts
                post_trading_date = underlying.post_trading_date(month_containing(get_end(time_period)))

            ex_date = max(self.vol_corr_start_date, post_trading_date - timedelta(1))

            assert self.vol_corr_start_date <= ex_date <= post_trading_date, \
                "Exercise time (%s) must be between VolCorr start (%s) and last trading date (%s)" % \
                (ex_date, self.vol_corr_start_date, post_trading_date)

        else:  # Just check t >= 0
            assert self.vol_corr_start_date <= ex_date, \
                f"Exercise time ({ex_date}) must be on or after VolCorr start date ({self.vol_corr_start_date})"

        T = time_between(self.vol_corr_start_date, get_start(time_period))
        t = time_between(self.vol_corr_start_date, ex_date)
        D = time_between(get_start(time_period), get_end(time_period) + timedelta(days=1))

        return t, T, D

    def calc_lsu_t_T_D(self, swap_rate, ex_date=None):
        """Get the LoadShapedUnderlying to be passed to VolCorr when asking about the volatility of the given SwapRate.
        Calculate the exercise time, start of delivery time and duration of delivery time in years
        for use with vol corr.
        This method is normally used by other vol models only."""
        lsu = self.get_lsu(swap_rate)
        t, T, D = self.calc_t_T_D(swap_rate, ex_date)
        return lsu, t, T, D

    def calc_vol(self, swap_rate, ex_date=None, include_fx=False):
        '''
        Calculate the integrated volatility associated with a European vanilla option.
        @param swap_rate the forward contract underlying the option
        @param ex_date the exercise date of the option (defaults to the day before start of delivery)
        '''
        lsu, t, T, D = self.calc_lsu_t_T_D(swap_rate, ex_date)
        return self.vol_corr.calc_vol(lsu, t, T, D, include_fx)

    def calc_corr(self, swap_rate1, swap_rate2, ex_date=None, include_fx=False):
        '''
        Calculate the integrated correlation associated with a European spread option.
        @param swap_rate1 the first forward contract underlying the option
        @param swap_rate2 the second forward contract underlying the option
        @param ex_date the exercise date of the option
        (defaults to the earlier of the day before the start of delivery of swap_rate1
        and the day before the start of delivery of swap_rate2)
        '''
        lsu1, t1, T1, D1 = self.calc_lsu_t_T_D(swap_rate1, ex_date)
        lsu2, t2, T2, D2 = self.calc_lsu_t_T_D(swap_rate2, ex_date)
        t = min(t1, t2)
        if t > 0:
            variance1 = self.vol_corr.calc_covariance(lsu1, lsu1, t, T1, D1, include_fx=include_fx)
            variance2 = self.vol_corr.calc_covariance(lsu2, lsu2, t, T2, D2, include_fx=include_fx)
            covariance = self.vol_corr.calc_covariance(lsu1, lsu2, t, T1, D1, T2, D2, include_fx=include_fx)
            return covariance / math.sqrt(variance1 * variance2)
        else:
            return self.vol_corr.calc_inst_corr(lsu1, lsu2, T1, D1, T2, D2, include_fx=include_fx)

    def calc_var(self, swap_rate, ex_date=None, include_fx=False):
        lsu, t, T, D = self.calc_lsu_t_T_D(swap_rate, ex_date)
        return self.vol_corr.calc_var(lsu, t, T, D, include_fx)

    def calc_cov(self, swap_rate1, swap_rate2, ex_date=None, include_fx=False):
        lsu1, t1, T1, D1 = self.calc_lsu_t_T_D(swap_rate1, ex_date)
        lsu2, t2, T2, D2 = self.calc_lsu_t_T_D(swap_rate2, ex_date)
        t = min(t1, t2)
        if t > 0:
            return self.vol_corr.calc_covariance(lsu1, lsu2, t, T1, D1, T2, D2, include_fx=include_fx)
        else:
            return 0


