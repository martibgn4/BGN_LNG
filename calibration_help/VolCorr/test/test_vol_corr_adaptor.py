import math
from datetime import date, timedelta

from general_utils import DAYS_PER_YEAR
from pytest_quants import assert_almost_equal
from thorn.core.market.volatility.vc.vol_corr import VolCorr
from thorn.core.market.volatility.vc.vol_corr_adaptor import DatedVolCorrAdaptor, MissingPriceProcessError
from time_period import PEAK, BASE, DAYTIME
from time_period import get_start, intersection
from time_period import quarter, DateRange
from underlying import SwapRate
from underlying.fixtures.underlyings import UKPOWER, NBP, TTF, PLATTSDATEDBRENT, BRENT, \
    UKPOWER_PEAK

from thorn.core.market.volatility.vc.test.fixtures import (
    MeanRevertingProcessFactory, vol_corr)
import pytest


class TestVolCorrAdaptor:
    # VolCorr test case
    # Magic numbers reproduceable with spreadsheet:
    # Misc/multi_factor_dynamics/Manual Calculations/VolCorr.xls

    pricing_date = date(2012, 10, 4)
    lsdr = intersection(quarter(2014, 1), PEAK)
    base_lsdr = intersection(quarter(2014, 1), BASE)

    def setup_method(self):
        self.vol_corr = vol_corr
        self.vol_corr_start_date = date(2012, 10, 4)
        self.vol_corr_adaptor = DatedVolCorrAdaptor(self.vol_corr, self.vol_corr_start_date)

    def test_calc_lsu_t_T_D(self):
        swap_rate = SwapRate(UKPOWER, self.lsdr)
        date_range = swap_rate.time_period.date_range
        expt_lsu = self.vol_corr_adaptor.get_lsu(swap_rate)
        expt_t = ((date_range.start - self.pricing_date).days - 1.0) / 365.0
        expt_T = expt_t + 1.0 / 365.0
        expt_D = ((date_range.end - date_range.start).days + 1) / 365.0

        assert self.vol_corr_adaptor.calc_t_T_D(swap_rate) == \
                         (expt_t, expt_T, expt_D)

        # Test default behaviour
        result = self.vol_corr_adaptor.calc_lsu_t_T_D(swap_rate)
        assert result == (expt_lsu, expt_t, expt_T, expt_D)

        # Test that the exercise date can be overridden
        day_offset = 2
        ex_date = self.vol_corr_adaptor.vol_corr_start_date + timedelta(days=day_offset)
        overridden_t = day_offset / 365
        result = self.vol_corr_adaptor.calc_lsu_t_T_D(swap_rate, ex_date)
        assert result == (expt_lsu, overridden_t, expt_T, expt_D)

        # Test that times must be ascending
        with pytest.raises(AssertionError):
            _ = self.vol_corr_adaptor.calc_lsu_t_T_D(swap_rate, self.pricing_date - timedelta(1))

        # Check that swap rate starting prior to the vol_corr_start_date fails
        with pytest.raises(AssertionError):
            broken_vol_corr_adapator = DatedVolCorrAdaptor(self.vol_corr, get_start(date_range) + timedelta(1))
            broken_vol_corr_adapator.calc_lsu_t_T_D(swap_rate)

        # Test that ex_date of start of period returns t == T
        _, t, T, _ = self.vol_corr_adaptor.calc_lsu_t_T_D(swap_rate, get_start(date_range))
        assert t == T

        # This should be call-able, and return zero time
        start_of_delivery_adapator = DatedVolCorrAdaptor(self.vol_corr, get_start(date_range))
        _, t, T, _ = start_of_delivery_adapator.calc_lsu_t_T_D(swap_rate, get_start(date_range))
        assert t == T
        assert t == 0

    def test_calc_terminal_vol_corr_bad(self):
        """Check that an error is raised if the exercise date is in the past
        """
        lsdr = intersection(DateRange(date(2012, 10, 4), date(2012, 10, 10)), PEAK)
        with pytest.raises(AssertionError):
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr), date(2012, 10, 3))

    def test_calc_terminal_vol_corr(self):
        lsdr = intersection(DateRange(date(2012, 10, 6)), PEAK)
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr)),
            0.30595741326870507,
            atol=1e-5,
        )
        nbp_vol = 0.22972448383751892
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(NBP, lsdr)),
            nbp_vol,
            atol=1e-5,
        )
        assert_almost_equal(
            self.vol_corr_adaptor.calc_var(SwapRate(NBP, lsdr)),
            nbp_vol * nbp_vol * (date(2012, 10, 5) - date(2012, 10, 4)).days / DAYS_PER_YEAR,
            atol=1e-5,
        )
        assert_almost_equal(
            self.vol_corr_adaptor.calc_corr(SwapRate(NBP, lsdr), SwapRate(UKPOWER, lsdr)),
            0.72161495421043131,
            atol=1e-5,
        )

    def test_calc_vol_with_load_shapes(self):
        dr = DateRange(date(2012, 10, 6), date(2012, 10, 6 + 7))  # A week so has off-peak days
        lsdr = intersection(dr, PEAK)
        # Peak in the time_period and the underlying
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER_PEAK, lsdr)),
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr)),
        )
        # Peak in time_period vs Peak in the underlying
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER_PEAK, dr)),
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr)),
        )

    def test_calc_terminal_vol_corr2(self):
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, self.lsdr)),
            0.18642021847963824,
            atol=1e-5,
        )
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(NBP, self.lsdr)),
            0.076153403081448259,
            atol=1e-5,
        )
        assert_almost_equal(
            self.vol_corr_adaptor.calc_corr(
                SwapRate(NBP, self.lsdr), SwapRate(UKPOWER, self.lsdr)),
            0.40741119838514983,
            atol=1e-5,
        )

    def test_calc_terminal_vol_corr_different_delivery_and_obs(self):
        lsdr1 = intersection(DateRange(date(2012, 10, 14)), PEAK)
        lsdr2 = intersection(DateRange(date(2012, 10, 16)), PEAK)
        obs_date = date(2012, 10, 12)

        assert_almost_equal(
            self.vol_corr_adaptor.calc_corr(SwapRate(UKPOWER, lsdr1), SwapRate(NBP, lsdr2), obs_date),
            0.69668581,
            atol=1e-5,
        )

        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr1), obs_date),
            0.277550234,
            atol=1e-5,
        )

    def test_calc_vol_date_range(self):
        '''Test we get the same price from a (DateRange, Base) and DateRange'''
        lsdr = intersection(DateRange(date(2012, 10, 14)), BASE)
        dr = DateRange(date(2012, 10, 14))
        obs_date = date(2012, 10, 12)

        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(NBP, lsdr), obs_date),
            self.vol_corr_adaptor.calc_vol(SwapRate(NBP, dr), obs_date)
        )

    def test_calc_terminal_vol_corr_zero_time_to_delivery(self):
        '''
        This test checks that, if the vol_corr is requested a volatility or a correlation
        with zero time to maturity, the returned value is almost the same as for a nearly zero
        time to maturity request done to the calculator, ie that the special case formulas
        with zero time to maturity degenerate correctly.

        NB: we compare vol_corr vs calculator because the vol_corr
        takes only dates, whereas the calculator can take any float
        We therefore compare:
        vol_corr(0 days)
        and
        calculator(1e-8 year)
        '''
        lsdr1 = intersection(DateRange(date(2012, 10, 14)), PEAK)
        lsdr2 = intersection(DateRange(date(2012, 10, 16)), PEAK)
        obs_date = self.pricing_date

        time_to_maturity = 1e-8
        vol_corr = self.vol_corr_adaptor.vol_corr
        power_variance = vol_corr.calc_covariance(UKPOWER_PEAK, UKPOWER_PEAK, t=time_to_maturity,
                                                  T1=(date(2012, 10, 14) - self.pricing_date).days / 365.0,
                                                  D1=1.0 / 365)
        gas_variance = vol_corr.calc_covariance(NBP, NBP, t=time_to_maturity,
                                                T1=(date(2012, 10, 16) - self.pricing_date).days / 365.0,
                                                D1=1.0 / 365)
        covariance = vol_corr.calc_covariance(UKPOWER_PEAK, NBP, t=time_to_maturity,
                                              T1=(date(2012, 10, 14) - self.pricing_date).days / 365.0,
                                              D1=1.0 / 365,
                                              T2=(date(2012, 10, 16) - self.pricing_date).days / 365.0,
                                              D2=1.0 / 365)

        assert_almost_equal(
            self.vol_corr_adaptor.calc_corr(
                SwapRate(UKPOWER, lsdr1), SwapRate(NBP, lsdr2), obs_date),
            covariance / math.sqrt(power_variance * gas_variance)
        )
        assert_almost_equal(
            self.vol_corr_adaptor.calc_vol(SwapRate(UKPOWER, lsdr1), obs_date),
            math.sqrt(power_variance / time_to_maturity)
        )

    def test_get_lsu_with_load_shape(self):
        lsdr = intersection(quarter(2014, 1), PEAK)
        assert self.vol_corr_adaptor.get_lsu(SwapRate(UKPOWER, lsdr)) == UKPOWER_PEAK
        assert self.vol_corr_adaptor.get_lsu(SwapRate(NBP, lsdr)) == NBP
        with pytest.raises(MissingPriceProcessError):
            _ = self.vol_corr_adaptor.get_lsu(SwapRate(TTF, lsdr))

    def test_get_lsu_with_proxy_underlying(self):
        swap_rate = SwapRate(PLATTSDATEDBRENT, self.base_lsdr)
        brent_vol_corr_adapator = DatedVolCorrAdaptor(VolCorr(MeanRevertingProcessFactory.build_brent_mr_procs()), self.vol_corr_start_date)
        expected_lsu = BRENT
        assert expected_lsu == brent_vol_corr_adapator.get_lsu(swap_rate)

    def test_get_day_time_price_process(self):
        lsdr = intersection(DateRange(date(2011, 3, 1)), DAYTIME)
        assert self.vol_corr_adaptor.get_lsu(SwapRate(UKPOWER, lsdr)) == UKPOWER_PEAK


