import math
from datetime import date, timedelta

import numpy as np

from general_utils import time_between
from pytest_quants import assert_almost_equal
from quantity import MWHG, MWHP, PENCE, THERM, TONNEE, maximum, value, array, MWH
from thorn.core.market.dynamic.intrinsic_mkt_dynamic import IntrinsicMktDynamic
from thorn.core.market.dynamic.ivc_mkt_dynamic import IvcMktDynamic
from thorn.core.market.dynamic.lvc_mkt_dynamic import LvcMktDynamic
from thorn.core.market.dynamic.vc_mkt_dynamic import VcMktDynamic
from thorn.core.market.mkt_env.keys import (
    IrPivotCurrency, PricingDate, EndOfDayKey, VolSurfaceKey,
    MeanRevertingProcessesKey, CurveDate, PathCountKey, SeedKey, MktCalendarKey,
)
from thorn.core.market.mkt_env.mkt_env import MktEnv
from thorn.core.market.mkt_env.shifts import SwapRateShift
from thorn.core.market.test.fixtures.utils import minimal_static_data, create_mkt_env
from thorn.core.market.volatility.lvc.test.fixtures import PillaredSmileCollectionFactory
from thorn.core.market.volatility.vc.test.fixtures import (
    mean_rev_procs, nbpttf_mean_rev_procs, vol_corr, MeanRevertingProcessFactory,
)
from thorn.core.market.volatility.vc.vol_corr_adaptor import DatedVolCorrAdaptor
from thorn.core.pricers.analytical.basket_option_params import BasketOptionParams
from thorn.core.pricers.analytical.european_option import Black76Option
from thorn.core.pricers.analytical.kirk import KirkSpreadPricer, POS, NEG
from thorn.core.pricers.analytical.test.fixtures.basket_spread_option import BasketSpreadOption
from thorn.core.pricers.analytical.test.fixtures.kirk_spread_option import KirkSpreadOption
from thorn.core.pricers.analytical.vol_model_basket import LvcBasketVol, VcBasketVol
from time_period import month, year, MktCalendar
from trade_data import OptionType, OPTION_TYPES
from underlying import SwapRate, GBP, EUR
from underlying.fixtures.underlyings import (
    CARBONTAX, EURGBP, GBPEUR, GBPIR, NBP, TTF, UKPOWER, UKPOWER_PEAK, UKPOWER_OFFPEAK, EURIR, PSV,
)
import pytest


class TestKirk:

    def test_intrinsic(self):
        kirk = KirkSpreadPricer(9 * EUR, -8 * EUR, 0.5 * EUR, 0.1, 0.1, 0.1, 0.1)
        assert kirk.option_value(OptionType.CALL, intrinsic=True) == (9 - 8 - 0.5) * EUR
        assert kirk.option_value(OptionType.PUT, intrinsic=True) == 0 * EUR

    @pytest.mark.parametrize("pos_fwd", (9 * EUR, 8 * EUR))
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_delta(self, pos_fwd, option_type):
        """
        Test Delta and Finite Difference consistency with all combinations of:
        . Put / Call
        . In-the-money / Out-the-money
        . Intrinsic / VolCorr
        """

        def pricer(pos_fwd, option_type=OptionType.CALL):
            return KirkSpreadPricer(pos_fwd, -8 * EUR, 0.5 * EUR, 0.1, 0.1, 0.1, 1.0, option_type)

        shift_size = 1e-6 * EUR
        down, mid, up = [
            pricer(pos_fwd + shift * shift_size, option_type) for shift in (-1.0, 0.0, 1.0)
        ]

        fd_delta = (up.option_value() - down.option_value()) / (2.0 * shift_size)
        assert_almost_equal(fd_delta, mid.delta())

        fd_delta = (up.option_value(intrinsic=True) - down.option_value(intrinsic=True)) / (2.0 * shift_size)
        assert_almost_equal(fd_delta, mid.delta(intrinsic=True))


class TestBasketVol:

    @classmethod
    def setup_class(cls):
        cls.vol_corr_gbp = vol_corr.to_numeraire_currency(GBP)

    def test_basket_vol_single_asset(self):
        """
        Test that when computing the vol of a single asset, we get the same vol whether
        we call get_volatility or get_basket_volatility
        """
        pricing_date = date(2014, 1, 1)
        mkt_env = create_mkt_env(pricing_date)

        exercise_date = date(2014, 2, 15)
        strike = 50 * PENCE / THERM
        vol_model = VcMktDynamic(mkt_env, {})
        basket_vol = VcBasketVol(
            [SwapRate(NBP, month(2014, 4))], [1], exercise_date, strike, GBP, vol_model)
        assert pytest.approx(basket_vol.calc_vol()) == vol_model.get_volatility(NBP, month(2014, 4), exercise_date, strike)

    def test_holiday_adjustment(self):
        """
        Test obs_dates get adjusted
        """
        pricing_date = date(2014, 1, 1)
        exercise_date = date(2014, 2, 15)
        obs_date = date(2014, 2, 5)
        quotes = create_mkt_env(pricing_date).items.copy()
        quotes[MktCalendarKey] = MktCalendar(dates=[exercise_date, obs_date])
        mkt_env = MktEnv(quotes)
        strike = 50 * PENCE / THERM
        vol_model = VcMktDynamic(mkt_env, {})
        basket_vol = VcBasketVol(
            [SwapRate(NBP, month(2014, 4)), SwapRate(NBP, month(2014, 5))], [0.5, 0.5], exercise_date, strike, GBP,
            vol_model, obs_dates=[exercise_date, obs_date]
        )
        assert [exercise_date - timedelta(1), obs_date - timedelta(1)] == basket_vol.obs_dates
        assert exercise_date - timedelta(1) == basket_vol.ex_date

    def test_basket_vol_single_deterministic_asset(self):
        """
        Check that the basket price can price deterministic underlyings
        Also check the degenerate case where all vols are zero does not cause nan's in the analytics
        """
        pricing_date = date(2014, 1, 1)
        carbon_price = 11 * GBP / TONNEE
        swap_rate = SwapRate(CARBONTAX, month(2014, 4))
        overrides = {swap_rate: carbon_price}
        mkt_env = create_mkt_env(pricing_date, overrides=overrides)

        exercise_date = date(2014, 2, 15)
        strike = 10 * GBP / TONNEE
        basket_vol = VcBasketVol([swap_rate], [1], exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))
        assert basket_vol.price_option(OptionType.CALL) == carbon_price - strike

        # Check that the delta is calculated and consistent too...
        delta = basket_vol.kirk.delta(OptionType.CALL)
        shift_size = 1e-6
        up, down = [
            VcBasketVol(
                [swap_rate], [1], exercise_date, strike, GBP, VcMktDynamic(mkt_env.apply_shifts(shifts), {}),
            ).price_option(OptionType.CALL)
            for shifts in ([SwapRateShift(swap_rate, 1.0 + shift_size)], [SwapRateShift(swap_rate, 1.0 - shift_size)])
        ]
        divisor = 2.0 * carbon_price * shift_size
        assert_almost_equal((up - down) / divisor, delta)

    def test_get_covariance(self):
        pricing_date = date(2013, 2, 15)
        exercise_date = date(2014, 2, 15)
        nbp_price = 50 * PENCE / THERM
        mkt_env = create_mkt_env(pricing_date)
        vol_model = VcMktDynamic(mkt_env, {})
        vol_corr_adaptor = DatedVolCorrAdaptor(vol_model.vol_corr, mkt_env.pricing_date)

        ttm = 1.0
        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(NBP, month(2014, 12))]
        strike = nbp_price + nbp_price

        st_vol = vol_model.get_volatility(NBP, month(2014, 3), exercise_date, strike)
        lt_vol = vol_model.get_volatility(NBP, month(2014, 12), exercise_date, strike)
        correlation = vol_corr_adaptor.calc_corr(swap_rates[0], swap_rates[1], exercise_date)

        def test_weights(_weights):
            weighted_nbp_price = [nbp_price * w for w in _weights]
            expected_vol = weighted_nbp_price[0] ** 2 * math.exp(st_vol ** 2) \
                           + 2.0 * (weighted_nbp_price[0] * weighted_nbp_price[1]) \
                                 * math.exp(st_vol * lt_vol * correlation * ttm) \
                           + weighted_nbp_price[1] ** 2 * math.exp(lt_vol ** 2)
            expected_cov = math.log(expected_vol / (weighted_nbp_price[0] + weighted_nbp_price[1]) ** 2)
            basket_vol = VcBasketVol(
                swap_rates, _weights, exercise_date, strike, GBP,
                VcMktDynamic(mkt_env, {}))
            actual_cov = basket_vol.get_covariance(POS, POS)
            assert pytest.approx(expected_cov) == actual_cov

        for weights in [[1, 1], [0.5, 1]]:
            test_weights(weights)

    def test_missing_volcorr_underlying(self):
        """ Test that we throw an error if it is not possible to compute the vol from VolCorr """
        pricing_date = date(2013, 2, 15)
        exercise_date = date(2014, 2, 15)
        ukpower_quote = {SwapRate(UKPOWER, year(2014)): 50 * GBP / MWH}
        mkt_env = create_mkt_env(pricing_date, overrides=ukpower_quote)

        # No TTF in VolCorr
        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(TTF, month(2014, 12))]
        weights = [1, -1]
        strike = 1 * PENCE / THERM

        with pytest.raises(AssertionError):
            _ = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

        # UKPOWER_PEAK is in Volcorr - this is fine
        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(UKPOWER_PEAK, month(2014, 12))]
        weights = [1, -1]
        strike = 1 * PENCE / THERM
        _ = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

        # UKPOWER_OFFPEAK is not in Volcorr
        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(UKPOWER_OFFPEAK, month(2014, 12))]
        weights = [1, -1]
        strike = 1 * PENCE / THERM

        with pytest.raises(AssertionError):
            _ = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

        # Missing UKPOWER_OFFPEAK to compute BASE vol
        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(UKPOWER, month(2014, 12))]
        weights = [1, -1]
        strike = 1 * PENCE / THERM

        with pytest.raises(AssertionError):
            _ = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

        # Ex_date is None - possible when called via get_volatility()
        with pytest.raises(ValueError, match="ex_date should be a date/*"):
            exercise_date = None
            _ = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

    def test_kirk_spread_vol(self):
        pricing_date = date(2013, 2, 15)
        exercise_date = date(2014, 2, 15)
        nbp_price = 50 * PENCE / THERM
        mkt_env = create_mkt_env(pricing_date)
        vol_model = VcMktDynamic(mkt_env, {})
        vol_corr_adaptor = DatedVolCorrAdaptor(vol_model.vol_corr, mkt_env.pricing_date)

        swap_rates = [SwapRate(NBP, month(2014, 3)), SwapRate(NBP, month(2014, 12))]
        weights = [1, -1]
        strike = 1 * PENCE / THERM

        st_vol = vol_model.get_volatility(NBP, month(2014, 3), exercise_date, strike)
        lt_vol = vol_model.get_volatility(NBP, month(2014, 12), exercise_date, strike)
        correlation = vol_corr_adaptor.calc_corr(swap_rates[0], swap_rates[1], exercise_date)
        b = nbp_price / (nbp_price + strike)

        expected_vol = math.sqrt(st_vol ** 2 + lt_vol ** 2 * b ** 2 - 2 * correlation * st_vol * lt_vol * b)
        basket_vol = VcBasketVol(swap_rates, weights, exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))
        actual_vol = basket_vol.kirk.spread_vol
        assert pytest.approx(expected_vol) == actual_vol


class TestBasketSpreadOption:

    def setup_method(self):
        self.forwards = [100 * PENCE / THERM, 80 * PENCE / THERM]
        self.pricing_date = date(2016, 1, 1)
        self.mkt_env = self._build_mkt_env(self.pricing_date)
        self.swap_rates = [SwapRate(NBP, month(2016, 4)), SwapRate(NBP, month(2016, 6))]
        self.exercise_date = date(2016, 3, 24)

    @staticmethod
    def _build_mkt_env(pricing_date=None, eod_flag=True, mean_rev_procs=mean_rev_procs):
        pricing_date = date(2016, 1, 1) if pricing_date is None else pricing_date
        quotes = {
            IrPivotCurrency: GBP,
            SwapRate(NBP, month(2016, 4)): 100 * PENCE / THERM,
            SwapRate(NBP, month(2016, 5)): 90 * PENCE / THERM,
            SwapRate(NBP, month(2016, 6)): 80 * PENCE / THERM,
            SwapRate(UKPOWER_PEAK, month(2016, 4)): 110 * GBP / MWHP,
            SwapRate(UKPOWER, month(2016, 4)): 80 * GBP / MWHP,
            SwapRate(GBPIR, date(2014, 1, 1)): 0.03,
            SwapRate(GBPIR, date(2017, 1, 1)): 0.03,
            PricingDate: pricing_date,
            CurveDate: pricing_date,
            EndOfDayKey: eod_flag,
            MeanRevertingProcessesKey: mean_rev_procs,
            VolSurfaceKey(NBP): PillaredSmileCollectionFactory.build_nbp_2016(),
        }
        return MktEnv(quotes)

    def _create_basket_vol(self, weights, strike, swap_rates=None, **kwargs):
        mkt_env = self._build_mkt_env(**kwargs)
        sr = swap_rates or [
            SwapRate(NBP, month(2016, 4)),
            SwapRate(NBP, month(2016, 6)),
        ]
        return VcBasketVol(sr, weights, self.exercise_date, strike, GBP, VcMktDynamic(mkt_env, {}))

    def test_kirk(self):
        weights = [1 * THERM, -1 * THERM]
        strike = sum(w * f for w, f in zip(weights, self.forwards))
        basket_vol = self._create_basket_vol(weights, strike)
        price = basket_vol.price_option(OptionType.CALL)

        te = time_between(self.mkt_env.pricing_date, self.exercise_date)
        vol_corr_adaptor = DatedVolCorrAdaptor(self.mkt_env.vol_corr, self.mkt_env.pricing_date)
        vols = [vol_corr_adaptor.calc_vol(sr, self.exercise_date) for sr in self.swap_rates]
        corr = vol_corr_adaptor.calc_corr(self.swap_rates[0], self.swap_rates[1], self.exercise_date)
        kirk_pricer = KirkSpreadOption(s1=100,
                                       s2=80,
                                       k=value(strike),
                                       t=te,
                                       vol1=vols[0],
                                       vol2=vols[1],
                                       corr12=corr)
        ref_price = kirk_pricer.value

        assert_almost_equal(ref_price[0], value(price))

    def test_option_on_basket(self):
        """
        Compare the generic spread option pricer to an option on (S1 + S2)
        Where the price is computed by matching the expectation and variance of (S1 + S2) with a log-normal variable
        """
        weights = [2 * THERM, 1 * THERM]
        weighted_forwards = [w * f for w, f in zip(weights, self.forwards)]
        strike = sum(weighted_forwards)
        basket_vol = self._create_basket_vol(weights, strike)
        price = basket_vol.price_option(OptionType.CALL)

        te = time_between(self.mkt_env.pricing_date, self.exercise_date)
        vol_corr_adaptor = DatedVolCorrAdaptor(self.mkt_env.vol_corr, self.pricing_date)
        vols = [vol_corr_adaptor.calc_vol(sr, self.exercise_date) for sr in self.swap_rates]
        corr = vol_corr_adaptor.calc_corr(self.swap_rates[0], self.swap_rates[1], self.exercise_date)
        expected_vol = (
            weighted_forwards[0] ** 2 * math.exp(vols[0] ** 2 * te)
            + 2.0 * (weighted_forwards[0] * weighted_forwards[1]) * math.exp(vols[0] * vols[1] * corr * te)
            + weighted_forwards[1] ** 2 * math.exp((vols[1] ** 2) * te)
        )
        expected_vol = math.sqrt(math.log(expected_vol / (weighted_forwards[0] + weighted_forwards[1]) ** 2) / te)
        expected_price = Black76Option(s=strike, k=strike, te=te, sigma=expected_vol).value

        assert_almost_equal(price, expected_price)
        old_option = BasketSpreadOption(np.array([100, 80]), np.array([2, 1]), 280, te, np.array(vols),
                                        np.array([[1, corr], [corr, 1]]))
        assert_almost_equal(value(price), old_option.value[0])

    def test_basket_put_call_parity(self):
        """
        Compare the generic spread option pricer to an option on (S1 + S2)
        Where the price is computed by matching the expectation and variance of (S1 + S2) with a log-normal variable
        """
        weights = [2 * THERM, 1 * THERM]
        weighted_forwards = [w * f for w, f in zip(weights, self.forwards)]
        fwd = sum(weighted_forwards)
        for k in (-1, 0, 0.9, 1, 1.1):
            strike = k * fwd
            basket_vol = self._create_basket_vol(weights, strike)
            call_price = basket_vol.price_option(OptionType.CALL)
            put_price = basket_vol.price_option(OptionType.PUT)
            assert_almost_equal(call_price, put_price + fwd - strike)

    def test_get_one_optimal_strike(self):
        weights = [1]
        swap_rates = [
            SwapRate(NBP, month(2016, 4)),
        ]
        strike = 100 * PENCE / THERM
        mkt_env = self._build_mkt_env()
        lvc_basket_vol = LvcBasketVol(swap_rates, weights, self.exercise_date, strike, GBP,
                                      LvcMktDynamic(mkt_env, {}))
        expected = [strike]
        actual = lvc_basket_vol._get_optimal_strikes(strike, POS)
        assert_almost_equal(expected, actual)

    def test_get_optimal_strikes_atm(self):
        weights = [1, 2]
        swap_rates = [
            SwapRate(NBP, month(2016, 4)),
            SwapRate(NBP, month(2016, 6)),
        ]
        strike = 1 * 100 * PENCE / THERM + 2 * 80 * PENCE / THERM
        mkt_env = self._build_mkt_env()
        lvc_basket_vol = LvcBasketVol(swap_rates, weights, self.exercise_date, strike, GBP,
                                      LvcMktDynamic(mkt_env, {}))
        expected = [100 * PENCE / THERM, 80 * PENCE / THERM]
        actual = lvc_basket_vol._get_optimal_strikes(strike, POS)
        assert_almost_equal(expected, actual)

    def test_get_optimal_strikes(self):
        mkt_env = self._build_mkt_env()
        mkt_dyn = LvcMktDynamic(mkt_env, {}, mkt_env.ir_pivot_currency)
        weights = [1, 2]
        apr_sr = SwapRate(NBP, month(2016, 4))
        jun_sr = SwapRate(NBP, month(2016, 6))
        swap_rates = [apr_sr, jun_sr]

        strike = (
            1.05 * mkt_dyn.get_price(apr_sr.underlying, apr_sr.time_period)
            + 2.1 * mkt_dyn.get_price(jun_sr.underlying, jun_sr.time_period)
        )
        lvc_basket_vol = LvcBasketVol(swap_rates, weights, self.exercise_date, strike, GBP, mkt_dyn)

        # april vol is higher than jun vol, hence the higher expected move on apr
        # in fact the ratio of the log ratios should match the vols ratio
        apr_ratio, jun_ratio = 1.06014895, 1.04400300
        apr_price = mkt_dyn.get_price(apr_sr.underlying, apr_sr.time_period)
        jun_price = mkt_dyn.get_price(jun_sr.underlying, jun_sr.time_period)
        actual_apr, actual_jun = lvc_basket_vol._get_optimal_strikes(strike, POS)
        assert_almost_equal(apr_ratio, actual_apr / apr_price)
        assert_almost_equal(jun_ratio, actual_jun / jun_price)

        adaptor = DatedVolCorrAdaptor(mkt_dyn.vol_corr, mkt_dyn.pricing_date)
        apr_vol, jun_vol = (adaptor.calc_vol(sr, self.exercise_date) for sr in swap_rates)
        assert_almost_equal(apr_vol / jun_vol, math.log(apr_ratio) / math.log(jun_ratio), atol=1e-6)

    def test_option_on_lvc_basket(self):
        """
        Compare the generic spread option pricer to an option on (S1 + S2)
        Where the price is computed by matching the expectation and variance of (S1 + S2) with a log-normal variable
        """
        weights = [1 * THERM, 1.5 * THERM]
        weighted_forwards = [w * f for w, f in zip(weights, self.forwards)]
        strike = sum(weighted_forwards)
        mkt_env = self._build_mkt_env()
        sr = [
            SwapRate(NBP, month(2016, 4)),
            SwapRate(NBP, month(2016, 6)),
        ]
        path_count = 100000
        mkt_dyn = LvcMktDynamic(mkt_env, minimal_static_data(path_count, 1), GBP)
        mc_fwd = array(np.zeros(path_count), PENCE)
        for weight, swap_rate in zip(weights, sr):
            mc_fwd += weight * mkt_dyn.get_price(
                swap_rate.underlying, swap_rate.time_period, override_valuation_date=self.exercise_date
            )

        tol = 3.0
        lvc_basket_vol = LvcBasketVol(sr, weights, self.exercise_date, strike, GBP, LvcMktDynamic(mkt_env, {}))
        ivc_basket_vol = VcBasketVol(sr, weights, self.exercise_date, strike, GBP, IvcMktDynamic(mkt_env, {}))
        mc_price = maximum(0, mc_fwd - strike).mean()
        mc_std_err = maximum(0, mc_fwd - strike).std_err
        analytic_price = lvc_basket_vol.price_option(OptionType.CALL)
        assert_almost_equal(analytic_price, ivc_basket_vol.price_option(OptionType.CALL))
        assert mc_price - tol * mc_std_err < analytic_price < mc_price + tol * mc_std_err

        for k in [0.85, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.15]:
            lvc_basket_vol = LvcBasketVol(sr, weights, self.exercise_date, k * strike, GBP, LvcMktDynamic(mkt_env, {}))
            mc_price = maximum(0, mc_fwd - k * strike).mean()
            mc_std_err = maximum(0, mc_fwd - k * strike).std_err
            analytic_price = lvc_basket_vol.price_option(OptionType.CALL)
            assert mc_price - tol * mc_std_err < analytic_price < mc_price + tol * mc_std_err

        vol_model = LvcMktDynamic(mkt_env, {})
        terms = BasketOptionParams(sr, weights, -0.1 * strike, OptionType.CALL, self.exercise_date, GBP)
        analytic_pv = vol_model.price_undiscounted_basket(terms)
        mc_price = maximum(0, mc_fwd + 0.1 * strike).mean()
        mc_std_err = maximum(0, mc_fwd + 0.1 * strike).std_err
        assert mc_price - tol * mc_std_err < analytic_pv < mc_price + tol * mc_std_err

    def test_negative_strike(self):
        weights = [-1 * THERM, 1 * THERM]
        weighted_forwards = [w * f for w, f in zip(weights, self.forwards)]
        strike = sum(weighted_forwards) - 5 * PENCE
        basket_vol = self._create_basket_vol(weights, strike)
        price = basket_vol.price_option(OptionType.CALL)

        neg_weights = [-w for w in weights]
        baseline = sum(w * f for w, f in zip(weights, self.forwards)) - strike
        basket_vol = self._create_basket_vol(neg_weights, -strike)
        expected_price = basket_vol.price_option(OptionType.CALL) + baseline
        assert_almost_equal(price / expected_price, 1.00105677)

        vol_corr_adaptor = DatedVolCorrAdaptor(self.mkt_env.vol_corr, self.pricing_date)
        vols = [vol_corr_adaptor.calc_vol(sr, self.exercise_date) for sr in self.swap_rates]
        corr = vol_corr_adaptor.calc_corr(self.swap_rates[0], self.swap_rates[1], self.exercise_date)
        te = time_between(self.mkt_env.pricing_date, self.exercise_date)
        old_option = BasketSpreadOption(np.array([100, 80]), np.array([-1, 1]), -25, te, np.array(vols),
                                        np.array([[1, corr], [corr, 1]]))
        assert_almost_equal(value(price) / old_option.value[0], 1.00105677)

    def test_basket_spread_option(self):
        """
        Compare the generic spread option pricer to an option on (w1 S1 + w2 S2)
        Where the price is computed by matching the expectation
        and variance of (w1 S1 + w2 S2) with a log-normal variable
        """
        forwards = [110 * GBP / MWHP, 100 * PENCE / THERM]
        weights = [1 * MWHP, -2 * MWHG]
        weighted_forwards = [w * f for w, f in zip(weights, forwards)]
        strike = sum(weighted_forwards)
        swap_rates = [SwapRate(UKPOWER_PEAK, month(2016, 4)), SwapRate(NBP, month(2016, 4))]
        basket_vol = self._create_basket_vol(weights, strike, swap_rates)
        price = basket_vol.price_option(OptionType.CALL)

        te = time_between(self.mkt_env.pricing_date, self.exercise_date)
        vol_corr_adaptor = DatedVolCorrAdaptor(self.mkt_env.vol_corr, self.pricing_date)
        vols = [vol_corr_adaptor.calc_vol(sr, self.exercise_date) for sr in swap_rates]
        corr = vol_corr_adaptor.calc_corr(swap_rates[0], swap_rates[1], self.exercise_date)
        b = -weighted_forwards[1] / (-weighted_forwards[1] + strike)
        expected_vol = np.sqrt(vols[0] ** 2 + (vols[1] ** 2 * b ** 2) - (2 * corr * vols[0] * vols[1] * b))
        strike += -weighted_forwards[1]
        expected_price = Black76Option(s=strike, k=strike, te=te, sigma=expected_vol).value

        assert_almost_equal(price, expected_price)

    def test_atm_put_call_parity(self):
        weights = [1 * THERM, -1 * THERM]
        strike = sum(w * f for w, f in zip(weights, self.forwards))
        basket_vol = self._create_basket_vol(weights, strike)
        atm_call_price = basket_vol.price_option(OptionType.CALL)
        atm_put_price = basket_vol.price_option(OptionType.PUT)
        assert_almost_equal(atm_call_price, atm_put_price)

    def test_put_call_parity(self):
        weights = [1 * THERM, -1 * THERM]
        strike = 0 * PENCE
        fwd = sum(w * f for w, f in zip(weights, self.forwards))
        basket_vol = self._create_basket_vol(weights, strike)
        call_price = basket_vol.price_option(OptionType.CALL)
        put_price = basket_vol.price_option(OptionType.PUT)
        assert_almost_equal(call_price, put_price + fwd)

    def test_ditm(self):
        weights = [1 * THERM, 0 * THERM]
        strike = 1e-6 * PENCE
        fwd = sum(w * f for w, f in zip(weights, self.forwards))
        basket_vol = self._create_basket_vol(weights, strike)
        price = basket_vol.price_option(OptionType.CALL)
        assert_almost_equal(price + strike, fwd)

    def test_itm_call_on_exercise(self):
        weights = [1 * THERM, -1 * THERM]
        strike = 18 * PENCE
        basket_vol = self._create_basket_vol(weights, strike, pricing_date=date(2016, 3, 24), eod_flag=False)
        call_price = basket_vol.price_option(OptionType.CALL)
        assert_almost_equal(call_price, 2 * PENCE)

    def test_otm_put_on_exercise(self):
        weights = [1 * THERM, -1 * THERM]
        strike = 18 * PENCE
        basket_vol = self._create_basket_vol(weights, strike, pricing_date=date(2016, 3, 24), eod_flag=False)
        put_price = basket_vol.price_option(OptionType.PUT)
        assert_almost_equal(put_price, 0 * PENCE)

    def test_otm_call_on_exercise(self):
        strike = 21 * PENCE
        weights = [1 * THERM, -1 * THERM]
        basket_vol = self._create_basket_vol(weights, strike, pricing_date=date(2016, 3, 24), eod_flag=False)
        call_price = basket_vol.price_option(OptionType.CALL)
        assert_almost_equal(call_price, 0 * PENCE)

    def test_itm_put_on_exercise(self):
        strike = 21 * PENCE
        weights = [1 * THERM, -1 * THERM]
        basket_vol = self._create_basket_vol(weights, strike, pricing_date=date(2016, 3, 24), eod_flag=False)
        put_price = basket_vol.price_option(OptionType.PUT)
        assert_almost_equal(put_price, 1 * PENCE)


class TestFxBasketOption:

    def test_compo_spread(self):
        """
        Value on option on the difference between NBP and TTF converted into GBP
        Check the analyical pricing against monte carlo
        """
        pricing_date = date(2016, 1, 1)
        quotes = {
            IrPivotCurrency: GBP,
            SwapRate(NBP, month(2016, 6)): 80 * PENCE / THERM,
            SwapRate(TTF, month(2016, 6)): 21.84 * EUR / MWHG,
            SwapRate(GBPIR, date(2014, 1, 1)): 0.00,
            SwapRate(GBPIR, date(2017, 1, 1)): 0.00,
            SwapRate(GBPEUR, date(2014, 1, 1)): 0.8 * EUR / GBP,
            SwapRate(GBPEUR, date(2017, 1, 1)): 0.8 * EUR / GBP,
            PricingDate: pricing_date,
            CurveDate: pricing_date,
            EndOfDayKey: True,
            MeanRevertingProcessesKey: nbpttf_mean_rev_procs
        }
        mkt_env = MktEnv(quotes)
        static_data = {PathCountKey: 10000, SeedKey: 1}
        delivery_date = date(2016, 6, 2)
        exercise_date = date(2016, 6, 1)

        for mkt_dynamic in [
            IntrinsicMktDynamic(mkt_env, static_data, GBP),
            VcMktDynamic(mkt_env, static_data, GBP),
        ]:
            fwd_eurgbp_price = mkt_dynamic.get_price(EURGBP, delivery_date)

            def option_params(option_type):
                return BasketOptionParams(
                    swap_rates=[SwapRate(NBP, delivery_date), SwapRate(TTF, delivery_date)],
                    weights=[1, -fwd_eurgbp_price],
                    strike=0 * GBP / MWHG,
                    option_type=option_type,
                    exercise_date=exercise_date
                )
            # Price with VolModel
            call_terms = option_params(OptionType.CALL)
            put_terms = option_params(OptionType.PUT)
            call_pv = mkt_dynamic.price_undiscounted_basket(call_terms)
            put_pv = mkt_dynamic.price_undiscounted_basket(put_terms)
            analytic_pv = call_pv + put_pv

            # Price with MC
            eurgbp_price = mkt_dynamic.get_price(EURGBP, delivery_date, override_valuation_date=exercise_date)
            nbp_price = mkt_dynamic.get_price(NBP, delivery_date, override_valuation_date=exercise_date)
            ttf_price = mkt_dynamic.get_price(TTF, delivery_date, override_valuation_date=exercise_date)
            ttf_price_in_gbp = ttf_price * eurgbp_price
            spread_pv = nbp_price - ttf_price_in_gbp
            mc_pv = maximum(spread_pv, -spread_pv)

            assert mc_pv.mean() - mc_pv.semi95_confidence_interval <= analytic_pv
            assert analytic_pv <= mc_pv.mean() + mc_pv.semi95_confidence_interval


class TestLvcSpread:

    def setup_method(self):
        self.pricing_date = date(2016, 1, 1)
        psc = PillaredSmileCollectionFactory.build_nbp_2016()
        mrp = MeanRevertingProcessFactory.build(underlying=[PSV, PSV, TTF, TTF])
        self.dp = month(2016, 6)
        self.psv_sr = SwapRate(PSV, self.dp)
        self.ttf_sr = SwapRate(TTF, self.dp)
        quotes = {
            IrPivotCurrency: EUR,
            self.psv_sr: 22.0 * EUR / MWH,
            SwapRate(PSV, month(2016, 7)): 22.0 * EUR / MWH,
            self.ttf_sr: 20.0 * EUR / MWH,
            SwapRate(TTF, month(2016, 7)): 22.0 * EUR / MWH,
            SwapRate(EURIR, date(2014, 1, 1)): 0.00,
            SwapRate(EURIR, date(2017, 1, 1)): 0.00,
            PricingDate: self.pricing_date,
            CurveDate: self.pricing_date,
            EndOfDayKey: True,
            MeanRevertingProcessesKey: mrp,
            VolSurfaceKey(PSV): psc,
            VolSurfaceKey(TTF): psc,
        }
        mkt_env = MktEnv(quotes)
        static_data = {PathCountKey: 10000, SeedKey: 1}
        self.mkt_dyn = LvcMktDynamic(mkt_env, static_data)
        self.ex_date = date(2016, 6, 1)
        self.basket_vol = LvcBasketVol(
            [self.psv_sr, SwapRate(PSV, month(2016, 7)), self.ttf_sr, SwapRate(TTF, month(2016, 7))],
            [1, 0.001, -1, -0.001], self.ex_date, 1 * EUR / MWH, EUR, self.mkt_dyn)

    def test_correlation(self):
        """ check that basket._correlation matches volcorr's"""
        expected = DatedVolCorrAdaptor(self.mkt_dyn.vol_corr, self.pricing_date)\
            .calc_corr(self.psv_sr, self.ttf_sr, self.ex_date)
        assert_almost_equal(expected, self.basket_vol._correlation, atol=1e-4)

    @pytest.mark.parametrize("ul, sign", [(PSV, POS), (TTF, NEG)])
    def test_surface(self, ul, sign):
        """ check that basket._surface matches the psc"""
        expected = self.mkt_dyn.sim_cache._surface(self.dp, ul)
        assert_almost_equal(expected.phi2.eta, self.basket_vol._surface(sign).phi2.eta, atol=1e-4)
        assert_almost_equal(expected.phi2.nu, self.basket_vol._surface(sign).phi2.nu, atol=1e-4)


