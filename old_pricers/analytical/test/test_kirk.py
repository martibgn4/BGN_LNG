import math
import pytest
from math import log
from unittest.mock import patch, PropertyMock

from pytest_quants import assert_almost_equal
from quantity import MWH
from trade_data import OPTION_TYPES, OptionType
from underlying import EUR

from thorn.core.market.volatility.lvc.phi import Phi2
from thorn.core.market.volatility.lvc.ssvi_vol_surface import SSVIBasketVolSurface
from thorn.core.pricers.analytical.european_option import Black76Option
from thorn.core.pricers.analytical.kirk import GaussianCopulaSpreadPricer, KirkSpreadPricer


class TestGaussianCopulaPricer:
    """ Test properties of the GaussianCopula Spread Pricer
        with a fwd price of 50, we expect a tol of 1e-3 at least around the money
        that should best any (feasible) mc valuation
    """

    def setup_method(self):
        self.unit = EUR / MWH
        self.pos_fwd = 50 * self.unit
        self.neg_fwd = -50 * self.unit
        self.pos_surface = SSVIBasketVolSurface(atm_vol_func=lambda x: 0.5, phi2=Phi2(0., 1.9))
        self.pos_flat_surface = SSVIBasketVolSurface(atm_vol_func=lambda x: 0.5, phi2=Phi2(0., 0.))
        self.neg_surface = SSVIBasketVolSurface(atm_vol_func=lambda x: 0.4, phi2=Phi2(0.9, 1.9))
        self.neg_flat_surface = SSVIBasketVolSurface(atm_vol_func=lambda x: 0.4, phi2=Phi2(0., 0.))
        self.correlation = 0.75
        self.te = 1.0

    def _gc(self, strike):
        return GaussianCopulaSpreadPricer(
            self.pos_fwd, self.neg_fwd, strike, self.pos_surface, self.neg_surface, self.correlation, self.te
        )

    def _flat_gc(self, strike):
        return GaussianCopulaSpreadPricer(
            self.pos_fwd, self.neg_fwd, strike, self.pos_flat_surface, self.neg_flat_surface, self.correlation, self.te
        )

    def _kirk(self, strike):
        return KirkSpreadPricer(
            self.pos_fwd, self.neg_fwd, strike, 0.5, 0.4, self.correlation, self.te
        )

    @pytest.mark.parametrize("ot", OPTION_TYPES)
    @pytest.mark.parametrize("intrinsic", (False, True))
    @pytest.mark.parametrize(
        "strike, tol",
        [(-100, 2), (-10, 2), (-5, 2), (-1, 3), (0, 4), (1, 3), (5, 2), (10, 2), (100, 1)],
    )
    def test_flat_gc_vs_kirk(self, ot, intrinsic, strike, tol):
        """ check that kirk and gc converge in absence of smile"""
        # tolerance should be lower in the wings where kirk under-performs
        flat_gc = self._flat_gc(strike * self.unit)
        kirk = self._kirk(strike * self.unit)
        assert_almost_equal(flat_gc.option_value(ot, intrinsic), kirk.option_value(ot, intrinsic), atol=1/10**tol)

    @pytest.mark.parametrize("ot", OPTION_TYPES)
    @pytest.mark.parametrize("strike", (-100, -5, 0, 5, 10, 100))
    def test_gc_vs_flat(self, ot, strike):
        """ gc vol should be higher than flat one because convexity is positive"""
        assert self.pos_surface.phi2.eta > 0
        tol = 1e-3 * self.unit
        flat_gc = self._flat_gc(strike * self.unit)
        gc = self._gc(strike * self.unit)
        assert gc.option_value(ot) + tol >= flat_gc.option_value(ot)

    @pytest.mark.parametrize("ot", OPTION_TYPES)
    @pytest.mark.parametrize("k", (30, 40, 50, 80, 110))
    def test_degenerate_to_european(self, ot, k):
        """ if second leg is zero, should degenerate to a european """
        strike = k * self.unit
        gc = GaussianCopulaSpreadPricer(self.pos_fwd, -1e-6 * self.pos_fwd, strike, self.pos_surface,
                                        self.pos_surface, self.correlation, self.te)
        vol = self.pos_surface.get_implied_vol(self.te, log(strike / self.pos_fwd)) if strike > 0 else 0
        bs = Black76Option(self.pos_fwd, strike, vol, self.te, ot)
        assert_almost_equal(gc.option_value(ot), bs.value, atol=1e-3)


class TestKirkSpreadPricer:
    def setup_method(self):
        self.option = KirkSpreadPricer(
            pos_fwd=109.998, neg_fwd=-100, strike=5, pos_vol=0.10, neg_vol=0.15, corr=0.3, te=1, option_type=OptionType.CALL
        )
        self.put_option = KirkSpreadPricer(
            pos_fwd=109.998, neg_fwd=-100, strike=5, pos_vol=0.10, neg_vol=0.15, corr=0.3, te=1, option_type=OptionType.PUT
        )

    def test_delta_call(self):
        value = self.option.option_value()
        delta = self.option.delta()
        shock = 1e-6
        self.option.pos_fwd += shock
        up_value = self.option.option_value()
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-4)

    def test_delta_put(self):
        value = self.put_option.option_value()
        delta = self.put_option.delta()
        shock = 1e-6
        self.put_option.pos_fwd += shock
        up_value = self.put_option.option_value()
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-4)

        call_delta = self.option.delta()
        adj = math.exp(-self.option._option(OptionType.CALL, False).r * self.option._option(OptionType.CALL, False).tm)
        assert pytest.approx(delta) == call_delta - adj

    def test_delta_neg_call(self):
        value = self.option._option_neg(None, False).value
        delta = self.option.delta_neg_leg()
        shock = 1e-6
        self.option.neg_fwd -= shock
        up_value = self.option._option_neg(None, False).value
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-3)

    def test_delta_neg_put(self):
        value = self.put_option._option_neg(None, False).value
        delta = self.put_option.delta_neg_leg()
        shock = 1e-6
        self.put_option.neg_fwd -= shock
        up_value = self.put_option._option_neg(None, False).value
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-4)

        call_delta_neg = self.option.delta_neg_leg()
        adj = math.exp(-self.option._option(OptionType.CALL, False).r * self.option._option(OptionType.CALL, False).tm)
        assert pytest.approx(delta) == call_delta_neg + adj

    def test_gamma(self):
        delta = self.option.delta()
        gamma = self.option.gamma()
        shock = 1e-6
        self.option.pos_fwd += shock
        up_delta = self.option.delta()
        fd_gamma = (up_delta - delta) / shock
        assert_almost_equal(gamma, fd_gamma)

    def test_gamma_neg(self):
        delta = self.option.delta_neg_leg()
        gamma = self.option.gamma_neg_leg()
        shock = 1e-6
        self.option.neg_fwd -= shock
        up_delta = self.option.delta_neg_leg()
        fd_gamma = (up_delta - delta) / shock
        assert_almost_equal(gamma, fd_gamma)

    def test_vega(self):
        value = self.option.option_value()
        vega = self.option.vega()
        shock = 1e-6
        spread_vol = self.option.spread_vol
        with patch.object(KirkSpreadPricer, 'spread_vol', new_callable=PropertyMock) as m_spread_vol:
            m_spread_vol.return_value = spread_vol + shock
            up_value = self.option.option_value()
        fd_vega = (up_value - value) / shock
        assert_almost_equal(vega, fd_vega, atol=1e-4)

    def test_vega_pos_leg(self):
        value = self.option.option_value()
        vega = self.option.vega_pos_leg()
        shock = 1e-6
        self.option.pos_vol += shock
        up_value = self.option.option_value()
        fd_vega = (up_value - value) / shock
        assert_almost_equal(vega, fd_vega, atol=5e-4)

    def test_vega_neg_leg(self):
        value = self.option.option_value()
        vega = self.option.vega_neg_leg()
        shock = 1e-6
        self.option._starting_neg_vol += shock
        self.option.neg_vol = self.option._adjust_neg_vol(
            self.option.neg_fwd, self.option._starting_neg_vol, self.option.strike
        )
        up_value = self.option.option_value()
        fd_vega = (up_value - value) / shock
        assert_almost_equal(vega, fd_vega, atol=1e-4)

    def test_theta(self):
        value = self.option.option_value()
        theta = self.option.theta()
        shock = 1e-6
        self.option.te -= shock
        new_value = self.option.option_value()
        fd_theta = (new_value - value) / shock
        assert_almost_equal(theta, fd_theta, atol=1e-4)


