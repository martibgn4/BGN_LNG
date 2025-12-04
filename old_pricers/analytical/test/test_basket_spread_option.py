import math

from numpy import array

from pytest_quants import assert_almost_equal
from thorn.core.pricers.analytical.european_option import Black76Option
from thorn.core.pricers.analytical.test.fixtures.basket_spread_option import BasketSpreadOption
from thorn.core.pricers.analytical.test.fixtures.kirk_spread_option import KirkSpreadOption
from trade_data import OptionType


class TestBasketSpreadOption:

    def setup_method(self):
        self.forwards = array([100, 80])
        self.vols = array([0.3, 0.4])
        self.corrs = array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])

    def test_kirk(self):
        ttm = 1.0 - 1.0 / 365.0
        weights = [1, -1]
        strike = self.forwards[0] - self.forwards[1]

        option = BasketSpreadOption(self.forwards, weights, strike, ttm, self.vols, self.corrs)
        price = option.value

        kirk_pricer = KirkSpreadOption(s1=self.forwards[0],
                                       s2=self.forwards[1],
                                       k=strike,
                                       t=ttm,
                                       vol1=self.vols[0],
                                       vol2=self.vols[1],
                                       corr12=self.corrs[0, 1])
        ref_price = kirk_pricer.value

        assert_almost_equal(ref_price, price)

    def test_option_on_basket(self):
        """
        Compare the generic spread option pricer to an option on (S1 + S2)
        Where the price is computed by matching the expectation and variance of (S1 + S2) with a log-normal variable
        """
        ttm = 1.0
        weights = [1, 1]
        strike = self.forwards[0] + self.forwards[1]

        option = BasketSpreadOption(self.forwards, weights, strike, ttm, self.vols, self.corrs)
        price = option.value

        expected_vol = self.forwards[0] ** 2 * math.exp(self.vols[0] ** 2)\
            + 2.0 * (self.forwards[0] * self.forwards[1]) * math.exp(self.vols[0] * self.vols[1] * self.corrs[0, 1])\
            + self.forwards[1] ** 2 * math.exp(self.vols[1] ** 2)
        expected_vol = math.sqrt(math.log(expected_vol / (self.forwards[0] + self.forwards[1]) ** 2))
        expected_price = Black76Option(s=strike, k=strike, te=ttm, sigma=expected_vol).value
        assert_almost_equal(price, expected_price)

    def test_option_on_basket_with_weights(self):
        """
        Compare the generic spread option pricer to an option on (w1 S1 + w2 S2)
        Where the price is computed by matching the expectation
        and variance of (w1 S1 + w2 S2) with a log-normal variable
        """
        ttm = 2.0 - 1.0 / 365.0
        weights = array([0.5, 1])

        weighted_forwards = weights * self.forwards
        strike = weighted_forwards[0] + weighted_forwards[1]
        option = BasketSpreadOption(self.forwards, weights, strike, ttm, self.vols, self.corrs)
        price = option.value

        expected_vol = (
            weighted_forwards[0] ** 2 * math.exp(self.vols[0] ** 2 * ttm)
            + 2.0 * weighted_forwards[0] * weighted_forwards[1] * math.exp(
                self.vols[0] * self.vols[1] * self.corrs[0, 1] * ttm
            ) + weighted_forwards[1] ** 2 * math.exp(self.vols[1] ** 2 * ttm)
        )

        expected_vol = math.sqrt(math.log(expected_vol / (weighted_forwards[0] + weighted_forwards[1]) ** 2) / ttm)
        expected_price = Black76Option(s=strike, k=strike, te=ttm, sigma=expected_vol).value
        assert_almost_equal(price, expected_price)

    def test_atm_put_call_parity(self):
        weights = array([1, -1])
        strike = sum(self.forwards * weights)
        te = 2
        atm_call_option = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs)
        atm_call_price = atm_call_option.value

        atm_put_option = BasketSpreadOption(self.forwards, -weights, -strike, te, self.vols, self.corrs)
        atm_put_price = atm_put_option.value

        assert_almost_equal(atm_call_price, atm_put_price)

    def test_put_call_parity(self):
        te = 0.5
        weights = array([1, -1])
        weighted_forwards = weights * self.forwards
        strike = 0
        spread_option_args = (self.forwards, weights, strike, te, self.vols, self.corrs)
        call_option = BasketSpreadOption(*spread_option_args)
        call_price = call_option.value
        put_option = BasketSpreadOption(*spread_option_args, put_call=OptionType.PUT)
        put_price = put_option.value
        assert_almost_equal(call_price, put_price + sum(weighted_forwards))

    def test_ditm(self):
        weights = array([1, 0])
        strike = 1e-6
        te = 2
        call_option = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs)
        call_mtm = call_option.value
        assert_almost_equal(call_mtm + strike, self.forwards[0])

    def test_valuation_on_exercise(self):
        te = 0.0
        weights = array([1, -1])

        strike = 18
        call_price = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs).value
        assert call_price == 2
        put_price = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs, OptionType.PUT).value
        assert put_price == 0

        strike = 21
        call_price = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs).value
        assert call_price == 0
        put_price = BasketSpreadOption(self.forwards, weights, strike, te, self.vols, self.corrs, OptionType.PUT).value
        assert put_price == 1

    def test_multipath_forwards(self):
        """
        Test that we can use numpy vectorisation to compute the price in relation to multiple forwards
        simultaneously
        """
        te = 1.0
        weights = [1, 1]
        strike = 180
        # forwards for 3 different paths
        multipath_forwards = array([[100, 99, 98],
                                    [80, 79, 78]])

        def price_option(forwards):
            return BasketSpreadOption(forwards, weights, strike, te, self.vols, self.corrs).value

        multipath_prices = price_option(multipath_forwards)

        for path in range(3):
            forwards_per_path = [f[path] for f in multipath_forwards]
            price_per_path = price_option(forwards_per_path)
            assert multipath_prices[path] == price_per_path


