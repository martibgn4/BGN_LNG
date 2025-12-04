import math

from pytest_quants import assert_almost_equal
from thorn.core.pricers.analytical.european_option import Black76Option
from thorn.core.pricers.analytical.test.fixtures.kirk_spread_option import KirkSpreadOption


class TestKirkSpreadOption:

    def test_margrabe(self):
        s1 = 50.0
        s2 = 30.0
        k = 0.0
        t = 1.0
        vol1 = 0.5
        vol2 = 0.5
        corr12 = 0.5
        kirk_pricer = KirkSpreadOption(s1, s2, k, t, vol1, vol2, corr12)
        price = kirk_pricer.value
        margrabe_price = Black76Option(s=s1,
                                       k=s2,
                                       sigma=math.sqrt(vol1 ** 2 + vol2 ** 2 - 2 * corr12 * vol1 * vol2),
                                       te=t).value

        assert_almost_equal(margrabe_price, price)

    def test_kirk_strike(self):
        """
        Weak test
        Test that the strike is taken into account - we don't have a ref pricer, so we check that:
        P(strike = 0) - strike < P(strike) < P(strike = 0)
        """
        s1 = 50.0
        s2 = 30.0
        k = 0.0
        t = 1.0
        vol1 = 0.5
        vol2 = 0.5
        corr12 = 0.5
        kirk_pricer = KirkSpreadOption(s1, s2, k, t, vol1, vol2, corr12)
        price = kirk_pricer.value

        k2 = 1.0
        kirk_pricer_with_strike = KirkSpreadOption(s1, s2, k2, t, vol1, vol2, corr12)
        price_with_strike = kirk_pricer_with_strike.value

        assert price_with_strike < price
        assert price_with_strike > price - k2


