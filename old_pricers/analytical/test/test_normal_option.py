import numpy as np

from pytest_quants import assert_almost_equal, repeat
from quantity import PENCE, THERM
from thorn.core.pricers.analytical.normal_option import BachelierOption
from trade_data import OptionType


class TestBachelierOption:

    def setup_method(self):
        # First example from Haug, Option Pricing Formulas, as B76 option
        self.atm_option = BachelierOption(s=19, k=19, sigma=2.0, te=0.75, tm=1.0, r=0.1)
        self.option = BachelierOption(s=18, k=19, sigma=2.0, te=0.75, tm=1.0, r=0.1)
        self.put_option = BachelierOption(call_put=OptionType.PUT, s=18, k=19, sigma=2.0, te=0.75, tm=1.0, r=0.1)
        at_expiry_kwargs = dict(call_put=OptionType.PUT, s=18 * PENCE / THERM, sigma=0.28, te=0,)
        self.expiring_option_itm = BachelierOption(**at_expiry_kwargs, k=19 * PENCE / THERM)
        self.expiring_option_otm = BachelierOption(**at_expiry_kwargs, k=17 * PENCE / THERM)
        self.expiring_option_atm = BachelierOption(**at_expiry_kwargs, k=18 * PENCE / THERM)

    def test_d(self):
        assert_almost_equal(self.option.d, -0.57735, atol=1e-4)
        assert_almost_equal(self.put_option.d, self.option.d, atol=1e-4)  # Put and call options are mirrors of each other
        assert_almost_equal(self.atm_option.d, 0.0, atol=1e-4)

    def test_value(self):
        result = self.option.value
        assert_almost_equal(0.27421, result, atol=1e-4)
        assert_almost_equal(self.option.sigma, self.option.get_vol_from_premium(result))
        assert_almost_equal(self.put_option.value, 1.179055, atol=1e-4)
        assert_almost_equal(self.option.sigma, self.option.get_vol_from_premium(result))

        # Put-call parity
        df = np.exp(-0.1 * 1.0)
        assert_almost_equal(self.put_option.value - self.option.value, 1 * df)

    def test_value_at_expiry_itm(self):
        assert_almost_equal(self.expiring_option_itm.value, 1 * PENCE / THERM)

    def test_value_at_expiry_otm(self):
        assert_almost_equal(self.expiring_option_otm.value, 0 * PENCE / THERM)

    def test_value_at_expiry_atm(self):
        assert_almost_equal(self.expiring_option_atm.value, 0 * PENCE / THERM)

    def test_vol_from_premium_at_expiry_itm(self):
        assert np.isnan(self.expiring_option_itm.get_vol_from_premium(1 * PENCE / THERM))

    def test_vol_from_premium_at_expiry_otm(self):
        assert np.isnan(self.expiring_option_otm.get_vol_from_premium(0 * PENCE / THERM))

    def test_vol_from_premium_at_expiry_atm(self):
        assert np.isnan(self.expiring_option_atm.get_vol_from_premium(0 * PENCE / THERM))

    @repeat(number_of_runs=10)
    def test_vol_premium_roundtrip_itm(self, rng):
        sigma = rng.uniform(10, 100)
        kwargs = dict(call_put=OptionType.CALL, s=100, k=90, te=1)
        opt1 = BachelierOption(**kwargs, sigma=sigma)
        opt2 = BachelierOption(**kwargs, sigma=None)
        assert_almost_equal(sigma, opt2.get_vol_from_premium(opt1.value), atol=1e-4)

    @repeat(number_of_runs=10)
    def test_vol_premium_roundtrip_otm(self, rng):
        sigma = rng.uniform(10, 100)
        kwargs = dict(call_put=OptionType.CALL, s=90, k=100, te=1)
        opt1 = BachelierOption(**kwargs, sigma=sigma)
        opt2 = BachelierOption(**kwargs, sigma=None)
        assert_almost_equal(sigma, opt2.get_vol_from_premium(opt1.value), atol=1e-4)

    @repeat(number_of_runs=10)
    def test_vol_premium_roundtrip_atm(self, rng):
        sigma = rng.uniform(10, 100)
        kwargs = dict(call_put=OptionType.CALL, s=100, k=100, te=1)
        opt1 = BachelierOption(**kwargs, sigma=sigma)
        opt2 = BachelierOption(**kwargs, sigma=None)
        assert_almost_equal(sigma, opt2.get_vol_from_premium(opt1.value), atol=1e-4)

    def test_zero_sigma(self):
        itm_zero_sigma_option = BachelierOption(s=20 * PENCE / THERM, k=18 * PENCE / THERM, sigma=0, te=1)
        assert itm_zero_sigma_option.value == 2 * PENCE / THERM
        otm_zero_sigma_option = BachelierOption(s=16 * PENCE / THERM, k=18 * PENCE / THERM, sigma=0, te=1)
        assert otm_zero_sigma_option.value == 0 * PENCE / THERM

    def test_put_premium_to_call_premium(self):
        # Test that we can create a put premium from a is_call premium for an option
        price, sigma, te, tm = 100, 0.25, 0.5, 0.6
        strikes = (90, 100, 110)
        for r in (0.0, 0.05):  # the rfrs
            for strike_from_delta in strikes:
                call_option = BachelierOption(
                    s=price, k=strike_from_delta, sigma=sigma, te=te, tm=tm, r=r, call_put=OptionType.CALL,
                )
                put_option = BachelierOption(
                    s=price, k=strike_from_delta, sigma=sigma, te=te, tm=tm, r=r, call_put=OptionType.PUT,
                )
                assert_almost_equal(put_option.put_premium_to_call_premium(put_option.value), call_option.value)

    def test_multiple_fwds(self):
        """
        Test that we can price options with an array of fwds.
        This happens in PPA strip calculator when we value the caps/ floors using a simulated mkt state
        """
        fwds = np.array([90, 100, 110])
        option = BachelierOption(s=fwds, k=100, sigma=0.5, te=1)

        expected_values = np.array([BachelierOption(s, 100, 0.5, 1).value for s in fwds])
        assert_almost_equal(expected_values, option.value)

    def test_multiple_strikes(self):
        """
        Test that we can price options with an array of strikes.
        This happens in kahale_pricer
        """
        strikes = np.array([90, 100, 110])
        option = BachelierOption(s=100, k=strikes, sigma=0.5, te=1)

        expected_values = np.array([BachelierOption(100, s, 0.5, 1).value for s in strikes])
        assert_almost_equal(expected_values, option.value)

    def test_inf_strike(self):
        inf_call = BachelierOption(s=100, k=np.inf, sigma=0.5, te=1, call_put=OptionType.CALL).value
        assert 0 == inf_call
        inf_put = BachelierOption(s=100, k=np.inf, sigma=0.5, te=1, call_put=OptionType.PUT).value
        assert np.inf == inf_put
        m_inf_call = BachelierOption(s=100, k=-np.inf, sigma=0.5, te=1, call_put=OptionType.CALL).value
        assert np.inf == m_inf_call
        m_inf_put = BachelierOption(s=100, k=-np.inf, sigma=0.5, te=1, call_put=OptionType.PUT).value
        assert 0 == m_inf_put

    def test_vectorized_edge_cases(self):
        np.testing.assert_equal(np.array([99, 0, 99]), BachelierOption(s=100, k=np.array([1, np.inf, 1]), sigma=0.5, te=1, call_put=OptionType.CALL).value)
        np.testing.assert_equal(np.array([99, 101, 99]), BachelierOption(s=100, k=np.array([1, -1, 1]), sigma=0.5, te=1, call_put=OptionType.CALL).value)
        np.testing.assert_equal(np.array([np.inf, 99]), BachelierOption(s=np.array([np.inf, 100]), k=1, sigma=0.5, te=1, call_put=OptionType.CALL).value)
        np.testing.assert_equal(np.array([np.inf, np.inf]), BachelierOption(
                s=np.array([np.inf, 1]), k=np.array([1, -np.inf]), sigma=0.5, te=1, call_put=OptionType.CALL,
            ).value)

    def test_delta(self):
        value = self.option.value
        delta = self.option.delta
        shock = 1e-6
        self.option.s += shock
        up_value = self.option.value
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-4)

        value = self.put_option.value
        delta = self.put_option.delta
        shock = 1e-6
        self.put_option.s += shock
        up_value = self.put_option.value
        fd_delta = (up_value - value) / shock
        assert_almost_equal(delta, fd_delta, atol=1e-4)

    def test_gamma(self):
        delta = self.option.delta
        gamma = self.option.gamma
        shock = 1e-6
        self.option.s += shock
        up_delta = self.option.delta
        fd_gamma = (up_delta - delta) / shock
        assert_almost_equal(gamma, fd_gamma)

    def test_vega(self):
        value = self.option.value
        vega = self.option.vega
        shock = 1e-6
        self.option.sigma += shock
        up_value = self.option.value
        fd_vega = (up_value - value) / shock
        assert_almost_equal(vega, fd_vega, atol=1e-6)

    def test_theta(self):
        value = self.option.value
        theta = self.option.theta
        shock = 1e-6
        self.option.te -= shock
        self.option.tm -= shock
        new_value = self.option.value
        fd_theta = (new_value - value) / shock
        assert_almost_equal(theta, fd_theta, atol=1e-6)


