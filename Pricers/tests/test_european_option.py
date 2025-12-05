import unittest
import numpy as np
from BGN_LNG.Pricers.european_option import Black76Option


def vectors_equal(v1, v2, tol=1e-4):
    if len(v1) != len(v2):
        return False
    le_tol = all(v1 - v2 < tol)
    ge_mtol = all(v1 - v2 > -tol)
    return le_tol and ge_mtol

class TestBlack76Option(unittest.TestCase):

    def setUp(self):
        # First example from Haug, Option Pricing Formulas
        self.option = Black76Option(s=19, k=19, sigma=0.28, te=0.75, r=0.1)
        self.put_option = Black76Option(call_put="p", s=18, k=19, sigma=0.28, te=1.0)
        self.expiring_option_itm = Black76Option(
            call_put="p", s=18, k=19, sigma=0.28, te=0,
        )
        self.expiring_option_otm = Black76Option(
            call_put="p", s=18, k=17, sigma=0.28, te=0,
        )

    def test_d1_d2(self):
        self.assertAlmostEqual(self.option.d1, 0.1212, places=4)
        self.assertAlmostEqual(self.option.d2, -0.1212, places=4)

    def test_value(self):
        self.assertAlmostEqual(self.option.value, 1.7010507, places=4)
        self.assertAlmostEqual(self.put_option.value, 2.5978, places=4)

    def test_value_at_expiry_itm(self):
        self.assertAlmostEqual(self.expiring_option_itm.value, 1)

    def test_value_at_expiry_otm(self):
        self.assertAlmostEqual(self.expiring_option_otm.value, 0)

    def test_value_at_expiry_atm(self):
        expiring_option_atm = Black76Option(
            call_put="p", s=18, k=18, sigma=0.28, te=0,
        )
        self.assertAlmostEqual(expiring_option_atm.value, 0)

    def test_zero_sigma(self):
        itm_zero_sigma_option = Black76Option(s=20, k=18, sigma=0, te=1)
        assert itm_zero_sigma_option.value == 2
        otm_zero_sigma_option = Black76Option(s=16, k=18, sigma=0, te=1)
        assert otm_zero_sigma_option.value == 0

    def test_delta(self):
        value = self.option.value
        delta = self.option.delta
        shock = 1e-6
        self.option.s += shock
        up_value = self.option.value
        fd_delta = (up_value - value) / shock
        self.assertAlmostEqual(delta, fd_delta)

        value = self.put_option.value
        delta = self.put_option.delta
        shock = 1e-6
        self.put_option.s += shock
        up_value = self.put_option.value
        fd_delta = (up_value - value) / shock
        self.assertAlmostEqual(delta, fd_delta)

    def test_gamma(self):
        delta = self.option.delta
        gamma = self.option.gamma
        shock = 1e-6
        self.option.s += shock
        up_delta = self.option.delta
        fd_gamma = (up_delta - delta) / shock
        self.assertAlmostEqual(gamma, fd_gamma)

    def test_vega(self):
        value = self.option.value
        vega = self.option.vega
        shock = 1e-6
        self.option.sigma += shock
        up_value = self.option.value
        fd_vega = (up_value - value) / shock
        self.assertAlmostEqual(vega, fd_vega, places=6)

    def test_theta(self):
        value = self.option.value
        theta = self.option.theta
        shock = 1e-6
        self.option.te -= shock
        new_value = self.option.value
        fd_theta = (new_value - value) / shock
        self.assertAlmostEqual(theta, fd_theta, places=6)

    def test_d_star(self):
        assert 0 == Black76Option(call_put="c", s=1, k=1, te=1, sigma=0.5).dstar  # s ==  k
        assert 1 == Black76Option(call_put="c", s=np.exp(0.5), k=1, te=1, sigma=0.5).dstar  # k / s = exp(0.5)
        assert 1 == Black76Option(call_put="c", s=np.exp(np.sqrt(2)), k=1, te=2, sigma=1).dstar  # t == 2

    def test_delta_star(self):
        assert 0.5 == Black76Option(call_put="c", s=100, k=100, sigma=0.5, te=1, r=0).delta_star
        assert -0.5 == Black76Option(call_put="p", s=100, k=100, sigma=0.5, te=1, r=0).delta_star

    def test_delta_to_strike(self):
        s, sigma, te = 100, 0.25, 0.5
        strikes = (95, 100, 105)
        for call_put in ["c", "p"]:
            for k in strikes:
                option = Black76Option(call_put=call_put, s=s, k=k, sigma=sigma, te=te)
                delta = option.delta
                strike_from_delta = option.strike_from_delta(delta)
                assert k == strike_from_delta

                # Test we can construct an option with no strike_from_delta
                option = Black76Option(call_put=call_put, s=s, k=None, sigma=sigma, te=te)
                strike_from_delta = option.strike_from_delta(delta)
                assert k == strike_from_delta

    def test_strike_star(self):
        # Check atm
        option = Black76Option(call_put="c", s=100, k=None, sigma=0.5, te=1, r=0)
        assert 100 == option.strike_from_delta_star(0.5)
        # Check round trip
        k = 110.0
        for option_type in ("p", "c"):
            delta_star = Black76Option(call_put=option_type, s=100, k=k, sigma=0.5, te=1, r=0).delta_star
            strike_from_delta = Black76Option(call_put=option_type, s=100, k=None, sigma=0.5, te=1,
                                              r=0).strike_from_delta_star(delta_star)
            self.assertAlmostEqual(k, strike_from_delta)

    def test_get_vol_from_premium(self):
        price, sigma, te = 100, 0.25, 0.5
        strikes = (90, 100, 110)
        tol_cm, tol_bcs, tol_secant = 0.02, 0.33, 1e-5
        for r in (0.0, 0.05):  # the rfrs
            for call_put in ["p", "c"]:
                for strike_from_delta in strikes:
                    option = Black76Option(call_put=call_put, s=price, k=strike_from_delta, sigma=sigma, te=te, r=r)
                    sigma_cm = option.get_vol_from_premium(option.value, "CorradoMiller")  # test the vol round trips from the CM model
                    sigma_bcs = option.get_vol_from_premium(option.value, "BharadiaChristopherSalkin")  # test the vol round trips from the BCS model
                    sigma_secant = option.get_vol_from_premium(option.value, "Secant")
                    # Check corrado miller error is less than 2%
                    assert abs(sigma - sigma_cm) / sigma < tol_cm, "%s error: %s !< %s for rfr=%s & %s @ %s" % ("corrado_miller",
                                                                                 tol_cm, abs(sigma - sigma_cm) / sigma,
                                                                                 r, call_put, strike_from_delta)
                    # Check bharadia christopher salkin error is less than 33%
                    # The BCS model is significantly less accurate - this test proves it!
                    assert abs(sigma - sigma_bcs) / sigma < tol_bcs, "%s error: %s !< %s for rfr=%s & %s @ %s" % ("bharadia_christopher_salkin",
                                                                                 tol_bcs, abs(sigma - sigma_bcs) / sigma,
                                                                                 r, call_put, strike_from_delta)
                    # Check that we used the secant method to solve for the vol
                    assert abs(sigma - sigma_secant) / sigma < tol_secant, "%s error: %s !< %s for rfr=%s & %s @ %s" % ("secant",
                                                                                 tol_secant, abs(sigma - sigma_secant) / sigma,
                                                                                 r, call_put, strike_from_delta)
        #Test case when premium approx. 0
        price, strike, sigma, te = 100, 1, 0.01, 0.1
        option = Black76Option(call_put="c", s=price, k=strike, sigma=sigma, te=te)
        assert np.isnan(option.get_vol_from_premium(option.value))

    def test_put_premium_to_call_premium(self):
        # Test that we can create a put premium from a is_call premium for an option
        price, sigma, te, tm = 100, 0.25, 0.5, 0.6
        strikes = (90, 100, 110)
        for r in (0.0, 0.05):  # the rfrs
            for strike_from_delta in strikes:
                call_option = Black76Option(
                    s=price, k=strike_from_delta, sigma=sigma, te=te, r=r, call_put="c",
                )
                put_option = Black76Option(
                    s=price, k=strike_from_delta, sigma=sigma, te=te, r=r, call_put="p",
                )
                self.assertAlmostEqual(put_option.put_premium_to_call_premium(put_option.value), call_option.value)

    def test_b76_consistency(self):
        """ Test valuation consistency for analytical greeks vs finite differences """
        b76_prices, b76_deltas, b76_gammas, b76_vegas, b76_thetas = {}, {}, {}, {}, {}
        price, sigma, te = 100, 0.25, 0.5
        tm = te + 0.1
        ds, dt = 0.001, (1.0 / 365.0) / 48.0
        for strike in (90, 100, 110):
            for r in (0.0, 0.05):  # the rfrs
                for call_put in ["c", "p"]:
                    options = [Black76Option(s=price + s_bump, k=strike, sigma=sigma + v_bump, te=te, r=r, call_put=call_put)
                               for s_bump, v_bump in [(ds, 0), (0, 0), (-ds, 0), (0, ds), (0, -ds)]]
                    b76_prices[call_put] = [opt.value for opt in options]
                    b76_deltas[call_put] = [opt.delta for opt in options]
                    b76_gammas[call_put] = [opt.gamma for opt in options][1]
                    b76_vegas[call_put] = [opt.vega for opt in options][1]
                    b76_thetas[call_put] = [opt.theta for opt in options][1]
                    # Calc values via finite difference
                    fd_delta = (b76_prices[call_put][0] - b76_prices[call_put][2]) / (2 * ds)
                    fd_gammas = ((b76_deltas[call_put][0] - b76_deltas[call_put][2]) / (2 * ds),
                                 (b76_prices[call_put][0] - 2 * b76_prices[call_put][1] + b76_prices[call_put][2])
                                 / ds ** 2)
                    fd_vega = (b76_prices[call_put][3] - b76_prices[call_put][4]) / (2 * ds)
                    fd_theta = (Black76Option(s=price, k=strike, sigma=sigma, te=te - dt, r=r, call_put=call_put).value
                                - b76_prices[call_put][1]) / dt
                    self.assertAlmostEqual(b76_deltas[call_put][1], fd_delta, places=4,
                                           msg='%s != %s for s: %s, r:%s, is_call:%s' % (b76_deltas[call_put][1],
                                                                                     fd_delta, strike, r,
                                                                                     call_put))
                    for fd_gamma in fd_gammas:  # Gamma is checked against fd via delta and price
                        self.assertAlmostEqual(b76_gammas[call_put], fd_gamma, places=4,
                                               msg='%s != %s for s: %s, r:%s, is_call:%s' % (b76_gammas[call_put],
                                                                                         fd_gamma, strike, r, call_put))
                    self.assertAlmostEqual(b76_vegas[call_put], fd_vega, places=3,
                                           msg='%s != %s for s: %s, r:%s, is_call:%s' % (b76_vegas[call_put], fd_vega,
                                                                                     strike, r, call_put))
                    self.assertAlmostEqual(b76_thetas[call_put], fd_theta, places=3,
                                           msg='%s != %s for s: %s, r:%s, is_call:%s' % (b76_thetas[call_put], fd_theta,
                                                                                     strike, r, call_put))
                # Check that the call and put values are equal where expected to be and vice-versa
                assert b76_prices["c"] != b76_prices["p"]
                assert b76_deltas["c"] != b76_deltas["p"]
                assert b76_gammas["c"] == b76_gammas["p"]
                assert b76_vegas["c"] == b76_vegas["p"]

    def test_multiple_fwds(self):
        """
        Test that we can price options with an array of fwds.
        This happens in PPA strip calculator when we value the caps/ floors using a simulated mkt state
        """
        fwds = np.array([90, 100, 110])
        option = Black76Option(s=fwds, k=100, sigma=0.5, te=1)

        expected_values = np.array([Black76Option(s, 100, 0.5, 1).value for s in fwds])
        self.assertTrue(vectors_equal(option.value, expected_values))

    def test_multiple_strikes(self):
        """
        Test that we can price options with an array of strikes.
        """
        strikes = np.array([90, 100, 110])
        option = Black76Option(s=100, k=strikes, sigma=0.5, te=1)

        expected_values = np.array([Black76Option(100, s, 0.5, 1).value for s in strikes])
        self.assertTrue(vectors_equal(option.value, expected_values))

    def test_inf_strike(self):
        inf_call = Black76Option(s=100, k=np.inf, sigma=0.5, te=1, call_put="c").value
        assert 0 == inf_call
        inf_put = Black76Option(s=100, k=np.inf, sigma=0.5, te=1, call_put="p").value
        assert np.inf == inf_put
        m_inf_call = Black76Option(s=100, k=-np.inf, sigma=0.5, te=1, call_put="c").value
        assert np.inf == m_inf_call
        m_inf_put = Black76Option(s=100, k=-np.inf, sigma=0.5, te=1, call_put="p").value
        assert 0 == m_inf_put

    def test_vectorized_edge_cases(self):
        self.assertTrue(vectors_equal(np.array([99, 0, 99]), Black76Option(s=100, k=np.array([1, np.inf, 1]), sigma=0.5, te=1, call_put="c").value))
        self.assertTrue(vectors_equal(np.array([99, 101, 99]), Black76Option(s=100, k=np.array([1, -1, 1]), sigma=0.5, te=1, call_put="c").value))

