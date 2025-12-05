import warnings

import numpy as np
from numpy import atleast_1d, exp, sqrt, isnan, isinf, isneginf, log, pi, nan
from scipy.optimize import newton

from scipy.stats import norm


N = norm.cdf
N_dash = norm.pdf
N_inv = norm.ppf


class BachelierOption:
    """Normal Option for analytical pricing"""

    def __init__(self, s, k, sigma, te, call_put="c", r=0):
        """ @param: s Initial value of underlying
            @param: k strike
            @param: te time to expiry
            @param: sigma volatility  - Black76 Vol * Par underlying price
            @param: r interest rate"""
        self.s = s
        self.k = k
        self.sigma = sigma
        self.te = te
        self.r = r
        assert call_put in ["c", "p"], f"{call_put} is not a recognised option type"
        self.call_put = call_put

    @property
    def omega(self):
        return 1. if self.call_put =="c" else -1.

    @property
    def d(self):
        r"""@return: \f$\frac{s - k}}{\sigma\sqrt{T}}\f$"""
        # Catch and ignore errors where self.te == 0
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return (self.s - self.k) / (self.sigma * sqrt(self.te))

    @property
    def value(self):
        result = np.maximum(self.omega * (self.s - self.k), 0)
        if self.te > 0 and self.sigma > 0:
            d = self.d
            nans = isnan(d) + isinf(d) + isneginf(d)
            if len(atleast_1d(d)) == 1:
                if not nans:
                    result = (
                        self.omega * (self.s - self.k) * N(self.omega * self.d)
                        + self.sigma * sqrt(self.te) * N_dash(self.d)
                    )
            else:  # array
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    result[~nans] = (
                        self.omega * (self.s - self.k) * N(self.omega * self.d)
                        + self.sigma * sqrt(self.te) * N_dash(self.d)
                    )[~nans]

        return self._discounted(result)

    def _discounted(self, val):
        if self.r != 0:
            val *= exp(-self.r * self.te)
        return val

    @property
    def delta(self):
        r"""@return: \f$ N(d_s)\f$"""
        delta = self.omega * N(self.omega * self.d)
        return self._discounted(delta)

    @property
    def gamma(self):
        r"""@return: \f$ \frac{n(d_s)}{\sigma\sqrt{T}}\f$"""
        gamma = N_dash(self.d) / (self.sigma * sqrt(self.te))
        return self._discounted(gamma)

    @property
    def vega(self):
        r"""@return: \f$ n(d_s)\sqrt{T}\f$"""
        vega = N_dash(self.d) * sqrt(self.te)
        return self._discounted(vega)

    @property
    def theta(self):
        r"""@return: \f$ rp e^{-rT} \frac{-\sigma n(d_s)}{2\sqrt{T}}\f$"""
        return (self.r * self.value) - self._discounted((self.sigma * N_dash(self.d)) / (2 * sqrt(self.te)))

    def put_premium_to_call_premium(self, put_premium):
        """ put call parity"""
        call_premium = self._discounted(self.s - self.k) + put_premium
        return call_premium

    def get_vol_from_premium(self, option_price):
        undiscounted_price = option_price * exp(self.r * self.te)

        if self.te < 1e-8:
            return nan

        if abs(self.k - self.s) < 1e-8:
            return undiscounted_price * sqrt(2 * pi / self.te)

        # get a first guess by estimating with http://www.jaeckel.org/ImpliedNormalVolatility.pdf
        phi_star = - (abs(undiscounted_price - (self.omega * (self.s - self.k))) / abs(self.k - self.s))
        g = 1 / (phi_star - 0.5)
        g2 = g**2
        eta_bar = (0.032114372355 - g2 * (0.016969777977 - g2 * (2.6207332461e-3 - 9.6066952861e-5 * g2))) / \
                  (1 - g2*(0.6635646938 - g2 * (0.14528712196 - 0.010472855461 * g2)))
        x_bar = g * ((1 / np.sqrt(2 * np.pi)) + eta_bar * g ** 2)
        fn_phi = lambda x: N(x, 0, 1) + (N_dash(x, 0, 1) / x)
        q = (fn_phi(x_bar) - phi_star) / (N_dash(x_bar, 0, 1))
        x_star = x_bar + (
                (3 * q * (x_bar ** 2) * (2 - q * x_bar * (2 + x_bar ** 2))) /
                (6 + q * x_bar * (-12 + x_bar * (6 * q + x_bar * (-6 + q * x_bar * (3 + x_bar ** 2)))))
            )
        sigma_guess = abs(self.k - self.s) / abs(x_star * np.sqrt(self.te))

        # converge with newton
        def obj_fn(_sigma):
            option = BachelierOption(self.s, self.k, _sigma, self.te, self.call_put, self.r)
            err = option.value - option_price
            return err

        sigma = newton(func=obj_fn, x0=sigma_guess, disp=False)
        return sigma
