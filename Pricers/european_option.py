import math

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from numpy import atleast_1d, exp, log, nan, sqrt, isnan, isinf, isneginf, divide
import warnings

from BGN_LNG.Utils.datetime_utils import time_between

N = norm.cdf
N_dash = norm.pdf
N_inv = norm.ppf


class Black76Option:
    """Black76Option class acts a data wrapper to group all the relevant inputs for analytical pricing"""

    def __init__(self, s, k, sigma, te, call_put="c", r=0.0):
        """
        @param: s Initial value of underlying
        @param: k strike
        @param: sigma volatility
        @param: te time to expiry
        @param: call_put option type, "c" or "p"
        @param: r interest rate
        """
        self.s = s
        self.k = k
        self.sigma = sigma
        self.te = te
        self.r = r
        assert call_put in ["c", "p"], f"'{call_put}' is not a recognised option type, should be 'c' or 'p'"
        self.call_put = call_put

    @property
    def omega(self):
        return 1. if self.call_put == "c" else -1.

    @property
    def d1(self):
        return (
                log(divide(self.s, self.k)) + (0.5 * self.sigma ** 2) * self.te
            ) / (self.sigma * sqrt(self.te))

    @property
    def d2(self):
        return self.d1 - self.sigma * sqrt(self.te)

    @property
    def value(self):
        result = np.maximum(self.omega * (self.s - self.k), 0)
        if self.te > 0:
            d1 = self.d1
            nans = isnan(d1) + isinf(d1) + isneginf(d1)
            if len(atleast_1d(d1)) == 1:
                if not nans:
                    result = self.omega * (self.s * N(self.omega * self.d1) - self.k * N(self.omega * self.d2))
            else:  # array
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    result[~nans] = \
                        self.omega * (self.s * N(self.omega * self.d1) - self.k * N(self.omega * self.d2))[~nans]

        if self.r != 0.0:
            result *= exp(-self.r * self.te)
        return result

    @property
    def delta(self):
        return self.omega * N(self.omega * self.d1) * exp(-self.r * self.te)

    @property
    def gamma(self):
        return exp(-self.r * self.te) * N_dash(self.d1) / (self.s * self.sigma * sqrt(self.te))

    @property
    def vega(self):
        return self.s * exp(-self.r * self.te) * N_dash(self.d1) * sqrt(self.te)

    @property
    def theta(self):
        r''' @return: \f$ - \frac{s e^{-r T_m} N'(d_1) \sigma}{2\sqrt{t_e}}
            \pm r s e^{-r T_m} N(\pm d_1) \mp rke^{-r T_m}N(\pm d_2)\f$
            where \f$ \pm \f$ is positive for a call option and negative for a put'''
        x_0 = self.s * exp(-self.r * self.te) * N_dash(self.d1) * self.sigma / (2 * sqrt(self.te))
        x_1 = self.r * self.s * math.exp(-self.r * self.te) * N(self.omega * self.d1)
        x_2 = self.r * self.k * math.exp(-self.r * self.te) * N(self.omega * self.d2)
        return -x_0 + self.omega * x_1 - self.omega * x_2

    @property
    def dstar(self):
        r'''@return: \f$\frac{\ln{s/k}}{\sigma_{atm}\sqrt{T}}\f$'''
        return log(self.s / self.k) / (self.sigma * sqrt(self.te))

    @property
    def delta_star(self):
        r'''@return: \f$ N(d_star) e^{-r T_m} \f$ if call else \f$ ( N(d_star) - 1 ) e^{-r T_m} \f$'''
        return self.omega * N(self.omega * self.dstar) * math.exp(-self.r * self.te)

    def strike_from_delta_and_vol(self, delta, vol):
        r'''Starting from \f$ d_1 = (\ln(S/K) + \sigma^2 T_e /2 ) / ( \sigma \sqrt{T_e} ) \f$
        Where S = Forward Price, K = strike_from_delta, \f$ \sigma \f$ = sigma, T = time to expiry
        and \f$ d_1 = N^{-1}(\delta) \f$ for a is_call or \f$ d_1 = -1 \times N^{-1}(-1 \times \delta) \f$ for a
        put option.
        For a call option: \f$ k = S_0 e^{\sigma^2 T_e / 2 - \sigma \sqrt{T_e} N^{-1}(\delta)} \f$ note \f$ r = 0 \f$
            When r != 0, \f$ \delta_{r!=0} = e^{-r T_m} \delta_{r=0} \f$ thus
            \f$ k = S_0 e^{\sigma^2 T_e /2 - \sigma \sqrt{T_e} N^{-1}(\delta e^{r T_m})} \f$'''
        delta_r0 = math.exp(self.r * self.te) * delta  # get undiscounted delta
        d_1 = self.omega * N_inv(self.omega * delta_r0)
        return self.s * exp(vol * vol * self.te * 0.5 - vol * sqrt(self.te) * d_1)

    def strike_from_delta(self, delta):
        return self.strike_from_delta_and_vol(delta, self.sigma)

    def strike_from_delta_star_and_atm_vol(self, delta_star, atm_vol):
        d_star = self.omega * N_inv(self.omega * delta_star)
        return self.s * exp(-atm_vol * sqrt(self.te) * d_star)

    def strike_from_delta_star(self, delta_star):
        return self.strike_from_delta_star_and_atm_vol(delta_star, self.sigma)

    def get_vol_from_premium(self, option_price, model_name="CorradoMiller"):
        if self.omega < 0:
            option_price = self.put_premium_to_call_premium(option_price)

        intrinsic = max(self.s - self.k, 0 * self.s) * exp(-self.r * self.te)
        extrinsic = option_price - intrinsic
        threshold = 1e-8
        if extrinsic <= threshold:
            return nan

        if model_name == "CorradoMiller":
            return self._corrado_miller(option_price)
        elif model_name == "BharadiaChristopherSalkin":
            return self._bharadia_christopher_salkin(option_price)
        elif model_name == "Secant":
            initial_guess = self._corrado_miller(option_price)

            def error(sigma):
                option = self.__class__(
                    s=self.s, k=self.k, sigma=sigma, te=self.te, call_put="c", r=self.r,
                )
                err = option.value - option_price
                return err

            sigma = newton(func=error, x0=initial_guess, disp=False)

            assert error(sigma) / option_price < 1e-4, f"{error(sigma)} / {option_price} !< 1e-4"
            return sigma
        else:
            raise ValueError("Unsupported model_name '%s'" % model_name)

    def _corrado_miller(self, call_option_price):
        r"""
        Slightly complex (corrado miller) model to approximate a call options implied volatility
        Solves \f$ \sigma = \left(\frac{1}{\sqrt{T}}\right) \left\{ \left[ \frac{\sqrt{2\pi}}{S+Ke^{-rT}}\right]
        \left[P_m - \left[\frac{S - Ke^{-rT}}{2}\right] +
        \sqrt{\left(P_m - \left[\frac{S - Ke^{-rT}}{2}\right] \right)^2 - \left[\frac{(S - Ke^{-rT})^2}{\pi}\right]}
        \right] \right\} \f$
        @note This has a slight adjustment to the published form to avoid imaginary roots, and enforces that the term
        inside the square root is greater than or equal to zero.
        """
        disc_k = self.k * exp(-self.r * self.te)
        disc_s = self.s * exp(-self.r * self.te)
        x_a = sqrt(2 * math.pi) / (disc_s + disc_k)
        x_b = call_option_price - (disc_s - disc_k) / 2.0
        # Away from the money, this can be negative, so ensure positive
        x_b2 = x_b ** 2
        x_c = max(x_b2 - ((disc_s - disc_k) ** 2 / math.pi), 0.0 * x_b2)
        x_d = x_b + sqrt(x_c)
        sigma_0 = x_a * x_d / sqrt(self.te)
        return sigma_0

    def _bharadia_christopher_salkin(self, call_option_price):
        r"""
        Simple (bharadia_christopher_salkin) model to approximate a call options implied volatility
        This is here for testing purposes as a reference point showing the CM model (above) is better
        """
        disc_k = self.k * math.exp(-self.r * self.te)
        disc_s = self.s * math.exp(-self.r * self.te)
        delta = (disc_s - disc_k) / 2.0
        sigma_0 = sqrt(2 * math.pi - self.te) * ((call_option_price - delta) / (disc_s - delta))
        return sigma_0

    def put_premium_to_call_premium(self, put_premium):
        r"""
        put_call parity
        """
        df = math.exp(-self.r * self.te)
        call_premium = df * (self.s - self.k) + put_premium
        return call_premium


def black76_option(
    pricing_date, expiry_date, fwd_price, strike, vol, r=0.0, option_type="c",
):
    expiry_time = time_between(pricing_date, expiry_date)
    # r = -math.log(discount_factor) / expiry_time
    return Black76Option(s=fwd_price, k=strike, sigma=vol, te=expiry_time, call_put=option_type, r=r)


def black76_price(pricing_date, expiry_date, fwd_price, strike, option_type, vol, r):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r, option_type)
    return option.value


def black76_delta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, r):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r, option_type)
    return option.delta


def black76_gamma(pricing_date, expiry_date, fwd_price, strike, vol, r):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r)
    return option.gamma


def black76_vega(pricing_date, expiry_date, fwd_price, strike, vol, r):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r)
    return option.vega


def black76_theta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, r):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r, option_type)
    return option.theta / 365


def black76_vol_from_premium(
    pricing_date, expiry_date, fwd_price, strike, premium, option_type="c", r=0.0,
):
    vol = None
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, r, option_type)
    return option.get_vol_from_premium(premium, model_name="Secant")