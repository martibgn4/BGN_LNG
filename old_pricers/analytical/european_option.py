'''
Created on 13 Mar 2013

Module to contain models that can provide analytical (or semi-analytical)
prices and/or deltas
'''
import math
import warnings

from numpy import atleast_1d, exp, log, nan, sqrt, isnan, isinf, isneginf, divide
from quantity import units, value, maximum
from scipy.optimize import newton
from scipy.stats import norm

from log_utils import log_capture_warnings
from trade_data import OptionType

from thorn.core.base.pillar import DeltaPillar, DeltaStarPillar, ATM_PILLAR

N = norm.cdf
N_dash = norm.pdf
N_inv = norm.ppf


class Black76Option:
    """Black76Option class acts a data wrapper to group all the relevant inputs for analytical pricing"""

    def __init__(self, s, k, sigma, te, call_put=OptionType.CALL, tm=None, r=0):
        """
        @param: s Initial value of underlying
        @param: k strike
        @param: sigma volatility
        @param: te time to expiry
        @param: call_put option type
        @param: tm time to maturity
        @param: r interest rate
        """
        self.s = s
        self.k = k
        self.sigma = sigma
        self.te = te
        self.tm = tm if tm else te
        self.r = r
        assert isinstance(call_put, OptionType), f"'{call_put}' is not a recognised option type"
        self.call_put = call_put

    @property
    def omega(self):
        return 1. if self.call_put is OptionType.CALL else -1.

    @property
    def d1(self):
        r'''@return: \f$\frac{\ln{s/k} + \sigma^2T/2}{\sigma\sqrt{T}}\f$'''
        # Catch and ignore errors where self.te == 0
        if not units(self.s).is_convertible_to(units(self.k)):
            raise AssertionError(f"{units(self.s)} vs {units(self.k)} ")
        unit_ratio = units(self.s) / units(self.k)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            return (
                log(divide(value(self.s) * unit_ratio, value(self.k))) + (0.5 * self.sigma ** 2) * self.te
            ) / (self.sigma * sqrt(self.te))

    @property
    def d2(self):
        r'''@return: \f$d1 - \sigma\sqrt{T}\f$'''
        return self.d1 - self.sigma * sqrt(self.te)

    @property
    def value(self):
        result = maximum(self.omega * (self.s - self.k), 0 * units(self.s))
        if self.te > 0:
            d1 = value(self.d1)
            nans = isnan(d1) + isinf(d1) + isneginf(d1)
            if len(atleast_1d(d1)) == 1:
                if not nans:
                    result = self.omega * (self.s * N(self.omega * self.d1) - self.k * N(self.omega * self.d2))
            else:  # array
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    result[~nans] = \
                        self.omega * (self.s * N(self.omega * self.d1) - self.k * N(self.omega * self.d2))[~nans]

        if self.r != 0:
            result *= exp(-self.r * self.tm)
        return result

    @property
    def delta(self):
        r'''@return: \f$ N(d_1) e^{-r T_m} \f$ if call else \f$ ( N(d_1) - 1 ) e^{-r T_m} \f$'''
        return self.omega * N(self.omega * self.d1) * exp(-self.r * self.tm)

    @property
    def gamma(self):
        r'''@return: \f$ e^{-r T_m} \frac{N'(d_1)}{s \sigma \sqrt{t}} \f$'''
        return exp(-self.r * self.tm) * N_dash(self.d1) / (self.s * self.sigma * sqrt(self.te))

    @property
    def vega(self):
        r'''@return: \f$ s e^{-r T_m} N'(d_1) \sqrt{t_e} \f$'''
        return self.s * exp(-self.r * self.tm) * N_dash(self.d1) * sqrt(self.te)

    @property
    def theta(self):
        r''' @return: \f$ - \frac{s e^{-r T_m} N'(d_1) \sigma}{2\sqrt{t_e}}
            \pm r s e^{-r T_m} N(\pm d_1) \mp rke^{-r T_m}N(\pm d_2)\f$
            where \f$ \pm \f$ is positive for a call option and negative for a put'''
        x_0 = self.s * exp(-self.r * self.tm) * N_dash(self.d1) * self.sigma / (2 * sqrt(self.te))
        x_1 = self.r * self.s * math.exp(-self.r * self.tm) * N(self.omega * self.d1)
        x_2 = self.r * self.k * math.exp(-self.r * self.tm) * N(self.omega * self.d2)
        return -x_0 + self.omega * x_1 - self.omega * x_2

    @property
    def dstar(self):
        r'''@return: \f$\frac{\ln{s/k}}{\sigma_{atm}\sqrt{T}}\f$'''
        return log(self.s / self.k) / (self.sigma * sqrt(self.te))

    @property
    def delta_star(self):
        r'''@return: \f$ N(d_star) e^{-r T_m} \f$ if call else \f$ ( N(d_star) - 1 ) e^{-r T_m} \f$'''
        return self.omega * N(self.omega * self.dstar) * math.exp(-self.r * self.tm)

    def strike_from_delta_and_vol(self, delta, vol):
        r'''Starting from \f$ d_1 = (\ln(S/K) + \sigma^2 T_e /2 ) / ( \sigma \sqrt{T_e} ) \f$
        Where S = Forward Price, K = strike_from_delta, \f$ \sigma \f$ = sigma, T = time to expiry
        and \f$ d_1 = N^{-1}(\delta) \f$ for a is_call or \f$ d_1 = -1 \times N^{-1}(-1 \times \delta) \f$ for a
        put option.
        For a call option: \f$ k = S_0 e^{\sigma^2 T_e / 2 - \sigma \sqrt{T_e} N^{-1}(\delta)} \f$ note \f$ r = 0 \f$
            When r != 0, \f$ \delta_{r!=0} = e^{-r T_m} \delta_{r=0} \f$ thus
            \f$ k = S_0 e^{\sigma^2 T_e /2 - \sigma \sqrt{T_e} N^{-1}(\delta e^{r T_m})} \f$'''
        delta_r0 = math.exp(self.r * self.tm) * delta  # get undiscounted delta
        d_1 = self.omega * N_inv(self.omega * delta_r0)
        return self.s * exp(vol * vol * self.te * 0.5 - vol * sqrt(self.te) * d_1)

    def strike_from_delta(self, delta):
        return self.strike_from_delta_and_vol(delta, self.sigma)

    def strike_from_delta_star_and_atm_vol(self, delta_star, atm_vol):
        d_star = self.omega * N_inv(self.omega * delta_star)
        return self.s * exp(-atm_vol * sqrt(self.te) * d_star)

    def strike_from_delta_star(self, delta_star):
        return self.strike_from_delta_star_and_atm_vol(delta_star, self.sigma)

    def strike_from_pillar_and_smile(self, pillar, smile):
        if isinstance(pillar, DeltaPillar):
            return self.strike_from_delta_and_vol(pillar.delta, smile[pillar])
        if isinstance(pillar, DeltaStarPillar):
            return self.strike_from_delta_star_and_atm_vol(pillar.delta_star, smile[ATM_PILLAR])
        raise TypeError(f"{pillar} should be of type DeltaPillar or DeltaStarPillar")

    def get_vol_from_premium(self, option_price, model_name="CorradoMiller"):
        if self.omega < 0:
            option_price = self.put_premium_to_call_premium(option_price)

        intrinsic = max(self.s - self.k, 0 * self.s) * exp(-self.r * self.tm)
        extrinsic = option_price - intrinsic
        threshold = 1e-8 * units(extrinsic)
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
                    s=self.s, k=self.k, sigma=sigma, te=self.te, call_put=OptionType.CALL, tm=self.tm, r=self.r,
                )
                err = option.value - option_price
                return err

            with log_capture_warnings():
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
        disc_k = self.k * exp(-self.r * self.tm)
        disc_s = self.s * exp(-self.r * self.tm)
        x_a = sqrt(2 * math.pi) / (disc_s + disc_k)
        x_b = call_option_price - (disc_s - disc_k) / 2.0
        # Away from the money, this can be negative, so ensure positive
        x_b2 = x_b ** 2
        x_c = max(x_b2 - ((disc_s - disc_k) ** 2 / math.pi), 0.0 * x_b2)
        x_d = x_b + sqrt(value(x_c)) * units(x_b)
        sigma_0 = x_a * x_d / sqrt(self.te)
        return sigma_0

    def _bharadia_christopher_salkin(self, call_option_price):
        r"""
        Simple (bharadia_christopher_salkin) model to approximate a call options implied volatility
        This is here for testing purposes as a reference point showing the CM model (above) is better
        """
        disc_k = self.k * math.exp(-self.r * self.tm)
        disc_s = self.s * math.exp(-self.r * self.tm)
        delta = (disc_s - disc_k) / 2.0
        sigma_0 = sqrt(2 * math.pi - self.te) * ((call_option_price - delta) / (disc_s - delta))
        return sigma_0

    def put_premium_to_call_premium(self, put_premium):
        r"""
        Converts a put premium to call premium using put-call parity
        @return:  \f$C - P = D(s - k)\f$ where \f$D = e^{-rT}\f$
        """
        df = math.exp(-self.r * self.tm)
        call_premium = df * (self.s - self.k) + put_premium
        return call_premium


