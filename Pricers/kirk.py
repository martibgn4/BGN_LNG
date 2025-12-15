from math import log, exp, sqrt
import numpy as np
from scipy.stats import norm, multivariate_normal

from BGN_LNG.Pricers.european_option import Black76Option
from BGN_LNG.Utils.datetime_utils import time_between

POS = True
NEG = False

N = norm.cdf
N_dash = norm.pdf
N_inv = norm.ppf

class KirkSpreadPricer:
    def __init__(self, pos_fwd, neg_fwd, strike, pos_vol, neg_vol, corr, te, option_type="c", r=0.0):
        self.pos_fwd = pos_fwd
        self.neg_fwd = neg_fwd
        self.pos_vol = pos_vol
        adj_neg_vol = self._adjust_neg_vol(neg_fwd, neg_vol, strike)
        self._starting_neg_vol = neg_vol
        self.neg_vol = adj_neg_vol
        self.strike = strike
        self.corr = corr
        self.te = te
        self.option_type = option_type
        self.r = r

    @staticmethod
    def _adjust_neg_vol(neg_fwd, neg_vol, strike):
        adj_neg_vol = -neg_fwd * neg_vol
        if adj_neg_vol != 0:
            try:
                adj_neg_vol /= (-neg_fwd + strike)
            except ZeroDivisionError as e:
                adj_neg_vol = np.inf
        return adj_neg_vol

    @property
    def spread_vol(self):
        if self.neg_vol <= 0:
            return self.pos_vol
        if self.pos_vol <= 0:
            return self.neg_vol
        pos_var = self.pos_vol * self.pos_vol
        neg_var = self.neg_vol * self.neg_vol
        pos_neg_cov = -2 * self.corr * self.pos_vol * self.neg_vol
        # Handle precision errors when using proxy vols
        return sqrt(max(0.0, pos_var + neg_var + pos_neg_cov))

    def _option(self, option_type, intrinsic):
        """ [s_1 - (s_2 + k)]^+ """
        s = self.pos_fwd
        k = -self.neg_fwd + self.strike
        vol = 0 if intrinsic else self.spread_vol
        return Black76Option(
            s, k, vol, self.te, r=self.r, call_put=option_type or self.option_type
        )

    def _option_neg(self, option_type, intrinsic):
        """ [(s_1 - k) - s_2]^+ """
        s = -self.neg_fwd
        k = self.pos_fwd - self.strike
        vol = 0 if intrinsic else self.spread_vol
        call_put = option_type or self.option_type
        inverse_call_put = "c" if call_put == "p" else "c"
        return Black76Option(
            s, k, vol, self.te, r=self.r, call_put=inverse_call_put
        )

    def option_value(self, option_type=None, intrinsic=False):
        option = self._option(option_type, intrinsic)
        return option.value

    def _delta_leg(self, is_pos_leg, option_type=None, intrinsic=False):
        # Does not support pathwise
        option = (self._option if is_pos_leg else self._option_neg)(option_type, intrinsic)
        tol = 1e-6
        if option.sigma < tol:
            if option.value > tol:  # In the money
                return option.omega
            return 0.0  # Out of the money
        return option.delta

    def delta_pos_leg(self, option_type=None, intrinsic=False):
        return self._delta_leg(True, option_type, intrinsic)

    def delta_neg_leg(self, option_type=None, intrinsic=False):
        return self._delta_leg(False, option_type, intrinsic)

    def delta(self, option_type=None, intrinsic=False):
        return self.delta_pos_leg(option_type, intrinsic)

    def gamma_pos_leg(self, option_type=None, intrinsic=False):
        option = self._option(option_type, intrinsic)
        return option.gamma

    def gamma_neg_leg(self, option_type=None, intrinsic=False):
        option = self._option_neg(option_type, intrinsic)
        return option.gamma

    def gamma(self, option_type=None, intrinsic=False):
        return self.gamma_pos_leg(option_type, intrinsic)

    def vega(self, option_type=None, intrinsic=False):
        option = self._option(option_type, intrinsic)
        return option.vega

    def _vega_leg(self, is_pos=True):
        dc_dsigma = self.vega()
        b = -self.neg_fwd / (-self.neg_fwd + self.strike)
        if is_pos:
            dsigma_dsigma_leg = (self.pos_vol - self._starting_neg_vol * b * self.corr) / self.spread_vol
        else:
            dsigma_dsigma_leg = (self._starting_neg_vol * (b ** 2) - self.pos_vol * b * self.corr) / self.spread_vol
        return dc_dsigma * dsigma_dsigma_leg

    def vega_pos_leg(self):
        r"""
        \frac{\partial C}{\partial \sigma_1}
        = \frac{\partial C}{\partial \sigma} \frac{\partial \sigma}{\partial \sigma_1}
        = \frac{\partial C}{\partial \sigma}  \frac{\sigma_1 - \sigma_2 b \rho}{\sigma}
        where b = \frac{s_2}{s_2 + k}
        """
        return self._vega_leg(True)

    def vega_neg_leg(self):
        r"""
        \frac{\partial C}{\partial \sigma_2}
        = \frac{\partial C}{\partial \sigma} \frac{\partial \sigma}{\partial \sigma_2}
        = \frac{\partial C}{\partial \sigma} \frac{\sigma_2 b^2 - \sigma_1 b \rho}{\sigma}
        where b = \frac{s_2}{s_2 + k}
        """
        return self._vega_leg(False)

    def theta(self, option_type=None, intrinsic=False):
        option = self._option(option_type, intrinsic)
        return option.theta

    def corr_sensitivity(self):
        corr_bump = 0.0001
        def get_bumped_corr_option_value(_corr_bump):
            return KirkSpreadPricer(
                self.pos_fwd, self.neg_fwd, self.strike, self.pos_vol, self.neg_vol, self.corr + _corr_bump,
                self.te, r=self.r, option_type=self.option_type
            ).option_value(self.option_type)
        up_option_value =  get_bumped_corr_option_value(corr_bump)
        down_option_value = get_bumped_corr_option_value(-corr_bump)
        s = (up_option_value - down_option_value) / (2*corr_bump)
        return s

class GaussianCopulaSpreadPricer:
    nb_integration_points = 500
    nb_stdevs = 7

    def __init__(self, pos_fwd, neg_fwd, strike, pos_surface, neg_surface, corr, te):
        assert pos_fwd > 0 > neg_fwd
        # self.unit = units(strike)
        self.pos_fwd = pos_fwd
        self.neg_fwd = -neg_fwd
        self.pos_surface = pos_surface
        self.neg_surface = neg_surface
        self.strike = strike
        self.corr = corr
        self.te = te

    def option_value(self, option_type, intrinsic=False):
        intrinsic_value = Black76Option(self.pos_fwd, self.neg_fwd + self.strike, 0, self.te, option_type).value
        if intrinsic:
            return intrinsic_value
        call_premium = self.call_value()
        if option_type == "c":
            return max(intrinsic_value, call_premium)
        else:
            return max(intrinsic_value, call_premium + self.neg_fwd - self.pos_fwd + self.strike)

    def call_value(self):
        k_min, k_max = self._integration_bounds()
        dk = (k_max - k_min) / self.nb_integration_points
        lower, upper = -np.inf * np.ones(2), np.inf * np.ones(2)
        correl = np.array(self.corr)
        integral = 0
        k_ys = np.exp([k_min + i * dk for i in range(self.nb_integration_points + 1)])
        k_xs = k_ys + self.strike
        lower_0s = norm.ppf(self.cdf(POS, k_xs))
        upper_1s = norm.ppf(self.cdf(NEG, k_ys))
        for k_y, lower_0, upper_1 in zip(k_ys, lower_0s, upper_1s):
            lower[0] = lower_0
            upper[1] = upper_1
            cdf_xy = self.norm2_cdf(lower=lower, upper=upper, correl=correl)
            integral += cdf_xy * k_y * dk
        return integral

    def _integration_bounds(self):
        k_min, k_max = np.inf, 0
        for fwd, surface, strike_adj in [
            (self.pos_fwd, self.pos_surface, self.strike),
            (self.neg_fwd, self.neg_surface, 0)
        ]:
            k_min = min(k_min, fwd * exp(surface.get_inv_cdf(self.te, norm.cdf(-self.nb_stdevs))) - strike_adj)
            k_max = max(k_max, fwd * exp(surface.get_inv_cdf(self.te, norm.cdf(self.nb_stdevs))) - strike_adj)

        return log(max(1e-6, k_min)), log(max(1e-6, k_max))

    def cdf(self, sign, k):
        _k = np.atleast_1d(k)
        res = np.zeros_like(_k)
        mask = _k > 0
        surface = self.pos_surface if sign == POS else self.neg_surface
        fwd = self.pos_fwd if sign == POS else self.neg_fwd
        res[mask] = surface.get_cdf(self.te, np.log(_k[mask] / fwd))
        return res

    @staticmethod
    def norm2_cdf(lower, upper, correl):
        if lower[0] == np.inf or lower[1] == np.inf or upper[0] == -np.inf or upper[1] == -np.inf:
            return 0.

        mean = np.zeros(2)
        cov = np.array([
            [1.0, correl],
            [correl, 1.0],
        ])

        return  multivariate_normal(mean=mean, cov=cov).cdf(upper, lower_limit=lower)




def kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                       r=0.0, option_type="c"):
    expiry_time = time_between(pricing_date, expiry_date)
    return KirkSpreadPricer(pos_fwd=pos_fwd_price, neg_fwd=-neg_fwd_price, strike=strike, pos_vol=pos_vol,
                            neg_vol=neg_vol, corr=corr, te=expiry_time, r=r, option_type=option_type)


def kirk_price(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.option_value(intrinsic=False)


def kirk_delta(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.delta_pos_leg(intrinsic=False), option.delta_neg_leg(intrinsic=False)-1.0


def kirk_gamma(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.gamma_pos_leg(intrinsic=False), option.gamma_neg_leg(intrinsic=False)


def kirk_vega(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
              r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.vega(intrinsic=False)


def kirk_vega_by_leg(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
              r=0.0, option_type="c"):
    """Vega for constituent vols"""
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.vega_pos_leg(), option.vega_neg_leg()


def kirk_theta(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.theta(intrinsic=False) / 365


def kirk_corr_sensitivity(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
              r=0.0, option_type="c"):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                r, option_type)
    return option.corr_sensitivity()