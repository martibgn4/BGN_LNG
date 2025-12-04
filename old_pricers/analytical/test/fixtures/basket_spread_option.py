import math
import warnings

import numpy as np

from trade_data import OptionType, OPTION_TYPES
from .kirk_spread_option import KirkSpreadOption


class BasketSpreadOption:
    """
    Class for the pricing of a generic spread option
    After expiry, it is not possible to price such an option

    Pricing is done by moment-matching a lognormal variable on the positive leg of the spread,
    another one on the negative leg,
    and throwing those two into a kirk formula
    """

    def __init__(self, forwards, weights, strike, te, vols, corrs, put_call=OptionType.CALL):
        r"""
        Constructor of the spread option pricer

        Prices an option which payoff is \f$(\sum_{i=1}^n w_i F_{T,T_1,T_2}^i - K)^+\f$

        Or if put_call=PUT \f$(\sum_{i=1}^n w_i F_{T,T_1,T_2}^i - K)^-\f$
        """
        assert put_call in OPTION_TYPES, f"Unexpected option type: {put_call} not in {OPTION_TYPES}"
        self.put_call = put_call
        put_call_multiplier = {OptionType.CALL: 1, OptionType.PUT: -1}[put_call]
        self.forwards = forwards
        self.weights = weights * put_call_multiplier
        self.wf = [self.forwards[i] * self.weights[i] for i in range(len(self.weights))]
        self.strike = strike * put_call_multiplier
        self.te = te
        self.vols = vols
        self.corrs = corrs
        self.pos = [i for i in range(len(weights)) if self.weights[i] > 0]
        self.neg = [i for i in range(len(weights)) if self.weights[i] < 0]
        assert self.te >= 0, f"Time to expiry should be positive ({self.te})"
        self.overrides = None

    def _get_covariance(self, indexes1, indexes2):
        r"""
        Gives the covariance of \f$(\sum_{i \in I}^n w_i F_{T,T_1,T_2}^i)\f$ and
        \f$(\sum_{j \in J}^n w_j F_{T,T_1,T_2}^j)\f$
        """
        if len(indexes1) * len(indexes2) == 0:
            return 0
        total_wf1 = sum([self.wf[i] for i in indexes1])
        total_wf2 = sum([self.wf[i] for i in indexes2])
        result = 0.0
        for i1 in indexes1:
            for i2 in indexes2:
                result += self.wf[i1] * self.wf[i2] * \
                    math.exp(self.corrs[i1, i2] * self.vols[i1] * self.vols[i2] * self.te)
        tmp = result / (total_wf1 * total_wf2)
        result = np.log(tmp) / self.te
        return result

    @property
    def moment_matched_vols_and_corrs(self):
        pos_vol = np.sqrt(self._get_covariance(self.pos, self.pos))
        neg_vol = np.sqrt(self._get_covariance(self.neg, self.neg))
        covariance = self._get_covariance(self.pos, self.neg)
        correlation = covariance / (pos_vol * neg_vol)
        if self.put_call is OptionType.CALL:
            return pos_vol, neg_vol, correlation
        else:
            return neg_vol, pos_vol, correlation

    def override_moment_matched_vols_and_corrs(self, long_vol, short_vol, correlation):
        self.overrides = (long_vol, short_vol, correlation)

    @property
    def value(self):
        # if the sum is empty we take the units from the strike with self.strike * 0
        sum_pos_weighted_forward = sum([self.wf[i] for i in self.pos]) if self.pos != [] else self.strike * 0
        sum_neg_weighted_forward = sum([self.wf[i] for i in self.neg]) if self.neg != [] else self.strike * 0

        if self.te > 0:
            if self.overrides:
                if self.put_call is OptionType.CALL:
                    pos_vol, neg_vol, correlation = self.overrides
                else:
                    neg_vol, pos_vol, correlation = self.overrides
            else:
                pos_vol = np.sqrt(self._get_covariance(self.pos, self.pos))
                neg_vol = np.sqrt(self._get_covariance(self.neg, self.neg))
                covariance = self._get_covariance(self.pos, self.neg)
                pos_vol = np.atleast_1d(pos_vol)
                neg_vol = np.atleast_1d(neg_vol)
                covariance = np.atleast_1d(covariance)

                # Div 0 can occur here, but we tidy up afterwards...
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    correlation = covariance / (pos_vol * neg_vol)
                filt = np.logical_or(pos_vol == 0.0, neg_vol == 0.0)
                correlation[filt] = 0.0

        else:
            pos_vol = None
            neg_vol = None
            correlation = None

        if self.strike >= 0:
            k = self.strike
            s1 = sum_pos_weighted_forward
            s2 = -sum_neg_weighted_forward
            vol1 = pos_vol
            vol2 = neg_vol
            baseline = 0.0
        else:
            k = -self.strike
            s1 = -sum_neg_weighted_forward
            s2 = sum_pos_weighted_forward
            baseline = s2 - s1 - self.strike
            vol1 = neg_vol
            vol2 = pos_vol

        if self.te > 0:
            kirk_value = KirkSpreadOption(
                s1=s1,
                s2=s2,
                k=k,
                t=self.te,
                vol1=vol1,
                vol2=vol2,
                corr12=correlation).value
        else:
            kirk_value = max(s1 - s2 - k, 0)

        return kirk_value + baseline


