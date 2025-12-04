import math
from scipy.stats import norm
import numpy as np


class KirkSpreadOption:

    def __init__(self, s1, s2, k, t, vol1, vol2, corr12):
        self.s1 = np.atleast_1d(s1)
        self.s2 = np.atleast_1d(s2)
        self.k = k
        self.t = t
        self.vol1 = np.atleast_1d(vol1)
        self.vol2 = np.atleast_1d(vol2)
        self.corr12 = np.atleast_1d(corr12)

    @property
    def value(self):
        """
        Prices a spread option using Kirk's Approximation
        @returns Analytical Call Premium for a spread option via the Margrabe formula
        """

        strike = self.s2 + self.k  # Combined strike equal to asset2 + asset3+ k, e.g. gas + carbon + k
        b2 = self.s2 / strike  # Weighting factor for asset 2
        # Calculate portfolio variance
        var4s = (self.vol1 ** 2 + (self.vol2 ** 2 * b2 ** 2) - (2 * self.corr12 * self.vol1 * self.vol2 * b2))
        vol4s = np.sqrt(var4s)  # Convert variance into volatility
        S_A = self.s1 / strike  # Ratio of long/ strike, where strike = asset2 + asset3 + k

        v4roott = vol4s * math.sqrt(self.t)
        d1 = (np.log(S_A) + 0.5 * vol4s ** 2 * self.t) / v4roott
        d2 = d1 - v4roott

        # Calculate the cumulative probability distribution function
        # for variable that is normally distributed with a mean of zero
        # and a standard deviation of 1
        CND1 = norm.cdf(d1)
        CND2 = norm.cdf(d2)
        callpremium = strike * (S_A * CND1 - CND2)

        filt = self.s1 == 0
        callpremium[filt] = 0

        return callpremium


