import math

import numpy as np

class VolCorr:
    def __init__(self, underlyings, sigmas, alphas, corr_matrix, n_factors=2):
        assert n_factors == 2, f"Only 2 factors VolCorr supported so far"
        self.underlyings = underlyings
        assert len(sigmas) == len(alphas), f"Sigmas and Alphas must have same length"
        assert len(sigmas) == n_factors*len(underlyings), f"Length of sigmas should be n_factors times underlyings"

        self.sigmas = sigmas
        self.alphas = alphas

        self.check_corr_matrix(corr_matrix)
        self.corr_matrix = corr_matrix

    def check_corr_matrix(self, matrix, rtol=1e-05, atol=1e-08):
        assert np.allclose(matrix, matrix.T, rtol=rtol, atol=atol), f"Correlation matrix should be symmetric"
        assert all(matrix.diagonal() == np.ones(matrix.shape[0])), "Correlation matrix should have 1 in diagonal"

    def get_covariance(self, u1, u2, t, T1, T2, D1, D2):
        # get covariance of underlying1, underlyng 2, as of t, delivering at T1 and T2, for a duration of D1 and D2 respectively
        # for TTF M2 and M3 as of today,
        # u1=TTF, u2=TTF, t=0, T1=30*2/365, T2=30*3/365, D1=30/365, D2=30/365
        return 2.0

    def get_volatility(self, u, t, T, D):
        # get volatility of underlying as of t, starting at T and for D duration
        # for TTF as of today on M2, would be t=0, T=start of M2 ahead - today, in years, and D = 30/365 aprox
        variance = self.get_covariance(u, u, t, T, T, D, D)
        return math.sqrt(variance)

    def get_correlation(self, u1, u2, t, T1, T2, D1, D2):
        cov = self.get_covariance(u1, u2, t, T1, T2, D1, D2)
        vol1 = self.get_volatility(u1, t, T1, D1)
        vol2 = self.get_volatility(u2, t, T2, D2)
        return cov/(vol1*vol2)