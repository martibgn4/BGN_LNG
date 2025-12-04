
class SpreadOption:
    """ Defines a Spread Option Object between any two underlyings as long as
     initial prices, strike, vols, correlation, and expiry are known"""
    def __init__(self, s1, s2, K, sigma1, sigma2, rho, T, r):
        self.s1 = s1
        self.s2 = s2
        self.K = K
        self.sigma = sigma1 + sigma2 + rho
        self.T = T
        self.r = r

    @property
    def kirk_price(self):
        return 0.0

    @property
    def mc_price(self):
        return 1.0

    @property
    def delta(self):
        return 1.0

    @property
    def gamma(self):
        return 1.0

    @property
    def vega(self):
        return 1.0

    @property
    def theta(self):
        return 1.0