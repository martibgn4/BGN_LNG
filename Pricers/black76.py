import math
from scipy.stats import norm
from BGN_LNG.Pricers.math_utils import d1, d2

class Black76Option:
    def __init__(self, F, K, sigma, T, r=0, option_type="call"):
        self.F = F
        self.K = K
        self.sigma = sigma
        self.T = T
        self.r = r
        self.option_type = option_type

    @property
    def price(self):
        _d1 = d1(self.F, self.K, self.sigma, self.T)
        _d2 = d2(self.F, self.K, self.sigma, self.T)
        df = math.exp(-self.r * self.T)
        if self.option_type == "call":
            return df * (self.F * norm.cdf(_d1) - self.K * norm.cdf(_d2))
        else:
            return df * (self.K * norm.cdf(-_d2) - self.F * norm.cdf(-_d1))

    @property
    def delta(self):
        _d1 = d1(self.F, self.K, self.sigma, self.T)
        _d2 = d2(self.F, self.K, self.sigma, self.T)
        df = math.exp(-self.r * self.T)
        if self.option_type == "call":
            return df * norm.cdf(_d1)
        else:
            return -df * norm.cdf(-_d1)

    @property
    def gamma(self):
        _d1 = d1(self.F, self.K, self.sigma, self.T)
        pdf_d1 = norm.pdf(_d1)
        df = math.exp(-self.r * self.T)
        return df * pdf_d1 / (self.F * self.sigma * math.sqrt(self.T))

    @property
    def vega(self):
        _d1 = d1(self.F, self.K, self.sigma, self.T)
        pdf_d1 = norm.pdf(_d1)
        df = math.exp(-self.r * self.T)
        return df * self.F * pdf_d1 * math.sqrt(self.T)

    @property
    def theta(self):
        _d1 = d1(self.F, self.K, self.sigma, self.T)
        pdf_d1 = norm.pdf(_d1)
        return - (self.F * pdf_d1 * self.sigma) / (2 * math.sqrt(self.T))

def black76_price(F, K, sigma, T, r, option_type="call"):
    return Black76Option(F, K, sigma, T, r, option_type).price

def black76_greeks(F, K, sigma, T, r, greek, option_type="call"):
    """Compute Greeks for Black-76."""
    black76Option = Black76Option(F, K, sigma, T, r, option_type)
    if greek.lower() == "gamma":
        return black76Option.gamma
    elif greek.lower() == "delta":
        return black76Option.delta
    elif greek.lower() == "vega":
        return black76Option.vega
    elif greek.lower() == "theta":
        return black76Option.theta
    else:
        raise ValueError("Invalid greek '{}'".format(greek))


    return {"Delta": delta, "Vega": vega, "Gamma": gamma, "Theta": theta}

