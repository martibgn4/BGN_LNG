import numpy as np
from scipy.stats import truncnorm

def truncated_normal_sample(n, L, U, mu=0.0, sigma=1.0, random_state=None):
    # Standardize bounds to N(0,1)
    a = (L - mu) / sigma
    b = (U - mu) / sigma
    rng = np.random.default_rng(random_state)
    return truncnorm.rvs(a=a, b=b, loc=mu, scale=sigma, size=n, random_state=rng)


def jump_rv(n, p=0.10, mu=0.5, sigma=0.1, random_state=None):
    """
    Draw n samples from a Bernoulli-Normal mixture:
      X = J * Z, J ~ Bernoulli(p), Z ~ Normal(mu, sigma^2)
    """
    rng = np.random.default_rng(random_state)
    J = rng.random(n) < p                  # 0/1 mask for jumps
    Z = rng.normal(loc=mu, scale=sigma, size=n)
    return J.astype(float) * Z