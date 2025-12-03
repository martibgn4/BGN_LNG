
import math

def d1(F, K, sigma, T):
    return (math.log(F/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))

def d2(F, K, sigma, T):
    return d1(F, K, sigma, T) - sigma*math.sqrt(T)
