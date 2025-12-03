import xlwings as xw

from BGN_LNG.Pricers.kirk_pricer import kirk_price

@xw.func
def bgn_kirk_price(s1, s2, sigma1, sigma2, corr):
    return kirk_price(s1, s2, sigma1, sigma2, corr)