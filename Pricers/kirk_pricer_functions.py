from BGN_LNG.Pricers.kirk import KirkSpreadPricer
from BGN_LNG.Utils.datetime_utils import get_T_and_D_from_tenor


def kirk_price_volcorr(u1, u2, s1, s2, tenor1, tenor2,
                       volcorr, pricing_date, expiry_date,
                       strike=0.0, option_type="c", method="analytic"):
    T1, D1 = get_T_and_D_from_tenor(pricing_date, tenor1)
    T2, D2 = get_T_and_D_from_tenor(pricing_date, tenor2)
    sigma1 = volcorr.get_volatility(u1, 0.0, T1, D1)
    sigma2 = volcorr.get_volatility(u2, 0.0, T2, D2)
    corr = volcorr.get_correlation(u1, u2, 0.0, T1, T2, D1, D2)

    T = (expiry_date - pricing_date).days/365

    option = KirkSpreadPricer(s1, -s2, strike, sigma1, sigma2, corr, T, option_type)

    if method == "analytic":
        price = option.option_value
    else:
        raise ValueError(f"Method {method} not implemented.")
    return price


def kirk_price(s1, s2, sigma1, sigma2, corr,
               pricing_date, expiry_date,
               strike=0.0, option_type="c", method="analytic"):
    T = (expiry_date - pricing_date).days / 365
    option = KirkSpreadPricer(s1, -s2, strike, sigma1, sigma2, corr, T, option_type)
    if method == "analytic":
        price = option.option_value
    else:
        raise ValueError(f"Method {method} not implemented.")
    return price