import math
from datetime import date
from numbers import Real

from general_utils import time_between
from thorn.core.pricers.analytical.normal_option import BachelierOption
from trade_data import OptionType

__all__ = [
    "bachelier_option",
    "bachelier_price",
    "bachelier_delta",
    "bachelier_gamma",
    "bachelier_vega",
    "bachelier_theta",
    "bachelier_vol_from_premium"
]


def bachelier_option(pricing_date: date, expiry_date: date, fwd_price: Real, strike: Real, vol: Real,
                     discount_factor: Real = 1.0, option_type=OptionType.CALL):
    option_type = OptionType.parse(option_type)
    expiry_time = time_between(pricing_date, expiry_date)
    r = -math.log(discount_factor) / expiry_time
    return BachelierOption(s=fwd_price, k=strike, sigma=vol, te=expiry_time, call_put=option_type, r=r)


def bachelier_price(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.value


def bachelier_delta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.delta


def bachelier_gamma(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor):
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor)
    return option.gamma


def bachelier_vega(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor):
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor)
    return option.vega


def bachelier_theta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.theta / 365


def bachelier_vol_from_premium(
    pricing_date, expiry_date, fwd_price, strike, premium, option_type=OptionType.CALL, discount_factor=1.0,
):
    vol = None
    option = bachelier_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.get_vol_from_premium(premium)


