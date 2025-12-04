import math

from general_utils import time_between
from thorn.core.pricers.analytical.kirk import KirkParameters, KirkSpreadPricer
from trade_data import OptionType


__all__ = [
    "kirk_spread_option",
    "kirk_price",
    "kirk_delta",
    "kirk_gamma",
    "kirk_vega",
    "kirk_vega_by_leg",
    "kirk_theta",
]


def kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                       option_type=OptionType.CALL):
    option_type = OptionType.parse(option_type)
    expiry_time = time_between(pricing_date, expiry_date)
    return KirkSpreadPricer(pos_fwd=pos_fwd_price, neg_fwd=-neg_fwd_price, strike=strike, pos_vol=pos_vol,
                            neg_vol=neg_vol, corr=corr, te=expiry_time, option_type=option_type)


def kirk_price(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               option_type=OptionType.CALL):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.option_value(intrinsic=False)


def kirk_delta(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               option_type=OptionType.CALL):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.delta_pos_leg(intrinsic=False), option.delta_neg_leg(intrinsic=False)


def kirk_gamma(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               option_type=OptionType.CALL):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.gamma_pos_leg(intrinsic=False), option.gamma_neg_leg(intrinsic=False)


def kirk_vega(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
              option_type=OptionType.CALL):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.vega(intrinsic=False)


def kirk_vega_by_leg(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
              option_type=OptionType.CALL):
    """Vega for constituent vols"""
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.vega_pos_leg(), option.vega_neg_leg()


def kirk_theta(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
               option_type=OptionType.CALL):
    option = kirk_spread_option(pricing_date, expiry_date, pos_fwd_price, neg_fwd_price, strike, pos_vol, neg_vol, corr,
                                option_type)
    return option.theta(intrinsic=False) / 365


