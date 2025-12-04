import xlwings as xw

from thorn.api import (
    bachelier_price, bachelier_delta, bachelier_gamma, bachelier_vega, bachelier_theta, bachelier_vol_from_premium
)
from thorn_excel.utils.converter import DateConverter
from thorn_excel.utils.docstring_utils import docstring_formatter
from thorn_excel.utils.replay_logger_utils import replay_logger

__all__ = [
    "QABachelierPrice",
    "QABachelierDelta",
    "QABachelierGamma",
    "QABachelierVega",
    "QABachelierTheta",
    "QABachelierVolFromPremium",
]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility (20 == 20%)")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option price")
@replay_logger()
def QABachelierPrice(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes an option price using Black76."""
    return bachelier_price(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility (20 == 20%)")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option delta")
@replay_logger()
def QABachelierDelta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes option delta"""
    return bachelier_delta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility (20 == 20%)")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option gamma")
@replay_logger()
def QABachelierGamma(pricing_date, expiry_date, forward_price, strike, vol, discount_factor=1.0):
    """Computes option gamma"""
    return bachelier_gamma(pricing_date, expiry_date, forward_price, strike, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility (20 == 20%)")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option vega")
@replay_logger()
def QABachelierVega(pricing_date, expiry_date, forward_price, strike, vol, discount_factor=1.0):
    """Computes option vega"""
    return bachelier_vega(pricing_date, expiry_date, forward_price, strike, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility (20 == 20%)")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option theta")
@replay_logger()
def QABachelierTheta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes option theta"""
    return bachelier_theta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('premium', doc="Option premium")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Bachelier option implied volatility")
@replay_logger()
def QABachelierVolFromPremium(pricing_date, expiry_date, forward_price, strike, premium, option_type, discount_factor=1.0):
    """Computes an option implied vol"""
    return bachelier_vol_from_premium(
        pricing_date, expiry_date, forward_price, strike, premium, option_type, discount_factor,
    )


