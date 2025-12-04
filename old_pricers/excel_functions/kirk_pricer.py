import xlwings as xw

from thorn.api import kirk_price, kirk_delta, kirk_gamma, kirk_vega, kirk_theta, kirk_vega_by_leg
from thorn_excel.utils.converter import DateConverter
from thorn_excel.utils.docstring_utils import docstring_formatter
from thorn_excel.utils.replay_logger_utils import replay_logger

__all__ = [
    "QAKirkPrice",
    "QAKirkDelta",
    "QAKirkGamma",
    "QAKirkVega",
    "QAKirkVegaByLeg",
    "QAKirkTheta",
]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk option price")
@replay_logger()
def QAKirkPrice(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr):
    """Computes a spread option price"""
    return kirk_price(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, option_type)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk option delta -> Tuple[pos, neg]")
@replay_logger()
def QAKirkDelta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr):
    """Computes a spread option delta"""
    return kirk_delta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, option_type)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk option gamma -> Tuple[pos, neg]")
@replay_logger()
def QAKirkGamma(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr):
    """Computes a spread option gamma"""
    return kirk_gamma(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, pos_vol, neg_vol, corr,
                      option_type)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk option vega w.r.t spread vol")
@replay_logger()
def QAKirkVega(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
               pos_vol, neg_vol, corr):
    """Computes a spread option vega"""
    return kirk_vega(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                     pos_vol, neg_vol, corr, option_type)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk option vega w.r.t constituent vols -> Tuple[pos, neg]")
@replay_logger()
def QAKirkVegaByLeg(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
               pos_vol, neg_vol, corr):
    return kirk_vega_by_leg(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                            pos_vol, neg_vol, corr, option_type)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pos_forward_price', doc="Positive Forward Price")
@xw.arg('neg_forward_price', doc="Negative Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('pos_vol', doc="Positive Volatility")
@xw.arg('neg_vol', doc="Negative Volatility")
@xw.arg('corr', doc="Correlation")
@xw.ret(doc="Kirk spread option theta")
@replay_logger()
def QAKirkTheta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr):
    """Computes a spread option theta"""
    return kirk_theta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, option_type)


