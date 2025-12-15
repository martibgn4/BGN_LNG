import xlwings as xw
from BGN_LNG.Pricers.european_option import black76_price, black76_delta, black76_gamma, black76_vega, black76_theta, \
    black76_vol_from_premium
from BGN_LNG.Pricers.kirk import kirk_theta, kirk_vega_by_leg, kirk_vega, kirk_gamma, kirk_delta, kirk_price

from BGN_LNG.Utils.datetime_utils import DateConverter
# from xlwings.conversion.standard import DateConverter


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option price")
def BGNBlack76Price(pricing_date, expiry_date, forward_price, strike, option_type, vol, r=0.0):
    """Computes an option price using Black76."""
    return black76_price(pricing_date, expiry_date, forward_price, strike, option_type, vol, r)


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option delta")
def BGNBlack76Delta(pricing_date, expiry_date, forward_price, strike, option_type, vol, r=0.0):
    """Computes an option delta using Black76."""
    return black76_delta(pricing_date, expiry_date, forward_price, strike, option_type, vol, r)


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option gamma")
def BGNBlack76Gamma(pricing_date, expiry_date, forward_price, strike, vol, r=0.0):
    """Computes an option gamma using Black76."""
    return black76_gamma(pricing_date, expiry_date, forward_price, strike, vol, r)


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option vega")
def BGNBlack76Vega(pricing_date, expiry_date, forward_price, strike, vol, r=0.0):
    """Computes an option vega using Black76."""
    return black76_vega(pricing_date, expiry_date, forward_price, strike, vol, r)


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option theta")
def BGNBlack76Theta(pricing_date, expiry_date, forward_price, strike, option_type, vol, r=0.0):
    """Computes an option theta using Black76."""
    return black76_theta(pricing_date, expiry_date, forward_price, strike, option_type, vol, r)


@xw.func(call_in_wizard=False)
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('premium', doc="Option premium")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('r', doc="[Optional] Interest Rate, Defaults to 0.0")
@xw.ret(doc="Black76 option implied volatility")
def BGNBlack76VolFromPremium(pricing_date, expiry_date, forward_price, strike, premium, option_type, r=0.0):
    """Computes an option implied vol using Black76."""
    return black76_vol_from_premium(
        pricing_date, expiry_date, forward_price, strike, premium, option_type, r,
    )



@xw.func(call_in_wizard=False)
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
def BGNKirkPrice(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option price"""
    return kirk_price(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkDelta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option delta"""
    return kirk_delta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkGamma(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option gamma"""
    return kirk_gamma(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, pos_vol, neg_vol, corr,
                      r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkVega(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
               pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option vega"""
    return kirk_vega(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                     pos_vol, neg_vol, corr, r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkVegaByLeg(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
               pos_vol, neg_vol, corr, r=0.0):
    return kirk_vega_by_leg(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                            pos_vol, neg_vol, corr, r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkTheta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option theta"""
    return kirk_theta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)


@xw.func(call_in_wizard=False)
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
def BGNKirkCorrSensitivity(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes a spread option sensitivity to correlation"""
    return kirk_theta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)