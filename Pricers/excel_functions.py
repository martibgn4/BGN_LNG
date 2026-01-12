import xlwings as xw
from BGN_LNG.Pricers.european_option import black76_price, black76_delta, black76_gamma, black76_vega, black76_theta, \
    black76_vol_from_premium
from BGN_LNG.Pricers.kirk import kirk_theta, kirk_vega_by_leg, kirk_vega, kirk_gamma, kirk_delta, kirk_price, \
    kirk_corr_sensitivity

from BGN_LNG.Utils.datetime_utils import DateConverter
from BGN_LNG.adhoc_scripts.OmanDec25 import price_basket_option_mc, price_long_basket_option_mc


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
    return kirk_corr_sensitivity(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
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
@xw.ret(doc="All Kirk Option risks, single call")
def BGNKirkAllRisks(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike, option_type,
                pos_vol, neg_vol, corr, r=0.0):
    """Computes main relevant greeks of a spread option, via Kirk approximation"""
    price = kirk_price(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    delta1, delta2 = kirk_delta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    gamma1, gamma2 = kirk_gamma(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    vega1, vega2 = kirk_vega_by_leg(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    theta = kirk_theta(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    corr_sens = kirk_corr_sensitivity(pricing_date, expiry_date, pos_forward_price, neg_forward_price, strike,
                      pos_vol, neg_vol, corr, r, option_type)
    return price, delta1, delta2, gamma1, gamma2, vega1, vega2, theta, corr_sens


@xw.func(call_in_wizard=False)
@xw.arg('vol_u1', doc="Yearly volatility underlying 1")
@xw.arg('vol_u2', doc="Yearly volatility underlying 2")
@xw.arg('vol_freight', doc="Yearly volatility freight underlying")
@xw.arg('corr_u1u2', doc="Correlation underlying 1 and 2")
@xw.arg('corr_u1freight', doc="Correlation underlying 1 and freight")
@xw.arg('corr_u2freight', doc="Correlation underlying 2 and freight")
@xw.arg('r', doc="Yearly Interest Rat")
@xw.arg('T', doc="Option maturity")
@xw.arg('route_days', doc="Days from u1 to u2")
@xw.arg('mmbtu_start', doc="Starting MMBTU cargo")
@xw.arg('mmbtu_end', doc="Ending MMBTU cargo")
@xw.arg('u1_price', doc="Current u1 price for delivery, in usd/mmbtu")
@xw.arg('u2_price', doc="Current u2 price for delivery, in usd/mmbtu")
@xw.arg('freight_price', doc="Current freight price for delivery, in USD/day")
@xw.arg('N_paths', doc="Total simulation paths")
@xw.arg('extra_costs', doc="Extra costs in USD")
@xw.arg('mc_seed', doc="Seed to be used in MC simulations")
@xw.ret(doc="LNG Option price via MC")
def BGNLNGOption_mc(
        vol_u1, vol_u2, vol_freight, corr_u1u2, corr_u1freight, corr_u2freight,
        r, T, route_days, mmbtu_start, mmbtu_end, u1_price, u2_price, freight_price,
        N_paths=10000, extra_costs=0.0, mc_seed=1):
    """Computes a spread option price"""
    return price_basket_option_mc(
        vol_u1, vol_u2, vol_freight, corr_u1u2, corr_u1freight, corr_u2freight,
        1.0, r, T, route_days, mmbtu_start, mmbtu_end,
        u1_price, u2_price, freight_price, int(N_paths), mc_seed=mc_seed, extra_costs=extra_costs)





@xw.func(call_in_wizard=False)
@xw.arg('vol_u1', doc="Yearly volatility underlying 1")
@xw.arg('vol_u2', doc="Yearly volatility underlying 2")
@xw.arg('vol_u3', doc="Yearly volatility underlying 3")
@xw.arg('corr_u1u2', doc="Correlation underlying 1 and 2")
@xw.arg('corr_u1u3', doc="Correlation underlying 1 and 3")
@xw.arg('corr_u2u3', doc="Correlation underlying 2 and 3")
@xw.arg('weight_u1', doc="Weight for underlying 1")
@xw.arg('weight_u2', doc="Weight for underlying 2")
@xw.arg('weight_u3', doc="Weight for underlying 3")
@xw.arg('r', doc="Yearly Interest Rat")
@xw.arg('T', doc="Option maturity")
@xw.arg('price_u1', doc="Current u1 price for delivery, in usd/mmbtu")
@xw.arg('price_u2', doc="Current u2 price for delivery, in usd/mmbtu")
@xw.arg('price_u3', doc="Current u3 price for delivery, in usd/mmbtu")
@xw.arg('N_paths', doc="Total simulation paths")
@xw.arg('extra_costs', doc="Extra costs in USD")
@xw.arg('mc_seed', doc="Seed to be used in MC simulations")
@xw.ret(doc="LNG Basket Option price via MC")
def BGNLNGBasketOption_mc(
        vol_u1, vol_u2, vol_u3, corr_u1u2, corr_u1u3, corr_u2u3,
        weight_u1, weight_u2, weight_u3, r, T,
        price_u1, price_u2, price_u3,
        N_paths=10000, extra_costs=0.0, mc_seed=1):
    """Computes a spread option price"""
    return price_long_basket_option_mc(vol_u1, vol_u2, vol_u3,
                           corr_u1u2, corr_u1u3, corr_u2u3,
                           weight_u1, weight_u2, weight_u3,
                           r, T,
                           price_u1, price_u2, price_u3,
                           int(N_paths), antithetic=True, mc_seed=mc_seed,
                           extra_costs = extra_costs)
