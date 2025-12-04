import math
from collections import defaultdict

import xlwings as xw
import numpy as np
from scipy.stats import norm

from thorn.api import (
    black76_delta, black76_gamma, black76_price, black76_theta, black76_vega, build_vanilla_smile,
    vanilla_smile_breakeven_price_move, vanilla_smile_delta, vanilla_smile_gamma, vanilla_smile_pillar_vega,
    vanilla_smile_price, vanilla_smile_theta, vanilla_smile_vega, vanilla_smile_vol, Pillar,
    vanilla_smile_vol_at_pillar, vanilla_smile_strike_at_pillar, black76_vol_from_premium, PillaredSmileCollection
)
from thorn_excel.functions.errors import ThornExcelFunctionError
from thorn_excel.utils.cache import CacheConverter, cache_results
from thorn_excel.utils.converter import DateConverter, TenorConverter, BooleanConverter
from thorn_excel.utils.docstring_utils import docstring_formatter
from thorn_excel.utils.profiler import profiler
from thorn_excel.utils.replay_logger_utils import replay_logger
from time_period import DateRange, tenor_to_strip, Month
from trade_data import OptionType

__all__ = [
    "QAVanillaSmile",
    "QAVanillaSmileVol",
    "QAVanillaSmileVolAtPillar",
    "QAVanillaSmileStrikeAtPillar",
    "QAVanillaBreakevenPriceMove",
    "QAVanillaSmilePrice",
    "QAVanillaSmileDelta",
    "QAVanillaSmileGamma",
    "QAVanillaSmileVega",
    "QAVanillaSmileSkewVega",
    "QAVanillaSmileTheta",
    "QABlack76Price",
    "QABlack76Delta",
    "QABlack76Gamma",
    "QABlack76Vega",
    "QABlack76Theta",
    "QABlack76VolFromPremium",
    "QAMonthlyEuropean",
]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('calibration_date', DateConverter, doc="Smile Calibration Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('pillars', doc="Pillars in delta star format")
@xw.arg('vols', doc="Vol Smile")
@xw.arg('relative', doc="[Optional] Are vols specified with relation to ATM* pillar")
@xw.arg('include_theta', doc="[Optional] Set to TRUE to calculate Theta using this smile")
@xw.arg('include_vega', doc="[Optional] Set to TRUE to calculate Vega using this smile")
@xw.arg('include_smile_vega', doc="[Optional] Set to TRUE to calculate Smile Vega using this smile")
@xw.arg('cache_label', doc='[Optional] User defined name used for storage and retrieval of values')
@xw.ret(doc="A label to be used to reference the built vanilla smile in other functions.")
@replay_logger()
@profiler()
@cache_results("VSmile")
def QAVanillaSmile(pricing_date, calibration_date, expiry_date, pillars, vols,
                   relative=False, include_theta=False, include_vega=False, include_smile_vega=False, cache_label=None):
    """Builds a vanilla vol smile and returns a handle."""
    if pricing_date != calibration_date:
        raise ThornExcelFunctionError(f"Pricing Date != Calibration Date [{pricing_date} != {calibration_date}] "
                                      "Internally only the pricing date is considered")
    return build_vanilla_smile(
        pricing_date, expiry_date, pillars, vols, relative, include_theta, include_vega, include_smile_vega
    )


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.ret(doc="Option volatility calculated from the vanilla smile")
@replay_logger()
def QAVanillaSmileVol(vanilla_smile, forward_price, strike):
    """Compute the vol from a vanilla smile with respect to a forward price and a strike price."""
    return vanilla_smile_vol(vanilla_smile, forward_price, strike)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('pillar', doc="Delta Pillar (eg '0.1*' or '0.75')")
@xw.ret(doc="Option volatility calculated from the vanilla smile")
@replay_logger()
def QAVanillaSmileVolAtPillar(vanilla_smile, pillar):
    """Compute the vol from a vanilla smile with respect to a forward price and a strike price."""
    return vanilla_smile_vol_at_pillar(vanilla_smile, Pillar.parse(str(pillar)))


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('pillar', doc="Delta Pillar (eg '0.1*' or '0.75')")
@xw.ret(doc="Option strike calculated from the vanilla smile. e.g 0.5* ==> 1")
@replay_logger()
def QAVanillaSmileStrikeAtPillar(vanilla_smile, pillar):
    """Compute the unitised strike (a.k.a leverage) from a vanilla smile (assumes forward or 1 and is a call)"""
    return vanilla_smile_strike_at_pillar(vanilla_smile, Pillar.parse(str(pillar)))


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('approx', doc="[Optional] Whether to approximate in the absence of theta")
@xw.ret(doc="Option breakeven daily price move calculated from the vanilla smile")
@replay_logger()
def QAVanillaBreakevenPriceMove(vanilla_smile, forward_price, strike, approx=False):
    """Compute the break-even daily price move from a vanilla smile with respect to a forward price and a strike price."""
    return vanilla_smile_breakeven_price_move(vanilla_smile, forward_price, strike, approx)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Discounted option price")
@replay_logger()
def QAVanillaSmilePrice(vanilla_smile, option_type, forward_price, strike, discount_factor=1.0):
    """Computes an option price from a vanilla vol smile."""
    return vanilla_smile_price(vanilla_smile, option_type, forward_price, strike, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Smile-aware option delta")
@replay_logger()
def QAVanillaSmileDelta(vanilla_smile, option_type, forward_price, strike, discount_factor=1.0):
    """Computes a smile-aware option delta."""
    return vanilla_smile_delta(vanilla_smile, option_type, forward_price, strike, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Smile-aware option gamma")
@replay_logger()
def QAVanillaSmileGamma(vanilla_smile, forward_price, strike, discount_factor=1.0):
    """Computes a smile-aware option gamma."""
    return vanilla_smile_gamma(vanilla_smile, forward_price, strike, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Smile-aware option vega")
@replay_logger()
def QAVanillaSmileVega(vanilla_smile, forward_price, strike, discount_factor=1.0):
    """Computes a smile-aware option vega."""
    return vanilla_smile_vega(vanilla_smile, forward_price, strike, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('pillar', doc="Pillar")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Smile-aware option skew vega")
@replay_logger()
def QAVanillaSmileSkewVega(vanilla_smile, forward_price, strike, pillar, discount_factor=1.0):
    """Computes a smile-aware option skew vega."""
    return vanilla_smile_pillar_vega(vanilla_smile, forward_price, strike, pillar, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('vanilla_smile', CacheConverter, doc="Handle to QAVanillaSmile")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('discount_factor_t0', doc="[Optional] Discount Factor at Pricing Date, Defaults to 1.0")
@xw.arg('discount_factor_t1', doc="[Optional] Discount Factor at Next Date, Defaults to 1.0")
@xw.ret(doc="Smile-aware option theta")
@replay_logger()
def QAVanillaSmileTheta(vanilla_smile, option_type, forward_price, strike,
                        discount_factor_t0=1.0, discount_factor_t1=1.0):
    """Computes an option theta from a vanilla vol smile."""
    return vanilla_smile_theta(
        vanilla_smile, option_type, forward_price, strike, discount_factor_t0, discount_factor_t1
    )


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option price")
@replay_logger()
def QABlack76Price(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes an option price using Black76."""
    return black76_price(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option delta")
@replay_logger()
def QABlack76Delta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes an option delta using Black76."""
    return black76_delta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option gamma")
@replay_logger()
def QABlack76Gamma(pricing_date, expiry_date, forward_price, strike, vol, discount_factor=1.0):
    """Computes an option gamma using Black76."""
    return black76_gamma(pricing_date, expiry_date, forward_price, strike, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('vol', doc="Volatility")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option vega")
@replay_logger()
def QABlack76Vega(pricing_date, expiry_date, forward_price, strike, vol, discount_factor=1.0):
    """Computes an option vega using Black76."""
    return black76_vega(pricing_date, expiry_date, forward_price, strike, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('vol', doc="Volatility")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option theta")
@replay_logger()
def QABlack76Theta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor=1.0):
    """Computes an option theta using Black76."""
    return black76_theta(pricing_date, expiry_date, forward_price, strike, option_type, vol, discount_factor)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc="Pricing Date")
@xw.arg('expiry_date', DateConverter, doc="Expiry Date")
@xw.arg('forward_price', doc="Forward Price")
@xw.arg('strike', doc="Strike price")
@xw.arg('premium', doc="Option premium")
@xw.arg('option_type', doc="Call or Put / C or P")
@xw.arg('discount_factor', doc="[Optional] Discount Factor, Defaults to 1.0")
@xw.ret(doc="Black76 option implied volatility")
@replay_logger()
def QABlack76VolFromPremium(pricing_date, expiry_date, forward_price, strike, premium, option_type, discount_factor=1.0):
    """Computes an option implied vol using Black76."""
    return black76_vol_from_premium(
        pricing_date, expiry_date, forward_price, strike, premium, option_type, discount_factor,
    )


_vcache = defaultdict(dict)  # {(del_period, exp): {(pd, pillars, smile): vsmile}}


def vcache(psc: PillaredSmileCollection, del_period, expiry_date=None):
    expiry_date = expiry_date or max(psc.get_expiry_dates(del_period))
    key1 = (del_period, expiry_date)
    smile = tuple(psc.get_smile(*key1))
    key2 = (psc.calibration_date, psc.pillars, smile)
    if key2 not in _vcache[del_period]:
        vsmile = build_vanilla_smile(
            pricing_date=psc.calibration_date, expiry_date=expiry_date, pillars=psc.pillars, vols=smile,
            relative_vols=False, include_theta=True, include_vega=True, include_skew_vega=False
        )
        _vcache[del_period][key2] = vsmile
    return _vcache[del_period][key2]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('fwd_curve', CacheConverter, doc="Handle to Fwd Curve")
@xw.arg('psc', CacheConverter, doc="Handle to PSC")
@xw.arg('period', TenorConverter, doc="Period")
@xw.arg('cross', doc="Delta Cross Price")
@xw.arg('strikes', np.array, ndim=1, dtype=float, doc="Array of Strike prices")
@xw.arg('option_types', np.array, ndim=1, dtype=str, doc="Array of Call or Put / C or P")
@xw.arg('weights', np.array, ndim=1, dtype=float, doc="[Optional] Array of weights, Defaults to [1, ..., 1]")
@xw.arg('fwd_equal_to_cross', BooleanConverter,
        doc="[Optional] Set reference forward price to cross. If provided fwd_curve is ignored")
@xw.ret(doc="[pv, delta, gamma, vega (%), theta, b76_delta, vol_1, ..., vol_n, delta_star_1, ..., delta_star_n]")
def QAMonthlyEuropean(fwd_curve, psc, period, cross, strikes, option_types, weights=None, fwd_equal_to_cross=False):
    """Price 1 or sum of many strips of (weighted) monthly europeans"""
    weights = np.ones_like(strikes) if weights is None else weights
    assert len(strikes) == len(option_types) == len(weights), \
        "strike(s), option_type(s) and weight(s) (if provided) are of different lengths"
    fwd_curve = fwd_curve.value
    psc = psc.value
    period = DateRange.parse(period.value)
    legs = [
        (float(i), OptionType.parse(j), float(k))
        for i, j, k in zip(strikes, option_types, weights)
    ]

    pv_by_leg = np.zeros_like(strikes)
    delta_by_leg = np.zeros_like(strikes)
    b76_delta_by_leg = np.zeros_like(strikes)
    gamma_by_leg = np.zeros_like(strikes)
    theta_by_leg = np.zeros_like(strikes)
    vol_by_leg = np.zeros_like(strikes)
    vega_by_leg = np.zeros_like(strikes)
    delta_star_by_leg = np.zeros_like(strikes)

    fwd = cross if fwd_equal_to_cross else fwd_curve.get_price(period).value
    period_duration = period.duration()
    strip = tenor_to_strip(period, Month)

    for tenor in strip:
        m_fwd = cross if fwd_equal_to_cross else fwd_curve.get_price(tenor).value
        duration_ratio = tenor.duration() / period_duration
        expiry_date = max(psc.get_expiry_dates(tenor))

        v_smile = vcache(psc, tenor)
        for i, (strike, opt, weight) in enumerate(legs):
            scalar = duration_ratio * weight

            pv_by_leg[i] += scalar * vanilla_smile_price(v_smile, opt, m_fwd, strike, 1)
            delta_by_leg[i] += scalar * vanilla_smile_delta(v_smile, opt, m_fwd, strike, 1)
            gamma_by_leg[i] += scalar * vanilla_smile_gamma(v_smile, m_fwd, strike, 1)
            theta_by_leg[i] += scalar * vanilla_smile_theta(v_smile, opt, m_fwd, strike, 1, 1)
            m_vega = scalar * vanilla_smile_vega(v_smile, m_fwd, strike, 1)
            vega_by_leg[i] += m_vega

            m_vol = vanilla_smile_vol(v_smile, m_fwd, strike)
            vol_by_leg[i] += m_vega * m_vol

            m_atm_vol = vanilla_smile_vol(v_smile, m_fwd, m_fwd)
            delta_star_by_leg[i] += duration_ratio * norm.cdf(
                math.log(m_fwd / strike) /
                (m_atm_vol * math.sqrt((expiry_date - psc.calibration_date).days / 365.0))
            )

            m_cross = cross - fwd + m_fwd
            b76_delta_by_leg[i] += scalar * black76_delta(
                psc.calibration_date, expiry_date, m_cross, strike, opt, m_vol, 1)

    vol = vol_by_leg / vega_by_leg
    adj_pv_2 = pv_by_leg - (b76_delta_by_leg * (fwd - cross))

    return [
        adj_pv_2.sum(), delta_by_leg.sum(), gamma_by_leg.sum(), vega_by_leg.sum() / 100.0, theta_by_leg.sum(),
        b76_delta_by_leg.sum(),
        *vol,
        *delta_star_by_leg
    ]


