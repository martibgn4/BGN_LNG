import concurrent.futures
import math
import numpy as np

from general_utils import time_between
from thorn.core.base.pillar import Pillar, ATM_PILLAR, DeltaStarPillar, DeltaPillar
from thorn.core.market.volatility.vv.kahale_call_price_functions import KahaleSmileFitter
from thorn.core.market.volatility.vv.kahale_interpolator import get_unitized_strike_from_bs_delta
from thorn.core.pricers.analytical.european_option import Black76Option
from time_period import MktCalendar
from trade_data import OptionType
from value_object import List

from .exception import APIException


__all__ = [
    "black76_option",
    "black76_price",
    "black76_delta",
    "black76_gamma",
    "black76_vega",
    "black76_theta",
    "black76_vol_from_premium",
    "build_vanilla_smile",
    "evaluate_unitised_price",
    "vanilla_smile_breakeven_price_move",
    "vanilla_smile_delta",
    "vanilla_smile_gamma",
    "vanilla_smile_pillar_vega",
    "vanilla_smile_price",
    "vanilla_smile_theta",
    "vanilla_smile_vega",
    "vanilla_smile_vol",
    "vanilla_smile_vol_at_pillar",
    "vanilla_smile_strike_at_pillar",
]

VEGA_SHIFT_SIZE = 0.001


def build_vanilla_smile(pricing_date, expiry_date, pillars, vols,
                        relative_vols=False, include_theta=False, include_vega=False, include_skew_vega=False):
    """Build a Vanilla Smile Call Price Function"""
    time_to_expiry = time_between(pricing_date, expiry_date)
    if time_to_expiry <= 0:  # On expiry date we have infinite vol so cannot calibrate a surface
        raise APIException(f"Smile has expired on pricing date ({expiry_date} <= {pricing_date})")
    return_val = {'te_pricing': time_to_expiry}

    pillar_vol_map = {Pillar.parse(str(p)): float(v) for p, v in zip(pillars, vols)}
    pillars, vols = zip(*sorted(pillar_vol_map.items()))
    pillars = List(Pillar)(pillars)

    atm_vol = pillar_vol_map.get(ATM_PILLAR)
    if relative_vols:
        if atm_vol is None:
            raise APIException(f"Cannot input relative delta with no ATM pillar")
        abs_vols = []
        for p, v in zip(pillars, vols):
            if p == ATM_PILLAR:
                abs_vols.append(v)
            else:
                abs_vols.append(atm_vol + v)
        vols = abs_vols

    return_val['atm_vol'] = atm_vol

    fitters = {'t0_v0': KahaleSmileFitter(pillars, vols, pricing_date, expiry_date)}

    if include_vega:
        vols_up = np.array(vols) + VEGA_SHIFT_SIZE
        fitters['t0_vup'] = KahaleSmileFitter(pillars, vols_up, pricing_date, expiry_date)

        vols_down = np.array(vols) - VEGA_SHIFT_SIZE
        fitters['t0_vdown'] = KahaleSmileFitter(pillars, vols_down, pricing_date, expiry_date)

    if include_skew_vega:
        for i, pillar in enumerate(pillars):
            vols_up = np.array(vols)
            vols_up[i] += VEGA_SHIFT_SIZE
            fitters[f't0_vup_{pillar}'] = KahaleSmileFitter(pillars, vols_up, pricing_date, expiry_date)

            vols_down = np.array(vols)
            vols_down[i] -= VEGA_SHIFT_SIZE
            fitters[f't0_vdown_{pillar}'] = KahaleSmileFitter(pillars, vols_down, pricing_date, expiry_date)

    if include_theta:
        pricing_date_t1 = MktCalendar().trading_date_after(pricing_date)
        fitters['t1_v0'] = KahaleSmileFitter(pillars, vols, pricing_date_t1, expiry_date)

    fitted_smiles = _compute_smile(fitters)
    return_val.update(fitted_smiles)

    if atm_vol is None:
        return_val['atm_vol'] = vanilla_smile_vol(return_val, 1, 1)

    return return_val


def _compute_smile(fitters):
    kahale_func_args = zip(*fitters.items())

    # Only run in parallel when we are calculating skew vega
    # Use ThreadPoolExecutor as ProcessPoolExecutor has large start-up overhead
    if len(fitters) > 4:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return {label: fit for label, fit in executor.map(_run_kahale_fit, *kahale_func_args)}
    else:
        return {label: fit for label, fit in map(_run_kahale_fit, *kahale_func_args)}


def _run_kahale_fit(label, fitter):
    try:
        return label, fitter.fit()
    except Exception as e:
        raise APIException(f"Failed to calibrate surface ({label})") from e


def vanilla_smile_vol(call_functions, forward_price, strike):
    """Calculate the Vol with respect to a forward price"""
    t0_v0_cf = call_functions['t0_v0']
    te_pricing = call_functions['te_pricing']
    unitised_strike = strike / forward_price
    otm_option_type = OptionType.CALL if unitised_strike >= 1.0 else OptionType.PUT
    unitised_option_price = evaluate_unitised_price(t0_v0_cf, forward_price, strike, otm_option_type)
    vol = Black76Option(1, unitised_strike, None, te_pricing, otm_option_type).get_vol_from_premium(
        unitised_option_price, "Secant"
    )
    if np.isnan(vol):
        return "Failed to calculate Vol for the inputs provided"
    return vol


def vanilla_smile_strike_at_pillar(call_functions, pillar):
    """Calculate the unitised strike from smile for pillar (assumes forward is 1 and is a call)"""
    assert isinstance(pillar, (DeltaStarPillar, DeltaPillar)), \
        f"{pillar} should be of type DeltaPillar or DeltaStarPillar"
    if isinstance(pillar, DeltaStarPillar):
        te_pricing = call_functions['te_pricing']
        atm_vol = call_functions['atm_vol']
        strike = Black76Option(1, None, atm_vol, te_pricing).strike_from_delta_star(pillar.delta_star)
    else:  # DeltaPillar
        te_pricing = call_functions['te_pricing']
        strike = get_unitized_strike_from_bs_delta(call_functions['t0_v0'], pillar.delta, te_pricing)
    return strike


def vanilla_smile_vol_at_pillar(call_functions, pillar):
    """Calculate the Vol with respect to a forward price"""
    strike = vanilla_smile_strike_at_pillar(call_functions, pillar)
    return vanilla_smile_vol(call_functions, 1, strike)


def vanilla_smile_breakeven_price_move(vanilla_smile, forward_price, strike, approx=False):
    BUS_DAYS_IN_YEAR = 252.0
    CAL_DAYS_IN_WEEK = 7.0
    BUS_DAYS_IN_WEEK = 5.0

    def exact_breakeven_move(vanilla_smile, forward_price, strike):
        theta = vanilla_smile_theta(vanilla_smile, OptionType.CALL, forward_price, strike, 1.0, 1.0)
        bus_day_theta = theta * (CAL_DAYS_IN_WEEK / BUS_DAYS_IN_WEEK)  # accrue 7 cal days of theta over 5 bus days
        gamma = vanilla_smile_gamma(vanilla_smile, forward_price, strike, 1.0)
        return math.sqrt(-2.0 * bus_day_theta / gamma)

    def approx_breakeven_move(vanilla_smile, forward_price, strike):
        implied_vol = vanilla_smile_vol(vanilla_smile, forward_price, strike)
        atm_vol = vanilla_smile_vol(vanilla_smile, forward_price, forward_price)
        local_vol = 2.0 * implied_vol - atm_vol  # Derman's rule that local vol has twice the skew of implied vol
        return local_vol / math.sqrt(BUS_DAYS_IN_YEAR) * forward_price

    if approx:
        return approx_breakeven_move(vanilla_smile, forward_price, strike)
    else:
        try:
            return exact_breakeven_move(vanilla_smile, forward_price, strike)
        except Exception:
            return approx_breakeven_move(vanilla_smile, forward_price, strike)


def evaluate_unitised_price(call_function, forward_price, strike, option_type=OptionType.CALL):
    unitised_strike = strike / forward_price
    price = call_function(unitised_strike)
    if OptionType.parse(option_type) is OptionType.PUT:
        price = price - (1 - unitised_strike)
    return price


def vanilla_smile_price(call_functions, option_type, forward_price, strike, discount_factor):
    """Calculate option price.  When the vol_adj is passed in we ignore the call function and use Black76"""
    t0_v0_cf = call_functions['t0_v0']
    unitised_price = evaluate_unitised_price(t0_v0_cf, forward_price, strike, option_type)
    price = unitised_price * forward_price * discount_factor
    return price


def vanilla_smile_delta(call_functions, option_type, forward_price, strike, discount_factor):
    """Calculate option delta.  When the vol_adj is passed in we ignore the call function and use Black76"""
    option_type = OptionType.parse(option_type)
    t0_v0_cf = call_functions['t0_v0']
    unitised_strike = strike / forward_price
    unitised_price = t0_v0_cf(unitised_strike)
    unitised_prime_price = t0_v0_cf.prime(unitised_strike)
    delta = discount_factor * (unitised_price - (unitised_prime_price * unitised_strike))

    if option_type is OptionType.PUT:
        delta = delta - discount_factor

    return delta


def vanilla_smile_gamma(call_functions, forward_price, strike, discount_factor):
    t0_v0_cf = call_functions['t0_v0']
    unitised_strike = strike / forward_price
    unitised_prime_prime = t0_v0_cf.prime_prime(unitised_strike)
    gamma = discount_factor * unitised_prime_prime * unitised_strike ** 2 / forward_price
    return gamma


def vanilla_smile_theta(call_functions, option_type, forward_price, strike, discount_factor_t0, discount_factor_t1):
    call_func_t0 = call_functions["t0_v0"]
    price_t0 = evaluate_unitised_price(call_func_t0, forward_price, strike, option_type)

    call_func_t1 = call_functions["t1_v0"]
    price_t1 = evaluate_unitised_price(call_func_t1, forward_price, strike, option_type)

    return forward_price * ((price_t1 * discount_factor_t1) - (price_t0 * discount_factor_t0))


def _vega(discount_factor, forward_price, unitised_price_up, unitised_price_down):
    return discount_factor * forward_price * (unitised_price_up - unitised_price_down) / (2 * VEGA_SHIFT_SIZE)


def vanilla_smile_vega(call_functions, forward_price, strike, discount_factor):
    call_func_v_up = call_functions["t0_vup"]
    unitised_price_up = evaluate_unitised_price(call_func_v_up, forward_price, strike)

    call_func_v_down = call_functions["t0_vdown"]
    unitised_price_down = evaluate_unitised_price(call_func_v_down, forward_price, strike)

    return _vega(discount_factor, forward_price, unitised_price_up, unitised_price_down)


def vanilla_smile_pillar_vega(call_functions, forward_price, strike, pillar, discount_factor):
    call_func_v_up = call_functions[f"t0_vup_{pillar}"]
    unitised_price_up = evaluate_unitised_price(call_func_v_up, forward_price, strike)

    call_func_v_down = call_functions[f"t0_vdown_{pillar}"]
    unitised_price_down = evaluate_unitised_price(call_func_v_down, forward_price, strike)

    return _vega(discount_factor, forward_price, unitised_price_up, unitised_price_down)


def black76_option(
    pricing_date, expiry_date, fwd_price, strike, vol, discount_factor=1.0, option_type=OptionType.CALL,
):
    option_type = OptionType.parse(option_type)
    expiry_time = time_between(pricing_date, expiry_date)
    r = -math.log(discount_factor) / expiry_time
    return Black76Option(s=fwd_price, k=strike, sigma=vol, te=expiry_time, call_put=option_type, r=r)


def black76_price(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.value


def black76_delta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.delta


def black76_gamma(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor)
    return option.gamma


def black76_vega(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor)
    return option.vega


def black76_theta(pricing_date, expiry_date, fwd_price, strike, option_type, vol, discount_factor):
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.theta / 365


def black76_vol_from_premium(
    pricing_date, expiry_date, fwd_price, strike, premium, option_type=OptionType.CALL, discount_factor=1.0,
):
    vol = None
    option = black76_option(pricing_date, expiry_date, fwd_price, strike, vol, discount_factor, option_type)
    return option.get_vol_from_premium(premium, model_name="Secant")


