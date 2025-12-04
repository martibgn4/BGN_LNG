import typing
from datetime import date

from quantity import Quantity
from thorn.core.priceable.base.option_terms import CallOrPutOption
from underlying import SwapRate
from value_object import Object, Attr


class BasketComponent(Object):
    swap_rate: SwapRate = Attr()
    weight: Quantity = Attr()
    fx_forward_date: typing.Optional[date] = Attr()
    obs_date: typing.Optional[date] = Attr()
    fixings_provider: typing.Optional[str] = Attr()
    fx_fixings_provider: typing.Optional[str] = Attr()

    def post_init(self):
        if self.obs_date is not None and self.fx_forward_date is not None:
            assert self.obs_date <= self.fx_forward_date, \
                f"Obs date ({self.obs_date}) cannot be after FX date ({self.fx_forward_date})."


class BasketOptionParams(CallOrPutOption):

    def __init__(self, swap_rates, weights, strike, option_type, exercise_date, currency=None, settlement_date=None,
                 obs_dates=None):
        self.swap_rates = swap_rates
        self.weights = weights
        self.strike = strike
        self.option_type = option_type
        self.exercise_date = exercise_date
        self.currency = currency if currency is not None else swap_rates[0].underlying.currency
        self.settlement_date = settlement_date
        self.obs_dates = obs_dates

    def post_init(self):
        if self.settlement_date is not None:
            assert self.exercise_date <= self.settlement_date, "%s !<= %s" % (self.exercise_date, self.settlement_date)


