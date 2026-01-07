import numpy as np

from BGN_LNG.Utils.utils_maths import truncated_normal_sample, jump_rv


class BaseUnderlyingSimulator:
    """Contains basic structure of an underlying
    to be simulated via bounded normal plus a stochastic jump"""
    def __init__(self, normal_lb, normal_ub, normal_avg, normal_sigma,
                 jump_probability, jump_avg, jump_sigma,
                 current_price,n=1000000):
        self.normal_lb = normal_lb
        self.normal_ub = normal_ub
        self.normal_avg = normal_avg
        self.normal_sigma = normal_sigma

        self.p = jump_probability
        self.jump_avg = jump_avg
        self.jump_sigma = jump_sigma

        self.current_price = current_price  # To be used for Intrinsic calcs

        self.n = n
        self._cache_normal = {}
        self._cache_jump = {}

    def _get_bounded_normal_sim(self, random_state=1):
        if random_state in self._cache_normal:
            return self._cache_normal[random_state]
        else:
            result = truncated_normal_sample(
                self.n,
                self.normal_lb,
                self.normal_ub,
                self.normal_avg,
                sigma=self.normal_sigma
                )
            self._cache_normal[random_state] = result
            return result


    def _get_stochastic_jump(self, random_state=1):
        if random_state in self._cache_jump:
            return self._cache_jump[random_state]
        else:
            result = jump_rv(
                self.n,
                self.p,
                self.jump_avg,
                self.jump_sigma
            )
            self._cache_jump[random_state] = result
            return result

    def simulate(self, random_state=1):
        normal_sim = self._get_bounded_normal_sim(random_state=random_state)
        stochastic_jump = self._get_stochastic_jump(random_state=random_state)
        return normal_sim + stochastic_jump

    @property
    def intrinsic_value(self):
        return self.current_price

class NWEUnderlyingSimulator(BaseUnderlyingSimulator):
    """"""

class FreightUnderlyingSimulator(BaseUnderlyingSimulator):
    """"""


class OptionValueResult:
    def __init__(self, option_mc, option_intrinsic, forward_mc, forward_intrinsic):
        self.option_value = option_mc
        self.intrinsic_value = option_intrinsic
        self.extrinsic_value = option_mc - option_intrinsic

        self.forward_mc = forward_mc
        self.forward_intrinsic = forward_intrinsic

    def __repr__(self):
        return (f"OptionValue: {self.option_value:3f}, ForwardValue: {self.forward_mc:3f}, "
                f"OptionIntrinsic: {self.intrinsic_value:3f}, ForwardIntrinsic: {self.forward_intrinsic:3f}")




class LNGOptionPricer:
    def __init__(self, nwe_simulator, freight_simulator,
                 bought_mmbtu, boil_off_rate, route_days,
                 selling_net_spread):
        self.nwe_simulator = nwe_simulator
        self.freight_simulator = freight_simulator
        self.bought_mmbtu = bought_mmbtu
        self.sold_mmbtu = bought_mmbtu - route_days*boil_off_rate
        self.route_days = route_days
        self.k = selling_net_spread

        self._cache = {}

    def forward_payoff(self, buying_mc_prices, freight_mc_prices):
        freight_usd = self.route_days * freight_mc_prices
        payoff = (-self.bought_mmbtu * buying_mc_prices + self.sold_mmbtu * self.k - freight_usd)/self.bought_mmbtu
        return payoff

    def price_option(self, random_state=1):
        buying_mc_prices = self.nwe_simulator.simulate(random_state=random_state)
        freight_mc_prices = self.freight_simulator.simulate(random_state=random_state)
        forward_mc = self.forward_payoff(buying_mc_prices, freight_mc_prices)
        option_mc = np.maximum(forward_mc, 0)

        buying_price_intrinsic = self.nwe_simulator.intrinsic_value
        freight_price_intrinsic = self.freight_simulator.intrinsic_value
        forward_intrinsic = self.forward_payoff(buying_price_intrinsic, freight_price_intrinsic)
        option_intrinsic = np.maximum(forward_intrinsic, 0)

        return OptionValueResult(
            option_mc.mean(),
            option_intrinsic,
            forward_mc.mean(),
            forward_intrinsic
        )

