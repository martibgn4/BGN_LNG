from abc import abstractmethod
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np

from general_utils import memoize_method, memoize_property, time_between, isiterable
from quantity import units, value, Quantity, DAY, SECONDS_PER_DAY

from thorn.core.market.volatility.lvc.phi import fit_phi2
from thorn.core.market.volatility.lvc.ssvi_vol_surface import SSVIBasketVolSurface
from thorn.core.market.volatility.vc.vol_corr_adaptor import DatedVolCorrAdaptor
from thorn.core.pricers.analytical.kirk import KirkParameters, KirkSpreadPricer, POS, NEG, GaussianCopulaSpreadPricer
from thorn.core.pricers.analytical.normal_option import BachelierOption
from thorn.core.pricers.pricing_formula.discount_curve_adapter import DiscountCurveAdapter
from time_period import get_duration, intersection
from underlying import SwapRate


class BasketVol:
    def __init__(self, swap_rates, weights, ex_date, strike, currency, mkt_dynamic,
                 override_price_valuation_date=None, obs_dates=None):
        self.pricing_date = override_price_valuation_date or mkt_dynamic.pricing_date
        self.mkt_dynamic = mkt_dynamic
        self.strike = strike

        try:
            self.ex_date = mkt_dynamic.mkt_calendar.trading_date_on_or_before(ex_date)
            self.te = time_between(self.pricing_date, ex_date)
        except Exception as e:
            raise ValueError(f"ex_date should be a date not {ex_date}") from e

        self.vol_corr = mkt_dynamic.vol_corr.to_numeraire_currency(currency)
        self.vol_corr_adaptor = DatedVolCorrAdaptor(self.vol_corr, self.pricing_date)
        self.swap_rates, self.weights, self.obs_dates = \
            self._decompose_swap_rates_and_weights(swap_rates, weights, obs_dates)

        # pre-compute forwards
        self._forwards = [
            self.mkt_dynamic.get_price(sr.underlying, sr.time_period, override_price_valuation_date)
            for sr in self.swap_rates
        ]
        weighted_forwards = [w * f for w, f in zip(self.weights, self._forwards)]
        if self.strike is not None:
            self.unit = units(self.strike)
        elif len(weighted_forwards) > 0:
            self.unit = units(weighted_forwards[0])
        self.weighted_forwards = np.array([value(wf.convert_into(self.unit)) for wf in weighted_forwards])
        self._atm_covs = None

    @property
    def atm_covs(self):
        # critical function. optimized here for performance
        if self._atm_covs is None:
            # initialize looping lists
            # discard non_stochastic underlyings
            indices = []
            swap_rates = []
            ex_dates = []
            lsus = []
            for i, (sr, obs_date) in enumerate(zip(self.swap_rates, self.obs_dates)):
                if sr.underlying.stochastic:
                    indices.append(i)
                    swap_rates.append(sr)
                    ex_dates.append(min(self.ex_date, obs_date))
                    lsus.append(self.vol_corr_adaptor.get_lsu(sr))

            self._atm_covs = np.zeros((len(self.swap_rates), len(self.swap_rates)))
            for i1, lsu1, sr1, ex_date1 in zip(indices, lsus, swap_rates, ex_dates):
                if ex_date1 > self.pricing_date:
                    for i2, lsu2, sr2, ex_date2 in zip(indices, lsus, swap_rates, ex_dates):
                        if i2 == i1:
                            t1, T1, D1 = self.vol_corr_adaptor.calc_t_T_D(sr1, ex_date1)
                            self._atm_covs[i1][i1] = \
                                self.vol_corr.calc_covariance(lsu1, lsu1, t1, T1, D1, T1, D1, include_fx=True)
                        if i2 > i1 and ex_date2 > self.pricing_date:
                            # covariance matrix is symmetric
                            ex_date = min(ex_date1, ex_date2)
                            t1, T1, D1 = self.vol_corr_adaptor.calc_t_T_D(sr1, ex_date)
                            t2, T2, D2 = self.vol_corr_adaptor.calc_t_T_D(sr2, ex_date)
                            t = min(t1, t2)
                            cov12 = self.vol_corr.calc_covariance(lsu1, lsu2, t, T1, D1, T2, D2, include_fx=True)
                            self._atm_covs[i1][i2] = cov12
                            self._atm_covs[i2][i1] = cov12

        return self._atm_covs

    @property
    @abstractmethod
    def covs(self):
        """ covariance matrix"""

    def _decompose_swap_rates_and_weights(self, swap_rates, weights, obs_dates):
        if obs_dates is None:
            obs_dates = [self.ex_date for _ in swap_rates]
        decomposed_swap_rates = []
        decomposed_weights = []
        decomposed_obs_dates = []
        for swap_rate, weight, obs_date in zip(swap_rates, weights, obs_dates):
            eff_obs_date = self.mkt_dynamic.mkt_calendar.trading_date_on_or_before(obs_date)
            swap_rates, factors = self._decompose_swap_rate(swap_rate)
            decomposed_swap_rates.extend(swap_rates)
            decomposed_weights.extend([f * weight for f in factors])
            decomposed_obs_dates.extend([eff_obs_date for _ in factors])
        return decomposed_swap_rates, decomposed_weights, decomposed_obs_dates

    def _decompose_swap_rate(self, swap_rate):
        underlying = swap_rate.underlying
        if not underlying.stochastic:
            return [swap_rate], [1]
        delivery_period = swap_rate.time_period
        discount_curve = DiscountCurveAdapter(self.mkt_dynamic, underlying.currency)
        ls_dps = [
            intersection(delivery_period, lsu.load_shape)
            for lsu in self.vol_corr.get_intersecting_lsus(underlying)
            if get_duration(intersection(delivery_period, lsu.load_shape)) > 0
        ]
        total_dp_time = get_duration(delivery_period)
        total_ls_dps_time = sum(get_duration(dp) for dp in ls_dps)
        if abs(total_dp_time - total_ls_dps_time) > 1 * DAY / SECONDS_PER_DAY:
            raise AssertionError(f"Can't compute vol for stochastic underlying {underlying}, lsus missing in Volcorr")
        factors = [underlying.settlement_rule.get_discounted_duration(dp, discount_curve) for dp in ls_dps]
        rescaled_factors = [f / sum(factors) for f in factors]
        swap_rates = [SwapRate(underlying, intersection(dp, delivery_period)) for dp in ls_dps]
        return swap_rates, rescaled_factors

    def indices(self, sign):
        return [i for i, w in enumerate(self.weights) if (w > 0 if sign == POS else w < 0)]

    @property
    def is_spread(self):
        return bool(self.indices(NEG))

    def sum_weighted_forwards(self, sign):
        return sum(self.weighted_forwards[self.indices(sign)])

    def vol(self, sign):
        return np.sqrt(self.get_covariance(sign, sign))

    @property
    def _kirk_arguments(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            covariance = self.get_covariance(POS, NEG)
            correlation = covariance / (self.vol(POS) * self.vol(NEG))
        return (
            Quantity(self.sum_weighted_forwards(POS), self.unit),
            Quantity(self.sum_weighted_forwards(NEG), self.unit),
            self.strike,
            self.vol(POS),
            self.vol(NEG),
            correlation,
            self.te,
        )

    def kirk_parameters(self, option_type):
        return KirkParameters(*self._kirk_arguments, option_type=option_type)

    @property
    def kirk(self):
        return KirkSpreadPricer(*self._kirk_arguments)

    def calc_vol(self):
        if self.is_spread:
            return self.kirk.spread_vol
        return self.vol(POS)

    def price_option(self, option_type):
        return self.kirk.option_value(option_type)

    def _get_covariance(self, sign1, sign2, covs):
        r"""
        Gives the covariance of \f$(\sum_{i \in I}^n w_i F_{T,T_1,T_2}^i)\f$ and
        \f$(\sum_{j \in J}^n w_j F_{T,T_1,T_2}^j)\f$
        """
        indices1 = self.indices(sign1)
        indices2 = self.indices(sign2)

        if len(indices1) * len(indices2) == 0 or self.te <= 0.:
            return 0.

        if len(self.weighted_forwards.shape) < 2:
            wfs1 = self.weighted_forwards[indices1, None]
            wfs2 = self.weighted_forwards[None, indices2]
            return float(np.log(np.einsum('ij, ij', wfs1 * wfs2, np.exp(covs[indices1, :][:, indices2])) /
                                (np.sum(wfs1) * np.sum(wfs2))) / self.te)
        else:
            # vectorized basket pricing
            wfs1 = self.weighted_forwards[indices1]
            wfs2 = self.weighted_forwards[indices2]
            res = 0.
            for i1, wf1 in zip(indices1, wfs1):
                for i2, wf2 in zip(indices2, wfs2):
                    res += wf1 * wf2 * np.exp(covs[i1][i2])
            return np.log(res / (np.sum(wfs1, axis=0) * np.sum(wfs2, axis=0))) / self.te

    @memoize_method
    def get_covariance(self, sign1, sign2):
        return self._get_covariance(sign1, sign2, self.covs)

    def get_atm_covariance(self, sign1, sign2):
        return self._get_covariance(sign1, sign2, self.atm_covs)


class VcBasketVol(BasketVol):

    @property
    def covs(self):
        return self.atm_covs


class NvcBasketVol(VcBasketVol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rescaled_weights = np.array([w / value(f) for f, w in zip(self._forwards, self.weighted_forwards)])

    def _get_covariance(self, sign1, sign2, covs):
        r"""
        Gives the covariance of \f$(\sum_{i \in I}^n w_i F_{T,T_1,T_2}^i)\f$ and
        \f$(\sum_{j \in J}^n w_j F_{T,T_1,T_2}^j)\f$
        """
        indices1 = self.indices(sign1)
        indices2 = self.indices(sign2)

        if len(indices1) * len(indices2) == 0 or self.te <= 0.:
            return 0.

        if len(self.weighted_forwards.shape) < 2:
            ws1 = self.rescaled_weights[indices1, None]
            ws2 = self.rescaled_weights[None, indices2]
            return float(np.einsum('ij, ij', ws1 * ws2, covs[indices1, :][:, indices2]) / self.te)
        else:
            # vectorized basket pricing
            ws1 = self.rescaled_weights[indices1]
            ws2 = self.rescaled_weights[indices2]
            res = 0.
            for i1, w1 in zip(indices1, ws1):
                for i2, w2 in zip(indices2, ws2):
                    res += w1 * w2 * covs[i1][i2]
            return res / self.te

    @property
    def kirk(self):
        raise NotImplementedError()

    def calc_vol(self):
        if self.is_spread:
            vol = np.sqrt(
                self.get_covariance(POS, POS)
                + 2 * self.get_covariance(POS, NEG)
                + self.get_covariance(NEG, NEG)
            )
        else:
            vol = self.vol(POS)
        return vol

    def price_option(self, option_type):
        return BachelierOption(
            s=Quantity(self.sum_weighted_forwards(POS), self.unit),
            k=-Quantity(self.sum_weighted_forwards(NEG), self.unit) + self.strike,
            sigma=Quantity(self.calc_vol(),  self.unit),
            te=self.te,
            call_put=option_type,
            tm=None,
            r=0
        ).value


class LvcBasketVol(BasketVol):
    STRIKE_TOL = 1e-6

    def __init__(self, swap_rates, weights, ex_date, strike, currency, mkt_dynamic,
                 override_price_valuation_date=None, obs_dates=None):
        super().__init__(swap_rates, weights, ex_date, strike, currency, mkt_dynamic,
                         override_price_valuation_date, obs_dates)
        if isiterable(value(self.strike)):
            raise NotImplementedError("Pricing of multiple strikes is not supported in LVC")

        self.sz = len(self.weights)
        self.atm_vols = [np.sqrt(self.atm_covs[i][i]) for i in range(self.sz)]
        self.optimal_strikes = None if self.is_spread else self._get_optimal_strikes(self.strike, POS)
        self.vols = None if self.is_spread else self._get_vols(self.optimal_strikes, self.strike)

    def _covs(self, vols):
        return np.array([
            [
                float(self.atm_covs[i][j] / self.atm_vols[j] * vols[j] / self.atm_vols[i] * vols[i])
                if abs(self.atm_covs[i][j]) > 0 else 0
                for j in range(self.sz)
            ] for i in range(self.sz)
        ])

    @memoize_property
    def covs(self):
        return None if self.is_spread else self._covs(self.vols)

    def _get_vols(self, optimal_strikes, strike):
        def _get_vol(swap_rate, optimal_strike, obs_date):
            new_ex_date = min(obs_date, self.ex_date)
            if not swap_rate.underlying.stochastic \
                    or value(strike) < self.STRIKE_TOL \
                    or new_ex_date <= self.pricing_date:
                return 0

            vol = self.mkt_dynamic.get_volatility(
                swap_rate.load_shaped_underlying, swap_rate.base_period, new_ex_date, optimal_strike)
            return vol * np.sqrt(time_between(self.pricing_date, new_ex_date))

        return [
            _get_vol(swap_rate, optimal_strike, obs_date)
            for swap_rate, optimal_strike, obs_date in zip(self.swap_rates, optimal_strikes, self.obs_dates)
        ]

    def _get_optimal_strikes(self, strike, sign):
        if self.sz == 1:
            return [strike / self.weights[0]]

        fwd = self.sum_weighted_forwards(sign)
        if strike.value * (-1 if sign == NEG else 1) < self.STRIKE_TOL:
            return [
                strike / fwd * weighted_forward / weight
                for weighted_forward, weight in zip(self.weighted_forwards, self.weights)
            ]

        sqrt_atm_var = sqrt(self.get_atm_covariance(sign, sign) * self.te)
        d_star = log(value(strike / fwd)) / sqrt_atm_var
        return [
            self.weighted_forwards[i] / weight * np.exp(self.atm_vols[i] * d_star) * self.unit
            for i, weight in enumerate(self.weights)
        ]

    @property
    def _kirk_arguments(self):
        if self.is_spread:
            raise NotImplementedError("Kirk not supported with LVC")
        return super()._kirk_arguments

    def _surface(self, sign):
        atm_vol = sqrt(self.get_atm_covariance(sign, sign))
        phi2 = self._calc_phi2(sign, atm_vol)
        return SSVIBasketVolSurface(lambda t: atm_vol, phi2)

    def _calc_phi2(self, sign, atm_vol):
        fwd = Quantity(self.sum_weighted_forwards(sign), self.unit)
        sqrt_atm_var = atm_vol * sqrt(self.te)
        pillars = [0.1, 0.5, 0.9]
        d_stars = []
        vol_ratios = []
        for pillar in pillars:
            d_star = norm.ppf(pillar)
            strike = fwd * exp(-d_star * sqrt_atm_var)
            optimal_strikes = self._get_optimal_strikes(strike, sign)
            vols = self._get_vols(optimal_strikes, strike * (-1 if sign == NEG else 1))
            covs = self._covs(vols)
            d_stars.append(d_star)
            vol_ratios.append(sqrt(self._get_covariance(sign, sign, covs)) / atm_vol)
        return fit_phi2(d_stars, vol_ratios, sqrt_atm_var)

    @property
    def _correlation(self):
        pos_atm_var = self.get_atm_covariance(POS, POS)
        neg_atm_var = self.get_atm_covariance(NEG, NEG)
        if pos_atm_var * neg_atm_var == 0:
            return 0.0
        return self.get_atm_covariance(POS, NEG) / sqrt(pos_atm_var * neg_atm_var)

    @property
    def gc(self):
        return GaussianCopulaSpreadPricer(
            pos_fwd=self.sum_weighted_forwards(POS) * self.unit,
            neg_fwd=self.sum_weighted_forwards(NEG) * self.unit,
            strike=self.strike,
            pos_surface=self._surface(POS),
            neg_surface=self._surface(NEG),
            corr=self._correlation,
            te=self.te
        )

    def price_option(self, option_type):
        if self.is_spread:
            return self.gc.option_value(option_type)
        return super().price_option(option_type)


