import math

import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from general_utils import flatten_list_of_lists, time_between, almost_equal, memoize_property
from bisect import bisect_right
from quantity import DAYS_PER_YEAR

from thorn.core.math_utils.function_types import ConstantFunction
from underlying import Underlying, FX
from value_object import Attr, Dict, List, Object


class SpotVolRatios(Object):
    """
    The ratios of spot to M2 volatility by delivery month, for a set of underlyings
    """
    lsu_ratios_dict: Dict(Underlying, List(float)) = Attr({})
    idx = list(range(1, 13))

    @staticmethod
    def from_table(data, underlying_filter=None) -> 'SpotVolRatios':
        """ Builds object from header row of load shaped underlyings, 1st column of month keys, and ratios """
        transposed = list(zip(*data))
        idx = transposed[0][1:]
        if not almost_equal([float(i) for i in idx], SpotVolRatios.idx):
            raise AssertionError("Index should be months from 1 to 12")
        ratios = {Underlying.parse(ul): ratio for ul, *ratio in transposed[1:]}
        svrs = SpotVolRatios(ratios)
        if underlying_filter:
            svrs = svrs.reshape(underlying_filter)
        return svrs

    def to_table(self):
        """Build the spot vol ratios table with underlyings as headers and a row for each month."""
        transposed = [(f"{k}", *v) for k, v in [("", self.idx)] + sorted(self.lsu_ratios_dict.items())]
        return list(zip(*transposed))

    def reshape(self, underlyings):
        """Given a list of underlyings return a new reduced SpotVolRatios object with only those underlyings."""
        filtered_ratios = {ul: r for ul, r in self.lsu_ratios_dict.items() if ul in underlyings}
        return self.clone(lsu_ratios_dict=filtered_ratios)


class DatedSpotVolRatios:
    """
    Calculates the ratio of Spot to M2 volatility for an underlying by time to expiry,
    where the time to expiry is determined relative to the Pricing Date.
    """

    tol = 0.001 / DAYS_PER_YEAR

    def __init__(self, pricing_date, spot_vol_ratios):
        """
        self._taus holds the time to start of monthes
        (current month is duplicated at front and end)
        self._ratios holds the dict from lsu to ratio list for each month
        """
        self.pricing_date = pricing_date
        self.spot_vol_ratios = spot_vol_ratios

        pricing_date_start_of_month = date(self.pricing_date.year, self.pricing_date.month, 1)
        self._taus = []
        self._ratios = {lsu: [] for lsu in self.spot_vol_ratios.lsu_ratios_dict}
        for i in range(13):
            start_of_month = pricing_date_start_of_month + relativedelta(months=i)
            self._taus.append(time_between(self.pricing_date, start_of_month) - self.tol)
            for lsu in self._ratios:
                self._ratios[lsu].append(self.spot_vol_ratios.lsu_ratios_dict[lsu][start_of_month.month - 1])

    def __call__(self, lsu, T):
        tau = T - int(T)
        index = bisect_right(self._taus, tau) - 1
        return self._ratios[lsu][index]

    def reshape(self, underlyings):
        return self.__class__(self.pricing_date, self.spot_vol_ratios.reshape(underlyings))


class VolCorr:
    """
    Calibrated object that computes volatilities and correlations
    """

    attributes = ('mr_processes', 'sigmaT_funcs', 'numeraire_ccy', 'spot_vol_ratios')

    def __init__(self, mr_processes, sigmaT_funcs=None, numeraire_ccy=None, spot_vol_ratios=None):
        self.mr_processes = mr_processes
        self.sigmaT_funcs = sigmaT_funcs
        self.numeraire_ccy = numeraire_ccy
        self.spot_vol_ratios = spot_vol_ratios

        # preprocess sigmas and alphas
        _sigma = self.sigma
        _alpha = self.alpha
        self.lsu_sigma = {lsu: _sigma[self.lsu_to_indices[lsu]] for lsu in self.lsus}
        self.lsu_alpha = {lsu: _alpha[self.lsu_to_indices[lsu]] for lsu in self.lsus}
        self._alpha_ij = _alpha[:, None] + _alpha[None, :]

        # reimplement memoizing here
        self._c_cache = {}
        self._d_correction_cache = {}
        self._numeraire_cache = {}

    def __eq__(self, other):
        return (
            isinstance(other, VolCorr)
            and self.mr_processes == other.mr_processes
            and self.sigmaT_funcs == other.sigmaT_funcs
            and self.numeraire_ccy == other.numeraire_ccy
            and self.spot_vol_ratios == other.spot_vol_ratios
            and self.lsu_sigma == other.lsu_sigma
            and self.lsu_alpha == other.lsu_alpha
        )

    @memoize_property
    def sigma(self):
        return np.array(self.mr_processes.sigma)

    @memoize_property
    def alpha(self):
        return np.array(self.mr_processes.alpha)

    @memoize_property
    def rho(self):
        return np.array(self.mr_processes.rho)

    @property
    def lsus(self):
        return self.mr_processes.lsus

    @property
    def factor_count(self):
        return self.mr_processes.factor_count

    @property
    def lsu_to_indices(self):
        return self.mr_processes.lsu_to_indices

    def get_intersecting_lsus(self, lsu):
        return self.mr_processes.get_intersecting_lsus(lsu)

    def num_factors_for_lsu(self, lsu) -> int:
        return len(self.lsu_to_indices[lsu])

    def clone(self, **kwargs):
        return self.__class__(**{attr: kwargs.get(attr, getattr(self, attr)) for attr in self.attributes})

    def to_numeraire_currency(self, numeraire_ccy):
        if numeraire_ccy is None or self.numeraire_ccy == numeraire_ccy:
            return self

        if numeraire_ccy in self._numeraire_cache:
            return self._numeraire_cache[numeraire_ccy]

        vol_corr = self.clone(
            mr_processes=self.mr_processes.to_numeraire_currency(numeraire_ccy),
            numeraire_ccy=numeraire_ccy,
        )
        self._numeraire_cache[numeraire_ccy] = vol_corr
        return vol_corr

    def set_sigmaT_funcs(self, sigma_t_func):
        # all numeraire-changed vol_corrs share the same sigmaT_funcs
        self.sigmaT_funcs = sigma_t_func
        for vol_corr in self._numeraire_cache.values():
            vol_corr.sigmaT_funcs = sigma_t_func

    def reshape(self, underlyings):
        """Given a list of underlyings return a new reduced VolCorr object with only those underlyings."""
        mr_processes = self.mr_processes.reshape(underlyings)
        spot_vol_ratios = None if self.spot_vol_ratios is None else self.spot_vol_ratios.reshape(underlyings)
        sigmaT_funcs = None if self.sigmaT_funcs is None else self.sigmaT_funcs.reshape(underlyings)
        return self.clone(mr_processes=mr_processes, sigmaT_funcs=sigmaT_funcs, spot_vol_ratios=spot_vol_ratios)

    def lsu_mapping(self, lsus):
        simulated_indices = flatten_list_of_lists([
            self.lsu_to_indices[lsu] for lsu in lsus
        ])
        simulated_indices = list(set(simulated_indices))
        lsu_to_simulated_indices = {
            lsu: [simulated_indices.index(i)
                  for i in self.lsu_to_indices[lsu]]
            for lsu in lsus
        }
        return simulated_indices, lsu_to_simulated_indices

    def C(self, t):
        """ xi covariance matrix"""
        try:
            return self._c_cache[t]
        except KeyError:
            filt = self._alpha_ij > 1e-10
            ret = self.rho * t
            if filt.any():
                if t == np.inf:
                    ret[filt] = (self.rho / self._alpha_ij)[filt]
                elif t > 1e-10:
                    alpha_ij_t = t * self._alpha_ij
                    ret[filt] *= (1 - np.exp(-alpha_ij_t[filt])) / alpha_ij_t[filt]

            self._c_cache[t] = ret
            return ret

    def _d_correction(self, lsu, D):
        """ Swap correction when D > 0"""
        try:
            return self._d_correction_cache[lsu, D]
        except KeyError:
            alpha_D = self.lsu_alpha[lsu] * D
            d_correction = np.ones_like(alpha_D)
            filt = alpha_D > 1e-10
            if filt.any():
                d_correction[filt] = (1 - np.exp(-alpha_D[filt])) / alpha_D[filt]
            self._d_correction_cache[lsu, D] = d_correction
            return d_correction

    def _M_B_no_sigmaT(self, lsu, t, T):
        if t == np.inf:
            mb = self.lsu_sigma[lsu]
        else:
            mb = self.lsu_sigma[lsu] * np.exp(-self.lsu_alpha[lsu] * (T - t))
        if self.spot_vol_ratios is not None:
            try:
                spot_vol_ratio = self.spot_vol_ratios(lsu, T)
                mb[2] *= spot_vol_ratio
            except (KeyError, IndexError):
                pass
        return mb

    def M_B(self, lsu, t, T, D=0):
        sigmaT = ConstantFunction(1) if self.sigmaT_funcs is None else self.sigmaT_funcs.get(lsu, ConstantFunction(1))

        if D == 1. / DAYS_PER_YEAR:
            return self._M_B_no_sigmaT(lsu, t, T) * sigmaT(T) * self._d_correction(lsu, D)

        if D == 0.:
            return self._M_B_no_sigmaT(lsu, t, T) * sigmaT(T)

        if t == np.inf:
            return self._M_B_no_sigmaT(lsu, t, T) * sigmaT(T) * self._d_correction(lsu, D)

        return sum(self._M_B_no_sigmaT(lsu, t, domain_start)
                   * self._d_correction(lsu, domain_end - domain_start)
                   * sigmaT(domain_start) * (domain_end - domain_start)
                   for domain_start, domain_end in sigmaT.get_sub_domains(T, T + D)) / D

    def _indices(self, lsus):
        indices = []
        for lsu in lsus:
            indices.extend(self.lsu_to_indices[lsu])
        return indices

    def _MBs(self, lsus, t, T, D):
        return np.concatenate([self.M_B(lsu, t, T, D) for lsu in lsus])

    def _build_N_B_indices(self, D1, D2, T1, T2, lsus1, lsus2, t):
        T2 = T1 if T2 is None else T2
        D2 = D1 if D2 is None else D2
        N_B = self._MBs(lsus1, t, T1, D1)[:, None] * self._MBs(lsus2, t, T2, D2)[None, :]
        return N_B, self._indices(lsus1), self._indices(lsus2)

    def N_B_and_indices(self, D1, D2, T1, T2, include_fx, lsu1, lsu2, t):
        if isinstance(lsu1, set) or isinstance(lsu2, set):
            assert isinstance(lsu1, set), f"Expected consistent sets of lsus not {lsu1} and {lsu2}"
            assert isinstance(lsu2, set), f"Expected consistent sets of lsus not {lsu1} and {lsu2}"
            assert not include_fx, f"Cannot include FX with sets of lsus ({lsu1}, {lsu2})"
            lsus1 = lsu1
            lsus2 = lsu2
        else:
            lsus1 = [lsu1]
            lsus2 = [lsu2]
            if include_fx:
                if lsu1.currency != self.numeraire_ccy:
                    fx_lsu = FX.get(self.numeraire_ccy, lsu1.currency)
                    lsus1.append(fx_lsu)
                if lsu2.currency != self.numeraire_ccy:
                    fx_lsu = FX.get(self.numeraire_ccy, lsu2.currency)
                    lsus2.append(fx_lsu)
        N_B, indices1, indices2 = self._build_N_B_indices(D1, D2, T1, T2, lsus1, lsus2, t)
        return N_B, indices1, indices2

    def calc_covariance(self, lsu1, lsu2, t, T1, D1, T2=None, D2=None, include_fx=False):
        N_B, indices1, indices2 = self.N_B_and_indices(D1, D2, T1, T2, include_fx, lsu1, lsu2, t)
        return np.einsum('ij, ij', N_B, self.C(t)[indices1, :][:, indices2])

    def calc_inst_covariance(self, lsu1, lsu2, T1, D1, T2=None, D2=None, include_fx=False):
        t = 0
        N_B, indices1, indices2 = self.N_B_and_indices(D1, D2, T1, T2, include_fx, lsu1, lsu2, t)
        return np.einsum('ij, ij', N_B, self.rho[indices1, :][:, indices2])

    def calc_var(self, lsu, t, T, D, include_fx=False):
        if t > 0:
            return self.calc_covariance(lsu, lsu, t, T, D, include_fx=include_fx)
        else:
            return self.calc_inst_covariance(lsu, lsu, T, D, include_fx=include_fx)

    def calc_vol(self, lsu, t, T, D, include_fx=False):
        var = self.calc_var(lsu, t, T, D, include_fx)
        if t > 0:
            var /= t
        return np.sqrt(var)

    def calc_inst_vol(self, lsu, T, D, include_fx=False):
        ''' Calculate the instantaneous volatility of a forward
        contract for delivery between T and T+D, associated with the
        price process pp'''
        variance = self.calc_inst_covariance(lsu, lsu, T, D, include_fx=include_fx)
        return np.sqrt(variance)

    def calc_P(self, lsu, tau, D):
        vol = self.calc_inst_vol(lsu, tau, D)
        MB = self.M_B(lsu, 0, tau, D)
        return MB / vol

    def calc_inst_corr(self, lsu1, lsu2, T1, D1, T2=None, D2=None, include_fx=False):
        """ Calculate the instantaneous correlation between two forward contracts.
        The first contract is for delivery between T1 and T1+D1 and is
        associated with the price process pp1. The second is for delivery
        between T2 and T2+D2 and is
        associated with the price process pp2.If T2 is omitted it is assumed
        to be T1, similarly D2."""
        T2 = T1 if T2 is None else T2
        D2 = D1 if D2 is None else D2
        variance1 = self.calc_inst_covariance(lsu1, lsu1, T1, D1, include_fx=include_fx)
        variance2 = self.calc_inst_covariance(lsu2, lsu2, T2, D2, include_fx=include_fx)
        covariance = self.calc_inst_covariance(lsu1, lsu2, T1, D1, T2, D2, include_fx=include_fx)
        return covariance / math.sqrt(variance1 * variance2)

    def calc_corr(self, lsu1, lsu2, t, T1, D1, T2=None, D2=None, include_fx=False):
        """ Calculate the correlation between two forward contracts until t.
        The first contract is for delivery between T1 and T1+D1 and is
        associated with the price process pp1. The second is for delivery
        between T2 and T2+D2 and is
        associated with the price process pp2.If T2 is omitted it is assumed
        to be T1, similarly D2."""
        T2 = T1 if T2 is None else T2
        D2 = D1 if D2 is None else D2
        if t > 0:
            variance1 = self.calc_covariance(lsu1, lsu1, t, T1, D1, include_fx=include_fx)
            variance2 = self.calc_covariance(lsu2, lsu2, t, T2, D2, include_fx=include_fx)
            covariance = self.calc_covariance(lsu1, lsu2, t, T1, D1, T2, D2, include_fx=include_fx)
        else:
            variance1 = self.calc_inst_covariance(lsu1, lsu1, T1, D1, include_fx=include_fx)
            variance2 = self.calc_inst_covariance(lsu2, lsu2, T2, D2, include_fx=include_fx)
            covariance = self.calc_inst_covariance(lsu1, lsu2, T1, D1, T2, D2, include_fx=include_fx)
        return covariance / math.sqrt(variance1 * variance2)

    def calc_ratio_vol(self, numerator_lsus, denominator_lsus, t, T1, D1, T2=None, D2=None):
        T2 = T1 if T2 is None else T2
        D2 = D1 if D2 is None else D2
        numerator_vol = self.calc_vol(numerator_lsus, t, T1, D1)
        denominator_vol = self.calc_vol(denominator_lsus, t, T2, D2)

        corr = self.calc_corr(numerator_lsus, denominator_lsus, t, T1, D1, T2, D2)
        # Handle precision errors when using proxy vols
        return math.sqrt(
            max(0, numerator_vol ** 2 + denominator_vol ** 2 - 2 * numerator_vol * denominator_vol * corr)
        )


