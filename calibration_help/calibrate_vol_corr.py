from collections import OrderedDict
from math import sqrt

import numpy as np
import scipy.optimize as opt
from scipy.special import expm1, expit

from general_utils import norm_inv_cdf, almost_equal
from quantity import DAYS_PER_YEAR
from thorn.core.calibration.base.contract_keys import Contract
from thorn.core.calibration.base.vol_corr_calib_info import VcCalibParams, CalibInfoDict
from thorn.core.calibration.base.vol_corr_time_periods import time_to_delivery, delivery_duration
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.vol_corr import VolCorr
from underlying import FX

sigmas_x0_bnds = [
    # (initial value, bounds)
    (0.1, (0.01, None)),  # sigma long
    (0.15, (0.01, None)),  # sigma short
    (0.4, (0.01, None)),  # sigma very short
]
alphas_robust_x0_bnds = [
    # (initial value, bounds)
    (0.25, (1e-05, 0.5)),  # alpha long
    (2.0, (0.5, 10)),  # alpha short
    (30.0, (10, 100)),  # alpha very short
]
alphas_x0_bnds = [
    # (initial value, bounds)
    (0.05, (0.0, 0.1)),  # alpha long
    (2.0, (0.1, 10)),  # alpha short
    (30.0, (10, 100)),  # alpha very short
]


def initial_parameter_set(factor_count, with_lt_alpha, _sigmas_x0_bnds, _alphas_x0_bnds,
                          fixed_alpha_len=0, fixed_sigma_len=0, with_probability_condition=False):
    """defines the start points and boundaries for the sigma, alpha optimisations for one underlying"""
    sigmas = _sigmas_x0_bnds.copy()
    last_sigma = factor_count
    if with_probability_condition:
        # no long term sigma calibrated, it has a deterministic relation to the long term alpha
        sigmas.pop(0)
        last_sigma -= 1
        # short term sigma is replaced with a parameter that defines its relationship with the alphas
        sigmas[0] = (0.0, (None, None))
    sigmas = sigmas[fixed_sigma_len:last_sigma]

    alphas = _alphas_x0_bnds.copy()
    first_alpha = 0 if with_probability_condition or with_lt_alpha else 1
    alphas = alphas[max(first_alpha, fixed_alpha_len):factor_count]

    x0 = [_x0 for _x0, _bnd in sigmas + alphas]
    bounds = [_bnd for _x0, _bnd in sigmas + alphas]
    return x0, bounds


def time_to_delivery_and_duration(contract: Contract, offset=0):
    return (
        (offset + time_to_delivery[contract.tenor]) / DAYS_PER_YEAR,
        delivery_duration[contract.tenor] / DAYS_PER_YEAR
    )


def calc_modelled_vols(lsu, vol_corr, contracts, calib_info=None, returns_window=0):
    """ Calculate the vol for the specified relative contracts
    using the supplied calculator and price process
    """
    modelled_vols = np.empty(len(contracts))
    number_of_days_before_rollover = 0 if calib_info is None else calib_info.number_of_days_before_rollover
    t = returns_window / DAYS_PER_YEAR
    for i, contract in enumerate(contracts):
        T, D = time_to_delivery_and_duration(contract, number_of_days_before_rollover)
        modelled_vols[i] = vol_corr.calc_vol(lsu, t, t + T, D)
    return modelled_vols


def calc_ratio_contract_vol(lsu, vol_corr, numerator: Contract, denominator: Contract, returns_window=0, offset=0):
    t = returns_window / DAYS_PER_YEAR
    tau1, d1 = time_to_delivery_and_duration(numerator, offset)
    tau2, d2 = time_to_delivery_and_duration(denominator, offset)
    return vol_corr.calc_ratio_vol(lsu, lsu, t, t + tau1, d1, t + tau2, d2)


def calc_modelled_ratio_vols(lsu, vol_corr, ratio_contracts, calib_info=None, returns_window=0):
    modelled_vols = np.empty(len(ratio_contracts))
    number_of_days_before_rollover = 0 if calib_info is None else calib_info.number_of_days_before_rollover
    for i, spread_contract in enumerate(ratio_contracts):
        modelled_vols[i] = calc_ratio_contract_vol(
            lsu, vol_corr, spread_contract.numerator, spread_contract.denominator, returns_window,
            offset=number_of_days_before_rollover
        )
    return modelled_vols


def calc_modelled_vol_from_probability_condition(probability_cond, percentage_cond):
    """
    Given a probability condition of the form
    P(F(t,T,D) <= percentage_value*F(0,T,D)) = probability_value, for a very large t, T=t,
    the volatility of F(t,T=t,D) as t -> inf, is the solution to a quadratic equation of the form
    0.5*ss^2 - N^{-1}(probability)*ss + log(percentage) = 0, where ss denotes this volatility.
    """
    a = 0.5
    b = -norm_inv_cdf(probability_cond)
    c = np.log(percentage_cond)
    d = b*b - 4*a*c
    ss = -b + np.sqrt(d)
    return ss


def obtain_sigmas_from_1F_alphas(probability_cond, percentage_cond, alpha_l):
    ss = calc_modelled_vol_from_probability_condition(probability_cond, percentage_cond)
    l_alpha_scale = -expm1(-alpha_l) / alpha_l
    sigma_l = ss * sqrt(2 * alpha_l) / l_alpha_scale
    return sigma_l


def obtain_sigmas_from_2F_alphas(
        probability_cond, percentage_cond, alpha_l, alpha_s, alpha_to_sigma_mapping_parameter, rho_s_l
):
    ss = calc_modelled_vol_from_probability_condition(probability_cond, percentage_cond)
    alpha_s_plus_alpha_l = alpha_s + alpha_l
    alpha_s_mul_alpha_l = alpha_s * alpha_l
    rho_sq = rho_s_l ** 2
    one_m_exp_m_alpha_l = -expm1(-alpha_l)
    one_m_exp_m_alpha_s = -expm1(-alpha_s)

    if almost_equal(rho_s_l, 0.0):
        theta_critical = 0.5 * np.pi  # if rho is zero, acot(0) is pi/2
    else:
        # numpy does not have arccot, but arccot(x) = arctan(1 / x)
        numerator = np.sqrt(alpha_s_plus_alpha_l ** 2 - 4.0 * rho_sq * alpha_s_mul_alpha_l)
        denominator = 2.0 * rho_s_l * np.sqrt(alpha_s_mul_alpha_l)
        theta_critical = np.arctan(numerator / denominator)
    theta_max = theta_critical if rho_s_l >= 0 else theta_critical + np.pi
    theta = expit(alpha_to_sigma_mapping_parameter) * theta_max

    xl = ss * np.sin(theta) / (one_m_exp_m_alpha_l * np.sqrt(
        0.5 / alpha_l * (1.0 - rho_sq * 4.0 * alpha_s_mul_alpha_l / alpha_s_plus_alpha_l ** 2)
    ))
    xs = np.sqrt(2.0 * alpha_s) / one_m_exp_m_alpha_s * (
            ss * np.cos(theta) - ss * 2 * np.sqrt(alpha_s_mul_alpha_l) * rho_s_l * np.sin(theta) / np.sqrt(
                alpha_s_plus_alpha_l ** 2 - 4.0 * rho_sq * alpha_s_mul_alpha_l
            )
    )
    sigma_l = xl * alpha_l
    sigma_s = xs * alpha_s
    return sigma_l, sigma_s


class FactorVolParameters:
    def __init__(self, sigmas=(), alphas=()):
        self.sigmas = list(sigmas)
        self.alphas = list(alphas)

    def __bool__(self):
        return bool(self.sigmas) or bool(self.alphas)


class SigmaAlphaCalibrator:

    def __init__(
            self, lsu, calib_info, factor_count, covariance_calculators,
            with_lt_alpha=False, adjust_vol_for_rw=False,
            fixed_vol_parameters=FactorVolParameters()
    ):
        self.lsu = lsu
        self.calib_info = calib_info
        self.factor_count = factor_count
        self.covariance_calculators = covariance_calculators
        self.first_cc = covariance_calculators[sorted(covariance_calculators)[0]]
        assert (self.calib_info.probability_cond is None) is (self.calib_info.percentage_cond is None), (
            f"probability_cond and percentage_cond must be specified together, received {self.calib_info}"
        )
        self.with_prob_cond = self.calib_info.probability_cond is not None
        self.with_lt_alpha = with_lt_alpha or self.with_prob_cond
        self.fixed_vol_parameters = fixed_vol_parameters

        self.adjust_vol_for_rw = adjust_vol_for_rw
        if self.first_cc.vc_calib_params.robust_calibration:
            self.alpha_bounds = list(alphas_robust_x0_bnds)
        else:
            self.alpha_bounds = list(alphas_x0_bnds)

    def _init_rho(self, factor_count):
        sz = len(self.covariance_calculators)

        def corr(f1, f2):
            return sum(
                cc.measured_corr(self.lsu, f1, self.lsu, f2)
                for cc in self.covariance_calculators.values()
            ) / sz

        rho = np.diag(np.ones(factor_count))
        if factor_count >= 2:
            rho[0][1] = rho[1][0] = corr(0, 1)
        if factor_count == 3:
            rho[0][2] = rho[2][0] = corr(0, 2)
            rho[1][2] = rho[2][1] = corr(1, 2)
        return rho

    def _split_sigmas_and_alphas(self, data, factor_count):
        sigmas = self.fixed_vol_parameters.sigmas[:factor_count]
        alphas = self.fixed_vol_parameters.alphas[:factor_count]
        if not (self.with_lt_alpha or alphas):
            alphas = [0]  # when not with_lt_alpha, long term alpha is zero unless set as a fixed parameter
        assert len(data) + len(alphas) + len(sigmas) == 2 * factor_count, (
            f"Unexpected size of data {data} for {factor_count} factor model "
            f"with fixed_vol_parameters {self.fixed_vol_parameters} (with_lt_alpha={self.with_lt_alpha}). "
        )
        idx = factor_count - len(sigmas)
        sigmas.extend(data[:idx])
        alphas.extend(data[idx:])
        return sigmas, alphas

    def _split_sigmas_and_alphas_prob_condition(self, data, factor_count):
        prob_cond = self.calib_info.probability_cond
        percent_cond = self.calib_info.percentage_cond

        if factor_count == 1:
            assert len(data) == 1, f"unexpected size of data for {factor_count} factor model, received {data}"
            alpha_lt = data[0]
            sigma_lt = obtain_sigmas_from_1F_alphas(prob_cond, percent_cond, alpha_lt)
            return [sigma_lt], [alpha_lt]
        elif factor_count == 2:
            assert len(data) == 3, f"unexpected size of data for {factor_count} factor model, received {data}"
            x, alpha_lt, alpha_st = data
            rho = self._init_rho(factor_count)
            sigma_lt, sigma_st = obtain_sigmas_from_2F_alphas(
                probability_cond=prob_cond,
                percentage_cond=percent_cond,
                alpha_l=alpha_lt,
                alpha_s=alpha_st,
                alpha_to_sigma_mapping_parameter=x,
                rho_s_l=rho[0][1],
            )
            return [sigma_lt, sigma_st], [alpha_lt, alpha_st]
        else:
            raise AssertionError(f"Expected 1 or 2 factor model, data should have length 1 or 3, received {data}")

    def split_sigmas_and_alphas(self, data, factor_count):
        if self.with_prob_cond:
            assert not self.fixed_vol_parameters, "probability condition does not support fixed alphas or sigmas"
            return self._split_sigmas_and_alphas_prob_condition(data, factor_count)
        else:
            return self._split_sigmas_and_alphas(data, factor_count)

    def _long_term_fitting_contracts(self, vol_contracts):
        if self.calib_info.proxy_contracts:
            return []
        lt_factor_contract = self.calib_info.get_historical_factor_ratios()[0]
        lt_factor_contract_index = vol_contracts.index(lt_factor_contract)
        lt_sigma_contracts = vol_contracts[lt_factor_contract_index:]
        return lt_sigma_contracts

    def optimise_alpha_sigma(self, factor_count):
        if len(self.fixed_vol_parameters.sigmas) >= factor_count:
            return []

        vol_contracts_to_fit = self.calib_info.fitting_contracts(factor_count)
        time_spreads = self.calib_info.fitting_contracts(factor_count, spreads=True)
        lt_sigma_contracts = self._long_term_fitting_contracts(vol_contracts_to_fit)
        rho = self._init_rho(factor_count)
        lt_measured, measured = [], []
        for rw, cc in self.covariance_calculators.items():
            lt_measured.append(cc.measured_vols(lt_sigma_contracts))
            measured.append(cc.measured_vols(vol_contracts_to_fit))
            measured.append(cc.measured_vols(time_spreads))
        lt_measured, measured = np.concatenate(lt_measured), np.concatenate(measured)

        def sum_squared_diff(a, b):
            a_minus_b = a - b
            return np.sum(a_minus_b * a_minus_b)

        def cost_function(x, *args):
            sigmas, alphas = self.split_sigmas_and_alphas(x, factor_count)
            vol_corr = VolCorr(MeanRevertingProcesses([self.lsu] * factor_count, sigmas, alphas, rho))
            lt_modelled, modelled = [], []
            for _rw in self.covariance_calculators:
                rw = _rw if self.adjust_vol_for_rw else 0
                lt_modelled.append(calc_modelled_vols(self.lsu, vol_corr, lt_sigma_contracts, self.calib_info, rw))
                modelled.append(calc_modelled_vols(self.lsu, vol_corr, vol_contracts_to_fit, self.calib_info, rw))
                modelled.append(calc_modelled_ratio_vols(self.lsu, vol_corr, time_spreads, self.calib_info, rw))
            lt_modelled, modelled = np.concatenate(lt_modelled), np.concatenate(modelled)
            return sum_squared_diff(measured, modelled) + sum_squared_diff(lt_measured, lt_modelled)

        to_fit, bounds = initial_parameter_set(
            factor_count=factor_count,
            with_lt_alpha=self.with_lt_alpha,
            _sigmas_x0_bnds=sigmas_x0_bnds,
            _alphas_x0_bnds=self.alpha_bounds,
            fixed_alpha_len=len(self.fixed_vol_parameters.alphas),
            fixed_sigma_len=len(self.fixed_vol_parameters.sigmas),
            with_probability_condition=self.with_prob_cond
        )

        params, func_val, fit_info = opt.fmin_l_bfgs_b(
            cost_function,
            np.array(to_fit),
            approx_grad=True,
            bounds=bounds,
            disp=0,
            factr=10,
        )

        # check the opt lib did not raise a warning
        if fit_info["warnflag"] != 0:
            import logging
            log = logging.getLogger(__name__)
            log.warning(f"{self.lsu} failed to optimise: {fit_info}")

        return params

    def optimise(self, joint=False):
        """ The core of the vol-corr calibration logic: For one underlying and load shape, find a set of factors
        that minimise the difference between measured and modelled volatilities
        """
        if self.with_prob_cond:
            assert 0 < self.factor_count <= 2, "Only 1 and 2 factor calibrations supported with probability condition"
        else:
            assert 0 < self.factor_count <= 3, "Only 1, 2 and 3 factor calibrations supported"

        initial_factor_count = self.factor_count if joint else min(2, self.factor_count)
        res = self.optimise_alpha_sigma(initial_factor_count)
        sigmas_initial, alphas_initial = self.split_sigmas_and_alphas(res, initial_factor_count)
        if self.factor_count > max(initial_factor_count, 2):
            self.fixed_vol_parameters = FactorVolParameters(
                sigmas=sigmas_initial + self.fixed_vol_parameters.sigmas[initial_factor_count:],
                alphas=alphas_initial + self.fixed_vol_parameters.alphas[initial_factor_count:],
            )
            res = self.optimise_alpha_sigma(self.factor_count)
            sigmas, alphas = self.split_sigmas_and_alphas(res, self.factor_count)
        else:
            sigmas = sigmas_initial
            alphas = alphas_initial
        return sigmas, alphas


class VolCorrCalibrator:

    def __init__(self, calib_info_dict: CalibInfoDict, covariance_calculators, vc_calib_params=VcCalibParams()):
        self.calib_info_dict = calib_info_dict
        self.covariance_calculators = covariance_calculators
        sz = len(covariance_calculators)
        self.correl_avg = lambda f: sum([f(cc) for cc in self.covariance_calculators.values()]) / sz
        self.with_lt_alpha = vc_calib_params.with_lt_alpha
        self.lsu_dict = self._lsu_dict()
        total_num_factors = sum(len(indices) for (indices, _) in self.lsu_dict.values())
        self._underlying = [[] for _ in range(total_num_factors)]
        self._sigma = np.zeros(total_num_factors)
        self._alpha = np.zeros(total_num_factors)
        self._rho = np.identity(total_num_factors)
        self.adjust_vol_for_rw = vc_calib_params.adjust_vol_for_rw
        self.joint_calibration = vc_calib_params.joint_calibration

    @property
    def _historical_factors_list(self):
        # get the historical factor list for all lsus
        columns = []
        for underlying, calib_info in sorted(self.calib_info_dict.items()):
            for load_shape in calib_info.price_process_weekly_shapes:
                lsu = underlying.get_shaped(load_shape)
                hist_factors = calib_info.with_load_shape(load_shape).get_historical_factor_ratios()
                columns.extend([
                    (lsu, factor_id) for factor_id, factor in enumerate(hist_factors)
                    if (factor.is_ratio and factor.numerator.lsu.base_underlying == underlying) or
                       (not factor.is_ratio and factor.lsu.base_underlying == underlying) or
                       (not factor.is_ratio and factor.lsu.base_underlying.currency != underlying.currency)
                ])
        return sorted(columns)

    @property
    def _fixed_factors_list(self):
        # get the fixed factor list for all lsus
        columns = []
        for underlying, calib_info in sorted(self.calib_info_dict.items()):
            for load_shape, fixed_factors in calib_info.fixed_parameters.items():
                lsu = underlying.get_shaped(load_shape)
                columns.extend([
                    (lsu, factor_id) for factor_id, _ in enumerate(fixed_factors)
                ])
        return sorted(columns)

    def _lsu_dict(self):
        # dict from lsu to (indices, factors) in the volcorr matrix
        lsus = {
            underlying.get_shaped(load_shape)
            for underlying, calib_info in sorted(self.calib_info_dict.items())
            for load_shape in calib_info.price_process_weekly_shapes + list(calib_info.fixed_parameters)
        }
        lsu_to_indices_dict = {lsu: [] for lsu in lsus}
        lsu_to_factors_dict = {lsu: [] for lsu in lsus}
        i = 0
        for lsu, factor_id in self._historical_factors_list + self._fixed_factors_list:
            lsu_to_indices_dict[lsu].append(i)
            lsu_to_factors_dict[lsu].append(factor_id)
            i += 1
        return OrderedDict([(lsu, (lsu_to_indices_dict[lsu], lsu_to_factors_dict[lsu]))
                            for lsu in sorted(lsu_to_indices_dict)])

    @staticmethod
    def _dependents_have_been_calibrated(currency, proxy_underlying, lsus_left):
        dependents = [proxy_underlying]
        if currency != proxy_underlying.currency:
            dependents.extend([
                FX.get(currency, proxy_underlying.currency),
                FX.get(proxy_underlying.currency, currency),
            ])
        return set(lsus_left).isdisjoint(set(dependents))

    def _compute_fixed_params(self, lsu, calib_info):
        """
        Computes the sigmas and alphas from proxy contracts or fixed parameters.

        Assumes that dependents have been calculated,
        i.e. self._sigma and self._alpha are populated for proxy_contracts
        """
        fixed_sigmas = []
        fixed_alphas = []

        def get_index(lsu, factor):
            _indices, _factors = self.lsu_dict[lsu]
            return _indices[_factors.index(factor)]

        # If we use a proxy contract then take the sigmas and alphas from there.
        for factor_index, proxy_contract in enumerate(calib_info.proxy_contracts):
            proxy_lsu = proxy_contract.lsu.get_shaped(lsu.load_shape)
            proxy_index = get_index(proxy_lsu, factor_index)
            sigma_proxy = self._sigma[proxy_index]

            if proxy_lsu.currency == lsu.currency:
                self._underlying[proxy_index].append(lsu)
                sigma_lsu = sigma_proxy
            else:  # compo proxy, sigma should include FX
                # Compo: dont 'underlying[proxy_index].append(lsu)'
                if factor_index > 0:
                    raise AssertionError(f"Can only compo-proxy long term factor. "
                                         f"Trying to proxy {lsu} {factor_index}th factor with {proxy_lsu}")
                _fx = FX.get(lsu.currency, proxy_lsu.currency)
                fx, sgn, fx_factor_index = (_fx, 1, 0) if _fx in self.lsu_dict else (_fx.inverse, -1, 0)
                sigma_fx = self._sigma[get_index(fx, fx_factor_index)]
                correl_proxy_fx = self.correl_avg(
                    lambda cc: cc.measured_corr(fx, fx_factor_index, proxy_lsu, factor_index),
                )
                cov_fx = sgn * sigma_proxy * sigma_fx * correl_proxy_fx
                sigma_lsu = sqrt(max(1e-8, sigma_proxy * sigma_proxy + sigma_fx * sigma_fx + 2 * cov_fx))
            fixed_sigmas.append(sigma_lsu)
            fixed_alphas.append(self._alpha[proxy_index])

        fixed_parameters = calib_info.fixed_parameters.get(lsu.load_shape, [])
        if fixed_parameters:
            # fixing alphas, sigmas and rhos - if provided, this takes precedence to fixing the alphas only
            for fixed_params in fixed_parameters:
                fixed_sigmas.append(fixed_params.sigma)
                fixed_alphas.append(fixed_params.alpha)
        elif calib_info.alphas is not None:
            # fixing the alphas only - extending with the alphas on the non-proxied factors
            fixed_alphas.extend(calib_info.alphas[len(fixed_alphas):])

        return FactorVolParameters(sigmas=fixed_sigmas, alphas=fixed_alphas)

    def calibrate(self):
        """
        Control method to runs the calibration
        """
        # initialize vol corr parameters
        lsus_to_be_calibrated = sorted(self.lsu_dict.keys())
        # for each load_shaped_underlying
        while len(lsus_to_be_calibrated) > 0:
            lsu = lsus_to_be_calibrated.pop(0)
            calib_info = self.calib_info_dict[lsu]

            # back of the queue if some dependents are not calibrated yet
            if not all(self._dependents_have_been_calibrated(lsu.currency, proxy_ul.get_shaped(lsu.load_shape),
                                                             lsus_to_be_calibrated)
                       for proxy_ul in calib_info.proxy_underlyings):
                lsus_to_be_calibrated += [lsu]
                continue

            fixed_vol_parameters = self._compute_fixed_params(lsu, calib_info)

            # fit alpha and sigma parameters
            indices, factors = self.lsu_dict[lsu]
            if factors:
                calibrator = SigmaAlphaCalibrator(
                    lsu=lsu,
                    calib_info=calib_info,
                    factor_count=max(factors) + 1,
                    covariance_calculators=self.covariance_calculators,
                    with_lt_alpha=self.with_lt_alpha,
                    adjust_vol_for_rw=self.adjust_vol_for_rw,
                    fixed_vol_parameters=fixed_vol_parameters,
                )
                lsu_sigma, lsu_alpha = calibrator.optimise(self.joint_calibration)

                for factor_index1, f1 in zip(indices, factors):
                    # actual
                    self._underlying[factor_index1] = [lsu] + self._underlying[factor_index1]
                    self._sigma[factor_index1] = lsu_sigma[f1]
                    self._alpha[factor_index1] = lsu_alpha[f1]

                    # and update the correlations
                    for lsu2, (indices2, factors2) in self.lsu_dict.items():
                        calib_info2 = self.calib_info_dict[lsu2]
                        for factor_index2, f2 in zip(indices2, factors2):
                            if lsu != lsu2 or factor_index1 != factor_index2:
                                lsu_fixed_params = calib_info.fixed_parameters.get(lsu.load_shape)
                                lsu2_fixed_params = calib_info2.fixed_parameters.get(lsu2.load_shape)
                                if lsu_fixed_params:
                                    rho = lsu_fixed_params[f1].rho.get(lsu2, {}).get(f2, 0.0)
                                elif lsu2_fixed_params:
                                    rho = lsu2_fixed_params[f2].rho.get(lsu, {}).get(f1, 0.0)
                                else:
                                    rho = self.correl_avg(lambda cc: cc.measured_corr(lsu, f1, lsu2, f2))
                                self._rho[factor_index1, factor_index2] = self._rho[factor_index2, factor_index1] = rho

        return MeanRevertingProcesses(self._underlying, self._sigma.tolist(), self._alpha.tolist(), self._rho.tolist())


