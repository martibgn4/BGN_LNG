from collections import defaultdict
from datetime import date

import factory
import numpy as np
from scipy.stats import random_correlation

from general_utils import isiterable
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.vol_corr import VolCorr, SpotVolRatios, DatedSpotVolRatios
from time_period import BASE, PEAK, OFFPEAK
from underlying import FX, Commodity
from underlying.fixtures.underlyings import (
    EUA, EURGBP, NBP, TTF, BRENT, API2, USDGBP, UKPOWER, UKPOWER_PEAK, UKPOWER_OFFPEAK, GBPUSD,
    GASOIL, FUELOIL, HEATHROWTEMP,
    DUTCHPOWER_OFFPEAK, DUTCHPOWER_PEAK, GBPEUR, HENRYHUB, DEPOWER, DK1POWER, DEPOWER_PEAK, DEPOWER_OFFPEAK,
)


class SpotVolRatiosFactory(factory.Factory):
    class Meta:
        model = SpotVolRatios

    lsu_ratios_dict = {
        NBP: [0.5 + 1 / 24 + i / 12 for i in range(12)],
        TTF: [1.0] * 12,
        UKPOWER_PEAK: [0.5 + 1 / 24 + i / 12 for i in range(12, 0, -1)],
    }

    @classmethod
    def build_small(cls, underlying):
        return cls.build(lsu_ratios_dict={underlying: [0.5 + 1 / 24 + i / 12 for i in range(12)]})

    @classmethod
    def build_random(cls, rng, underlyings):
        ratios = {
            underlying: [rng.uniform(0.5, 1.5) for _ in range(12)]
            for underlying in underlyings
        }
        # Normalise the values
        ratios = {key: [x / sum(value) for x in value] for key, value in ratios.items()}
        return cls.build(lsu_ratios_dict=ratios)


class DatedSpotVolRatiosFactory(factory.Factory):
    class Meta:
        model = DatedSpotVolRatios

    pricing_date = date(2018, 1, 15)
    spot_vol_ratios = SpotVolRatiosFactory.build()


class MeanRevertingProcessFactory(factory.Factory):
    class Meta:
        model = MeanRevertingProcesses

    sigma = [0.1, 0.15, 0.2, 0.2]
    alpha = [0.5, 10, 0.1, 20]
    rho = [
        [1.0, 0.8, 0.4, 0.6],
        [0.8, 1.0, 0.5, 0.7],
        [0.4, 0.5, 1.0, 0.3],
        [0.6, 0.7, 0.3, 1.0],
    ]
    underlying = [NBP, NBP, UKPOWER_PEAK, UKPOWER_PEAK]

    @classmethod
    def build_random(cls, rng, underlyings, factor_count=None, zero_lt_alpha=False, uncorrelated=False):
        # Ensure the underlyings are a set of disjoint shaped underlyings
        load_shapes = defaultdict(set)
        for ul in underlyings:
            load_shapes[ul.base_underlying].add(ul.load_shape)

        # sort on iteration so that order does not impact randomness
        shaped_underlyings = set.union(set(), *[
            {ul.get_shaped(ls) for ls in ({BASE} if load_shapes == {BASE} else {PEAK, OFFPEAK})}
            for ul, load_shapes in load_shapes.items()
        ])

        # Build the list of Factors, with sigma and alphas - Note FX are single factor, with no decay
        underlying_factors = []
        alpha = []
        sigma = []
        for u in sorted(shaped_underlyings, reverse=True):
            if isinstance(u, FX):
                underlying_factors.append(u)
                sigma.append(rng.uniform(0.05, 0.4))
                alpha.append(0.0)
            if isinstance(u, Commodity):
                underlying_factors += [u] * (factor_count or 2)
                sigma += [rng.uniform(0.05, 0.4) for _ in range(factor_count or 2)]
                alpha += [0.0 if zero_lt_alpha and i == 0 else rng.uniform(0.0, 2.0) for i in range(factor_count or 2)]

        if uncorrelated:
            rho = np.identity(len(underlying_factors))
        elif len(underlying_factors) > 1:
            # Choose random correlations
            eigenvalues = np.array([rng.uniform(0.001, 1) for _ in underlying_factors])
            eigenvalues *= len(eigenvalues) / sum(eigenvalues)
            rho = random_correlation.rvs(eigenvalues, np.random.default_rng(seed=rng.integers(0, 9999999)))
        else:
            rho = [[1]]

        return cls.build(underlying=underlying_factors, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_small(cls, underlying, sigma=None):
        """Builds the smallest possible vol corr for a single underlying."""
        return cls.build(sigma=[sigma or 0], alpha=[0], rho=[[1]], underlying=[underlying])

    @classmethod
    def build_small_for_multiple_underlyings(cls, underlyings, sigma=None):
        """Builds the smallest possible vol corr for multiple underlyings."""
        N = len(underlyings)
        return cls.build(sigma=[sigma or 0] * N if not isiterable(sigma) else sigma,
                         alpha=[0] * N, rho=np.identity(N), underlying=underlyings)

    @classmethod
    def build_modified_vc(cls):
        modified_alpha = [1.5, 20, 1.1, 30]
        modified_sigma = [1.1, 1.15, 1.2, 1.2]
        return cls.build(sigma=modified_sigma, alpha=modified_alpha)

    @classmethod
    def build_very_big_vc(cls, underlying=None):
        sigma = [0.1, 0.15, 0.2, 0.2, 0.5, 0.08, 0.2]
        alpha = [0.5, 10, 0.1, 20, 0.0, 0.0, 0.15]
        very_big_rho = [
            [1, 0.8, 0.4, 0.6, 0.8, 0.8, 0],
            [0.8, 1, 0.5, 0.7, 0.8, 0.8, 0],
            [0.4, 0.5, 1, 0.3, 0.5, 0.5, 0],
            [0.6, 0.7, 0.3, 1, 0.7, 0.7, 0],
            [0.8, 0.8, 0.5, 0.7, 1, 0.8, 0],
            [0.8, 0.8, 0.5, 0.7, 0.8, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]
        if underlying is None:
            underlying = [NBP] * 2 + [UKPOWER_PEAK] * 2 + [EUA, EURGBP, UKPOWER_OFFPEAK]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=very_big_rho)

    @classmethod
    def build_very_big_dutch_vc(cls, underlying=None):
        sigma = [0.1, 0.15, 0.2, 0.2, 0.5, 0.08, 0.2]
        alpha = [0.5, 10, 0.1, 20, 0.0, 0.0, 0.15]
        very_big_rho = [
            [1, 0.8, 0.4, 0.6, 0.8, 0.8, 0],
            [0.8, 1, 0.5, 0.7, 0.8, 0.8, 0],
            [0.4, 0.5, 1, 0.3, 0.5, 0.5, 0],
            [0.6, 0.7, 0.3, 1, 0.7, 0.7, 0],
            [0.8, 0.8, 0.5, 0.7, 1, 0.8, 0],
            [0.8, 0.8, 0.5, 0.7, 0.8, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]
        if underlying is None:
            underlying = [NBP] * 2 + [DUTCHPOWER_PEAK] * 2 + [EUA, GBPEUR, DUTCHPOWER_OFFPEAK]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=very_big_rho)

    @classmethod
    def build_nbp_ttf_mr_procs(cls):
        sigma = [0.110, 0.228, 0.334, 0.231, 0.292]
        alpha = [0.000, 0.000, 2.872, 0.000, 2.829]
        rho = [
            [1.000, 0.167, -0.058, -0.265, -0.050],
            [0.167, 1.000, 0.050, 0.877, 0.098],
            [-0.058, 0.050, 1.000, 0.131, 0.919],
            [-0.265, 0.877, 0.131, 1.000, 0.126],
            [-0.050, 0.098, 0.919, 0.126, 1.000]
        ]
        underlying = [EURGBP] + [NBP] * 2 + [TTF] * 2
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_solepit_mr_procs(cls):
        sigma = [0.110, 0.228, 0.334, 0.231, 0.292]
        alpha = [0.000, 0.000, 2.872, 0.000, 2.829]
        rho = [
            [1.000, 0.167, -0.058, -0.265, -0.050],
            [0.167, 1.000, 0.050, 0.877, 0.098],
            [-0.058, 0.050, 1.000, 0.131, 0.919],
            [-0.265, 0.877, 0.131, 1.000, 0.126],
            [-0.050, 0.098, 0.919, 0.126, 1.000]
        ]
        underlying = [GBPUSD] + [GASOIL] * 2 + [FUELOIL] * 2
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_power_mr_procs(cls, power_underlying=UKPOWER):
        alpha = [0.0, 3.0, 80.0] + [0.0, 3.0, 80.0]
        sigma = [0.2, 0.4, 3.0] + [0.2, 0.4, 3.0]
        rho = [
            [1, 0.1, 0.01] + [0, 0, 0], [0.1, 1, 0] + [0, 0, 0], [0.01, 0, 1] + [0, 0, 0],
            [0, 0, 0] + [1, 0.1, 0.01], [0, 0, 0] + [0.1, 1, 0], [0, 0, 0] + [0.01, 0, 1],
        ]
        underlying = [power_underlying.get_shaped(PEAK)] * 3 + [power_underlying.get_shaped(OFFPEAK)] * 3
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_brent_mr_procs(cls):
        sigma = [0.1386, 0.7195, 1.23, 2.34]
        alpha = [0.0, 0.7195, 1.23, 2.34]
        rho = [
            [1, 0.4521, 0, 0],
            [0.4521, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        underlying = [BRENT] * 2 + [API2, USDGBP]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_svc_mr_procs(cls):
        sigma = [0.2, 0.21, 0.1, 0.2]
        alpha = [0, 0, 0.02, 0]  # VC1F for EUA has non-zero alpha
        rho = [
            [1, 0.9, 0.9, 0],
            [0.9, 1, 0.9, 0],
            [0.9, 0.9, 1, 0],
            [0, 0, 0, 1]
        ]
        underlying = [DEPOWER, DK1POWER, EUA, EURGBP]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_svc_mr_procs_with_load_shape(cls):
        sigma = [0.2, 0.2, 0.21, 0.2]
        alpha = [0, 0, 0, 0]
        rho = [
            [1, 0.9, 0.9, 0],
            [0.9, 1, 0.9, 0],
            [0.9, 0.9, 1, 0],
            [0, 0, 0, 1]
        ]
        underlying = [DEPOWER_PEAK, DEPOWER_OFFPEAK, DK1POWER, EURGBP]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_temperature_vc(cls):
        sigma = [0.2052, 0.3383, 0.6142, 42.85724]
        alpha = [0, 2.956, 29.8412, 116]
        rho = [
            [1, 0.0249, 0.012, -0.0017],
            [0.0249, 1, 0.0962, -0.046],
            [0.012, 0.0962, 1, -0.3203],
            [-0.0017, -0.046, -0.3203, 1]
        ]
        underlying = [NBP] * 3 + [HEATHROWTEMP]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_proxy_example(cls):
        sigma = [0.110, 0.228, 0.334, 0.292]
        alpha = [0.000, 0.000, 2.872, 2.829]
        rho = [
            [1.000, 0.167, -0.058, -0.050],
            [0.167, 1.000, 0.050, 0.098],
            [-0.058, 0.050, 1.000, 0.919],
            [-0.050, 0.098, 0.919, 1.000]
        ]
        underlying = [EURGBP] + [[TTF, NBP], NBP] + [TTF]  # TTF is the root of the Pair, yet sorts after NBP
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)

    @classmethod
    def build_proxy_example_2(cls):
        sigma = [0.110, 0.228, 0.334, 0.292, 0.6, 0.6]
        alpha = [0.000, 0.000, 2.872, 2.829, 0.7, 0.6]
        rho = [
            [1.000, 0.167, -0.058, -0.050, 0.0, 0.0],
            [0.167, 1.000, 0.050, 0.098, 0.0, 0.0],
            [-0.058, 0.050, 1.000, 0.919, 0.0, 0.0],
            [-0.050, 0.098, 0.919, 1.000, 0.0, 0.0],
            [-0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]
        underlying = [EURGBP] + [[TTF, NBP], NBP] + [HENRYHUB, HENRYHUB, TTF]
        return cls.build(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)


mean_rev_procs = MeanRevertingProcessFactory.build()
modified_mean_rev_procs = MeanRevertingProcessFactory.build_modified_vc()
very_big_mean_rev_procs = MeanRevertingProcessFactory.build_very_big_vc()
nbpttf_mean_rev_procs = MeanRevertingProcessFactory.build_nbp_ttf_mr_procs()
solepit_mean_rev_procs = MeanRevertingProcessFactory.build_solepit_mr_procs()
power_mean_rev_procs = MeanRevertingProcessFactory.build_power_mr_procs()
temperature_mean_rev_procs = MeanRevertingProcessFactory.build_temperature_vc()
proxy_mean_rev_procs = MeanRevertingProcessFactory.build_proxy_example()
proxy_mean_rev_procs_2 = MeanRevertingProcessFactory.build_proxy_example_2()

vol_corr = VolCorr(mean_rev_procs)
modified_vol_corr = VolCorr(modified_mean_rev_procs)
very_big_vol_corr = VolCorr(very_big_mean_rev_procs)
nbpttf_vol_corr = VolCorr(nbpttf_mean_rev_procs)
solepit_vol_corr = VolCorr(solepit_mean_rev_procs)
power_vol_corr = VolCorr(power_mean_rev_procs)
temperature_vol_corr = VolCorr(temperature_mean_rev_procs)
proxy_vol_corr = VolCorr(proxy_mean_rev_procs)

spot_vol_ratios = SpotVolRatiosFactory.build()
dated_spot_vol_ratios = DatedSpotVolRatiosFactory.build()


