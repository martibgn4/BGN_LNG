from numpy.linalg import LinAlgError

from pytest_quants import assert_almost_equal
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.test.fixtures import (
    power_mean_rev_procs, very_big_mean_rev_procs, mean_rev_procs,
)
from thorn.core.market.volatility.vc.vol_corr_operations import (
    bump_down_correlations, bump_up_vs_sigma, max_factor_num, scale_sigma, mean_rev_2f, correl_shift
)
from underlying import GBP, EUR
from underlying.fixtures.underlyings import NBP, TTF, UKPOWER, UKPOWER_OFFPEAK, UKPOWER_PEAK
import pytest


class TestVolCorrOperations:

    def test_scale_sigma(self):
        actual = scale_sigma(mean_rev_procs, NBP, 0.1)
        expected_sigma = [0.01, 0.015, 0.2, 0.2]
        assert_almost_equal(expected_sigma, actual.sigma)
        assert_almost_equal(mean_rev_procs.rho, actual.rho)
        assert_almost_equal(mean_rev_procs.alpha, actual.alpha)

    def test_bump_down_correlations(self):
        actual = bump_down_correlations(mean_rev_procs, NBP, UKPOWER_PEAK, 0.05)
        expected_rho = [
            [1., 0.8, 0.35, 0.55],
            [0.8, 1., 0.45, 0.65],
            [0.35, 0.45, 1., 0.3],
            [0.55, 0.65, 0.3, 1.]
        ]
        assert_almost_equal(expected_rho, actual.rho)
        assert_almost_equal(mean_rev_procs.sigma, actual.sigma)
        assert_almost_equal(mean_rev_procs.alpha, actual.alpha)

        inverted_bump = bump_down_correlations(mean_rev_procs, UKPOWER_PEAK, NBP, 0.05)
        assert_almost_equal(actual.rho, inverted_bump.rho)

        with pytest.raises(LinAlgError):
            _ = bump_down_correlations(mean_rev_procs, NBP, UKPOWER_PEAK, 2)

    def test_bump_down_correlations_with_fake_proxy(self):
        initial_rho = [
            [1., 0.8, 0.35, 0.9999],
            [0.8, 1., 0.45, 0.8],
            [0.35, 0.45, 1., 0.35],
            [0.9999, 0.8, 0.35, 1.]
        ]
        mrp = MeanRevertingProcesses(underlying=[NBP, UKPOWER, UKPOWER_OFFPEAK, UKPOWER_PEAK],
                                     sigma=[1, 2, 3, 4], alpha=[5, 6, 7, 8], rho=initial_rho)

        # shift a proxied ul
        shifted_mrp = bump_down_correlations(mrp, NBP, UKPOWER, 0.05)
        expected_rho = [
            [1., 0.75, 0.35, 0.9999],
            [0.75, 1., 0.45, 0.75],
            [0.35, 0.45, 1., 0.35],
            [0.9999, 0.75, 0.35, 1.]
        ]
        assert_almost_equal(expected_rho, shifted_mrp.rho)

        # now shift the equivalent pair -> nothing happens
        shifted_mrp = bump_down_correlations(mrp, NBP, UKPOWER_PEAK, 0.05)
        assert_almost_equal(mrp.rho, shifted_mrp.rho)

    def test_bump_down_correlations_with_double_fake_proxy(self):
        initial_rho = [
            [1., 0.8, 0.8, 0.9999],
            [0.8, 1., 0.9999, 0.8],
            [0.8, 0.9999, 1., 0.8],
            [0.9999, 0.8, 0.8, 1.]
        ]
        mrp = MeanRevertingProcesses(underlying=[NBP, TTF, UKPOWER, UKPOWER_PEAK],
                                     sigma=[1, 2, 3, 4], alpha=[5, 6, 7, 8], rho=initial_rho)

        # shift a proxied ul
        shifted_mrp = bump_down_correlations(mrp, NBP, UKPOWER, 0.05)
        expected_rho = [
            [1., 0.75, 0.75, 0.9999],
            [0.75, 1., 0.9999, 0.75],
            [0.75, 0.9999, 1., 0.75],
            [0.9999, 0.75, 0.75, 1.]
        ]
        assert_almost_equal(expected_rho, shifted_mrp.rho)

    def test_bump_down_correlations_with_full_partial_proxy(self):
        initial_rho = [
            [1., 0.8, 0.35],
            [0.8, 1., 0.45],
            [0.35, 0.45, 1.],
        ]
        mrp = MeanRevertingProcesses(underlying=[[NBP, UKPOWER_PEAK], UKPOWER, UKPOWER_OFFPEAK],
                                     sigma=[1, 2, 3], alpha=[5, 6, 7], rho=initial_rho)

        # shift a proxied ul
        shifted_mrp = bump_down_correlations(mrp, NBP, UKPOWER, 0.05)
        expected_rho = [
            [1., 0.75, 0.35],
            [0.75, 1., 0.45],
            [0.35, 0.45, 1.],
        ]
        assert_almost_equal(expected_rho, shifted_mrp.rho)
        assert_almost_equal(bump_down_correlations(mrp, UKPOWER, NBP, 0.05).rho, shifted_mrp.rho)

        # now shift the equivalent pair -> no bump
        shifted_mrp = bump_down_correlations(mrp, NBP, UKPOWER_PEAK, 0.05)
        assert_almost_equal(initial_rho, shifted_mrp.rho)

    def test_bump_up_vs_sigma(self):
        actual = bump_up_vs_sigma(power_mean_rev_procs, UKPOWER_PEAK, 0.1)
        expected_sigma = power_mean_rev_procs.sigma[:]
        expected_sigma[power_mean_rev_procs.lsu_to_indices[UKPOWER_PEAK][2]] *= 1.1
        assert_almost_equal(expected_sigma, actual.sigma)
        assert_almost_equal(power_mean_rev_procs.rho, actual.rho)
        assert_almost_equal(power_mean_rev_procs.alpha, actual.alpha)

        # No clone as mean_rev_proc is a 2-factor VolCorr
        actual = bump_up_vs_sigma(mean_rev_procs, NBP, 0.1)
        assert max_factor_num(mean_rev_procs) < 3, "Check test config: Expected 2F VolCorr"
        assert mean_rev_procs is actual

    def test_max_factor_num(self):
        assert 2 == max_factor_num(mean_rev_procs)
        assert 3 == max_factor_num(power_mean_rev_procs)

    def test_2f_vol_corr(self):
        actual = mean_rev_2f(power_mean_rev_procs)
        underlying = [UKPOWER_PEAK, UKPOWER_PEAK, UKPOWER_OFFPEAK, UKPOWER_OFFPEAK]
        alpha = [0.0, 3.0] + [0.0, 3.0]
        sigma = [0.2, 0.4] + [0.2, 0.4]
        rho = [
            [1, 0.1] + [0, 0], [0.1, 1] + [0, 0],
            [0, 0] + [1, 0.1], [0, 0] + [0.1, 1],
        ]
        expected_vc_2f = MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)
        assert expected_vc_2f == actual

    def test_correl_shift(self):
        shifted_mrp = correl_shift(very_big_mean_rev_procs, shift=0.2, numeraire_ccy=GBP)
        shifted_rho = [
            [1.0, 0.84, 0.52, 0.68, 0.84, 0.84, 0.2],
            [0.84, 1.0, 0.6, 0.76, 0.84, 0.84, 0.2],
            [0.52, 0.6, 1.0, 0.44, 0.6, 0.6, 0.2],
            [0.68, 0.76, 0.44, 1.0, 0.76, 0.76, 0.2],
            [0.84, 0.84, 0.6, 0.76, 1.0, 0.84, 0.2],
            [0.84, 0.84, 0.6, 0.76, 0.84, 1.0, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0],
        ]
        expected = very_big_mean_rev_procs.clone(rho=shifted_rho)
        assert_almost_equal(expected, shifted_mrp)

    def test_correl_shift_numeraire_ccy_change(self):
        shifted_mrp = correl_shift(very_big_mean_rev_procs, shift=0.2, numeraire_ccy=EUR)
        shifted_rho = [
            [1.0, 0.84, 0.52, 0.68, 0.84, -0.44, 0.2],
            [0.84, 1.0, 0.6, 0.76, 0.84, -0.44, 0.2],
            [0.52, 0.6, 1.0, 0.44, 0.6, -0.2, 0.2],
            [0.68, 0.76, 0.44, 1.0, 0.76, -0.36, 0.2],
            [0.84, 0.84, 0.6, 0.76, 1.0, -0.44, 0.2],
            [-0.44, -0.44, -0.2, -0.36, -0.44, 1.0, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0]
        ]
        expected = very_big_mean_rev_procs.clone(rho=shifted_rho)
        assert_almost_equal(expected.rho, shifted_mrp.rho)
        assert_almost_equal(expected.alpha, shifted_mrp.alpha)
        assert_almost_equal(expected.sigma, shifted_mrp.sigma)


