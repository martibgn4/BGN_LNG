from math import sqrt

from pytest_quants import assert_almost_equal
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.test.fixtures import (
    proxy_mean_rev_procs_2, proxy_mean_rev_procs, very_big_mean_rev_procs,
)
from thorn.iolayer.market.test.test_vol_corr_parser import list_of_rows, bad_list_of_rows, bad_list_of_proxy_rows
from time_period import PEAK, OFFPEAK, WEEKDAY_EFA5
from underlying.fixtures.underlyings import (
    NBP, TTF, UKPOWER_PEAK, UKPOWER, UKPOWER_OFFPEAK, DUTCHPOWER, BELGIANPOWER, GBPUSD, USDGBP, USDEUR, EURGBP, GBPEUR,
)
from underlying import GBP, EUR
import pytest


class TestMeanRevertingProcessess:

    def test_to_and_from_table_with_proxies(self):
        as_table = proxy_mean_rev_procs.to_table()
        assert [list(map(str, l)) for l in list_of_rows] == as_table
        from_table = MeanRevertingProcesses.from_table(as_table)
        # Note NBP is Proxied from TTF, so out of sort order
        assert proxy_mean_rev_procs == from_table

    def test_to_and_from_table_with_proxies2(self):
        as_table = proxy_mean_rev_procs_2.to_table()
        from_table = MeanRevertingProcesses.from_table(as_table)
        # Note NBP is Proxied from TTF, so out of sort order
        assert proxy_mean_rev_procs_2 == from_table

    def test_to_and_from_table_empty(self):
        empty = MeanRevertingProcesses([], [], [], [])
        as_table = empty.to_table()
        from_table = MeanRevertingProcesses.from_table(as_table)
        assert empty == from_table

    def test_to_and_from_table_empty_column(self):
        as_table = proxy_mean_rev_procs.to_table()
        as_table = [row + [""] for row in as_table]
        from_table = MeanRevertingProcesses.from_table(as_table)
        assert proxy_mean_rev_procs == from_table

    def test_to_and_from_table_empty_row(self):
        as_table = proxy_mean_rev_procs.to_table()
        as_table += ["" * len(as_table[0])]
        as_table += [""]
        from_table = MeanRevertingProcesses.from_table(as_table)
        assert proxy_mean_rev_procs == from_table

    def test_to_and_from_table_empty_row_and_column(self):
        as_table = proxy_mean_rev_procs.to_table()
        as_table = [row + [""] for row in as_table]
        as_table += ["" * len(as_table[0])]
        from_table = MeanRevertingProcesses.from_table(as_table)
        assert proxy_mean_rev_procs == from_table

    def test_from_bad_csv(self):
        with pytest.raises(AssertionError):
            _ = MeanRevertingProcesses.from_table(bad_list_of_rows)

    def test_check_proxied_factors(self):
        # test simple case works
        underlyings = ['NBP', 'NBP', 'NBP', 'TTF', 'TTF', 'TTF']
        factors = [0, 1, 2, 0, 1, 2]
        MeanRevertingProcesses._check_proxied_factors(underlyings, factors)

        # test good proxy set up works
        underlyings = [['NBP', 'TTF'], 'NBP', 'NBP', 'TTF', 'TTF']
        factors = [0, 1, 2, 1, 2]
        MeanRevertingProcesses._check_proxied_factors(underlyings, factors)

        # test misspecified proxy set up fails
        underlyings = [['NBP', 'TTF'], 'NBP', 'NBP', 'TTF', 'TTF', 'TTF']
        factors = [0, 1, 2, 0, 1, 2]
        with pytest.raises(AssertionError):
            MeanRevertingProcesses._check_proxied_factors(underlyings, factors)

    def test_from_bad_proxy_csv(self):
        # If an underlying has a proxied factor then we need to make sure it is specified twice
        with pytest.raises(AssertionError):
            _ = MeanRevertingProcesses.from_table(bad_list_of_proxy_rows)

    def test_reshape(self):
        underlyings = [NBP, EURGBP]
        actual = very_big_mean_rev_procs.reshape(underlyings)
        sigma = [0.1, 0.15, 0.08]
        alpha = [0.5, 10, 0.0]
        rho = [
            [1, 0.8, 0.8],
            [0.8, 1, 0.8],
            [0.8, 0.8, 1],
        ]
        underlying = [NBP] * 2 + [EURGBP]
        expected = MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)
        assert expected == actual

    def test_reshape_with_proxies(self):
        underlyings = [TTF, NBP]
        actual = proxy_mean_rev_procs.reshape(underlyings)
        sigma = [0.228, 0.334, 0.292]
        alpha = [0.000, 2.872, 2.829]
        rho = [
            [1.000, 0.050, 0.098],
            [0.050, 1.000, 0.919],
            [0.098, 0.919, 1.000]
        ]
        underlying = [[TTF, NBP], NBP] + [TTF]
        expected = MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)
        assert expected == actual

    def test_reshape_with_pivot_only(self):
        actual = proxy_mean_rev_procs.reshape(underlyings=[], ir_pivot_ccy=GBP)
        expected = MeanRevertingProcesses([], [], [], [])
        assert expected == actual

    def test_build_vol_corr_with_filter_and_pivot(self):
        underlyings = [TTF]
        # With native currency should remove FX
        actual = proxy_mean_rev_procs.reshape(underlyings, EUR)
        expected = proxy_mean_rev_procs.reshape([TTF])
        assert expected == actual

        # With non-underlying currency, should keep FX
        actual = proxy_mean_rev_procs.reshape(underlyings, GBP)
        expected = proxy_mean_rev_procs.reshape([TTF, EURGBP])
        assert expected == actual

        # With non-native currency should keep FX (and inverse)
        underlyings = [NBP]
        actual = proxy_mean_rev_procs.reshape(underlyings, EUR)
        expected = proxy_mean_rev_procs.reshape([NBP, EURGBP])
        assert expected == actual

    def test_to_numeraire_currency(self):
        # Build a mr processes with fx rates to test converison
        underlying = [GBPUSD, EURGBP, UKPOWER_PEAK, UKPOWER_PEAK]
        sigma = [0.2, 0.3, 0.4, 0.4]
        alpha = [0, 0, 0.2, 2.0]
        rho = [
            [1, 0.3, 0.5, -0.5],
            [0.3, 1, 0.1, 0.25],
            [0.5, 0.1, 1, 0.2],
            [-0.5, 0.25, 0.2, 1],
        ]
        mr_procs = MeanRevertingProcesses(underlying, sigma, alpha, rho)

        expected_underlying = [USDGBP, EURGBP, UKPOWER_PEAK, UKPOWER_PEAK]
        expected_rho = [
            [1, -0.3, -0.5, 0.5],
            [-0.3, 1, 0.1, 0.25],
            [-0.5, 0.1, 1, 0.2],
            [0.5, 0.25, 0.2, 1],
        ]
        expected_mr_procs = MeanRevertingProcesses(expected_underlying, sigma, alpha, expected_rho)
        gbp_mr_procs = mr_procs.to_numeraire_currency(GBP)
        assert gbp_mr_procs == expected_mr_procs

        # C1 = USD, C2=GBP, C3=EUR
        sigma_usdgbp = 0.2
        sigma_gbpeur = 0.3
        rho_usdgbp_gbpeur = 0.3
        sigma_usdeur = sqrt(
            sigma_usdgbp * sigma_usdgbp +
            sigma_gbpeur * sigma_gbpeur +
            2 * rho_usdgbp_gbpeur * sigma_usdgbp * sigma_gbpeur
        )
        # m = gbpeur
        rho_usdeur_gbpeur = (rho_usdgbp_gbpeur * sigma_usdgbp + 1 * sigma_gbpeur) / sigma_usdeur
        # m = 2
        rho_usdgbp_2 = -0.5
        rho_gbpeur_2 = -0.1
        rho_usdeur_2 = (rho_usdgbp_2 * sigma_usdgbp + rho_gbpeur_2 * sigma_gbpeur) / sigma_usdeur
        # m = 3
        rho_usdgbp_3 = 0.5
        rho_gbpeur_3 = -0.25
        rho_usdeur_3 = (rho_usdgbp_3 * sigma_usdgbp + rho_gbpeur_3 * sigma_gbpeur) / sigma_usdeur

        expected_sigma = [sigma_usdeur, 0.3, 0.4, 0.4]
        expected_underlying = [USDEUR, GBPEUR, UKPOWER_PEAK, UKPOWER_PEAK]
        expected_rho = [
            [1, rho_usdeur_gbpeur, rho_usdeur_2, rho_usdeur_3],
            [rho_usdeur_gbpeur, 1, -0.1, -0.25],
            [rho_usdeur_2, -0.1, 1, 0.2],
            [rho_usdeur_3, -0.25, 0.2, 1],
        ]
        expected_mr_procs = MeanRevertingProcesses(expected_underlying, expected_sigma, alpha, expected_rho)
        eur_mr_procs = mr_procs.to_numeraire_currency(EUR)
        assert_almost_equal(eur_mr_procs, expected_mr_procs)

    def test_get_intersecting_lsus_with_load_shape(self):
        # Peak and Offpeak cover UKPower
        assert [UKPOWER_PEAK, UKPOWER_OFFPEAK] == \
                         very_big_mean_rev_procs.get_intersecting_lsus(UKPOWER)
        # Peak exists in the set of simulated LSUs
        assert [UKPOWER_PEAK] == \
                         very_big_mean_rev_procs.get_intersecting_lsus(UKPOWER_PEAK)
        # WD5 requries Peak
        assert [UKPOWER_PEAK] == \
                         very_big_mean_rev_procs.get_intersecting_lsus(UKPOWER.get_shaped(WEEKDAY_EFA5))

    def test_get_intersecting_lsus_with_proxy_underlying(self):
        new_u = lambda lsu: DUTCHPOWER.get_shaped(lsu.load_shape) if lsu.base_underlying == UKPOWER else lsu
        underlying = [new_u(u) for u in very_big_mean_rev_procs.underlying]
        new_mean_rev_procs = very_big_mean_rev_procs.clone(underlying=underlying)
        assert [DUTCHPOWER.get_shaped(PEAK), DUTCHPOWER.get_shaped(OFFPEAK)] == \
                         new_mean_rev_procs.get_intersecting_lsus(BELGIANPOWER)
        assert [DUTCHPOWER.get_shaped(PEAK)] == \
                         new_mean_rev_procs.get_intersecting_lsus(BELGIANPOWER.get_shaped(PEAK))

    def test_get_intersecting_lsus_with_missing_underlying(self):
        # Dutch Power is not defined in very_big_mean_rev_procs - we return an empty list
        assert tuple() == very_big_mean_rev_procs.get_intersecting_lsus(DUTCHPOWER)


