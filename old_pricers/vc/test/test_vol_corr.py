from datetime import date, timedelta
from math import exp, sqrt

import numpy as np
import pytest

from general_utils import time_between
from pytest_quants import assert_almost_equal
from thorn.core.market.dynamic.sigma_t_function_class import (
    CalibratedSingleExpirySigmaT, SigmaTFunction, SigmaTFunctionDict,
)
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.test.fixtures import (
    MeanRevertingProcessFactory, nbpttf_vol_corr, power_mean_rev_procs, proxy_vol_corr,
    spot_vol_ratios, very_big_vol_corr, mean_rev_procs, DatedSpotVolRatiosFactory,
)
from thorn.core.market.volatility.vc.vol_corr import DatedSpotVolRatios, SpotVolRatios, VolCorr
from thorn.core.math_utils.function_types import PiecewiseConstantFunction, PiecewiseConstantDict
from underlying import EUR, GBP
from underlying.fixtures.underlyings import (
    EUA, EURGBP, GBPEUR, NBP, TTF, UKPOWER_OFFPEAK, UKPOWER_PEAK, GBPUSD, UKPOWER,
)


class TestVolCorr:

    def setup_method(self):
        self.vol_corr = VolCorr(mean_rev_procs)

    def test_calc_covariance(self):
        # Cf test_vol_corr.VolCorrTestCase.test_calc_covariance

        variance = self.vol_corr.calc_covariance(UKPOWER_PEAK, UKPOWER_PEAK, t=0.01, T1=0.01, D1=0.00001)
        assert_almost_equal(
            variance,
            0.000946255,
            atol=1e-6)

    def test_calc_covariance2(self):
        T1 = 0.01
        D1 = 0.00001
        T2 = 0.02
        D2 = 0.01
        assert_almost_equal(
            self.vol_corr.calc_covariance(UKPOWER_PEAK, NBP, t=0.01, T1=T1, D1=D1, T2=T2, D2=D2),
            0.000465899,
            atol=1e-6)

    def test_calc_inst_vol_corr(self):
        T = 0.2
        D = 0.1
        assert_almost_equal(
            self.vol_corr.calc_inst_vol(UKPOWER_PEAK, T, D),
            0.195543738
        )
        assert_almost_equal(
            self.vol_corr.calc_inst_vol(NBP, T, D),
            0.098825049
        )
        assert_almost_equal(
            self.vol_corr.calc_inst_corr(UKPOWER_PEAK, NBP, T, D),
            0.426194457
        )
        assert_almost_equal(
            self.vol_corr.calc_corr(UKPOWER_PEAK, NBP, 0.1, T, D),
            0.446090982
        )

    def test_sigma_T_zero_alpha(self):
        """
        Calc M_B where alpha=0
        Becasue alpha=0 M_B is just the average of sigmaT over the domain [T, T+D]
        """
        xis = [0, 0.5, 1]
        yis = [0.9, 1.1, 1.2]
        sigma_t_function = SigmaTFunction(
            PiecewiseConstantDict({
                t: CalibratedSingleExpirySigmaT({NBP: sigmaT})
                for t, sigmaT in zip(xis, yis)
            }, extrapolate_left=True),
        )
        vol_corr = VolCorr(MeanRevertingProcessFactory.build_small(NBP, sigma=1.0)).clone(
            sigmaT_funcs=SigmaTFunctionDict({NBP: sigma_t_function}))
        t = 0
        T = 0
        D = 2
        avg = PiecewiseConstantFunction(xis, [yis[0]] + yis).integrate(T, T + D) / D
        assert_almost_equal(vol_corr.M_B(NBP, t, T, D), avg)

    def test_sigma_T_zero_piece(self):
        """
        Calc M_B where alpha=!0
        Because sigmaT is zero except for one piece, we can calculate
        M_B over the non-zero piece without sigmaT and scale.
        The Scaline is the value of sigmaT on the non-zero part multiplied
        by the ratio of the non-zero part to the total domain
        """
        sigma_t_function = SigmaTFunction(
            PiecewiseConstantDict({
                t: CalibratedSingleExpirySigmaT({NBP: sigmaT})
                for t, sigmaT in zip([0, 0.5, 1], [0, 1.1, 0])
            }, extrapolate_left=True),
        )
        vol_corr = self.vol_corr.clone(sigmaT_funcs=SigmaTFunctionDict({NBP: sigma_t_function}))
        t = 0
        T = 0
        D = 2
        assert_almost_equal(vol_corr.M_B(NBP, t, T, D),
                               vol_corr._M_B_no_sigmaT(NBP, t, 0.5) * vol_corr._d_correction(NBP, 0.5) * 1.1 * 0.5 / D)

    def test_Ps(self):
        '''Test that the Ps are normalised as expected and that they are stable from T==0 to T==dt'''
        lsu = NBP
        assert [0, 1] == self.vol_corr.lsu_mapping([lsu])[0]  # Check vol_corr has two factors for NBP at 0, 1
        rho = mean_rev_procs.rho[0][1]
        T = 1.1
        Pss = [self.vol_corr.calc_P(lsu, tau=T - ex_time, D=0.1) for ex_time in (0.0, 0.001, 0.5)]
        for Ps in Pss:
            assert 2 == len(Ps), 'Expected two P values, one for each factor, received: %s' % len(Ps)
            sumsq = sum(p * p for p in Ps) + 2 * Ps[0] * Ps[1] * rho
            assert pytest.approx(1.0) == sumsq  # Assert sum as Expected
        assert_almost_equal(Pss[0], Pss[1], atol=1e-3)  # Assert stable between T==0 and T==dt

    def test_single_factor_Ps(self):
        '''If only one factor then the Ps should be [1.0]'''
        lsu = NBP
        vol_corr = VolCorr(MeanRevertingProcessFactory.build_small(lsu, sigma=0.5))
        T = 1.1
        Pss = [vol_corr.calc_P(lsu, tau=T - ex_time, D=0.1) for ex_time in (0.0, 0.001, 0.5)]
        for Ps in Pss:
            assert [1.0] == Ps

    def test_calc_vol_with_fx(self):
        eur_vol_corr = nbpttf_vol_corr.to_numeraire_currency(EUR)
        t = 0.2
        T = 0.25
        D = 0.1

        var = eur_vol_corr.calc_covariance(NBP, NBP, t, T, D, include_fx=True)
        var_nbp = eur_vol_corr.calc_covariance(NBP, NBP, t, T, D)
        var_fx = eur_vol_corr.calc_covariance(GBPEUR, GBPEUR, t, T, D)
        cov_nbp_fx = eur_vol_corr.calc_covariance(NBP, GBPEUR, t, T, D)
        var_nbp_in_eur = var_nbp + var_fx + 2 * cov_nbp_fx
        assert_almost_equal(var, var_nbp_in_eur)

        sigma = eur_vol_corr.calc_vol(NBP, t, T, D, include_fx=True)
        assert sigma * sigma * t == var

    def test_calc_vol_ratio(self):
        T = 0.25
        D = 0.1
        actual = nbpttf_vol_corr.calc_ratio_vol({NBP}, {EURGBP, TTF}, 0, T, D)
        eur_vol_corr = nbpttf_vol_corr.to_numeraire_currency(EUR)
        nbp_vol = eur_vol_corr.calc_inst_vol(NBP, T, D, include_fx=True)
        ttf_vol = eur_vol_corr.calc_inst_vol(TTF, T, D)
        corr = eur_vol_corr.calc_inst_corr(NBP, TTF, T, D, include_fx=True)
        expected = sqrt(ttf_vol ** 2 + nbp_vol ** 2 - 2 * ttf_vol * nbp_vol * corr)
        assert_almost_equal(actual, expected)

        t = 0.1
        actual = nbpttf_vol_corr.calc_ratio_vol({NBP}, {EURGBP, TTF}, t, T, D)
        eur_vol_corr = nbpttf_vol_corr.to_numeraire_currency(EUR)
        nbp_vol = eur_vol_corr.calc_vol(NBP, t, T, D, include_fx=True)
        ttf_vol = eur_vol_corr.calc_vol(TTF, t, T, D)
        corr = eur_vol_corr.calc_corr(NBP, TTF, t, T, D, include_fx=True)
        expected = sqrt(ttf_vol ** 2 + nbp_vol ** 2 - 2 * ttf_vol * nbp_vol * corr)
        assert_almost_equal(actual, expected)

        t = 0.1
        T2 = T * 2
        D2 = D
        actual = nbpttf_vol_corr.calc_ratio_vol({NBP}, {NBP}, t, T, D, T2, D2)
        eur_vol_corr = nbpttf_vol_corr.to_numeraire_currency(EUR)
        nbp_vol = eur_vol_corr.calc_vol(NBP, t, T, D, include_fx=True)
        nbp2_vol = eur_vol_corr.calc_vol(NBP, t, T2, D2, include_fx=True)
        corr = eur_vol_corr.calc_corr(NBP, NBP, t, T, D, T2=T2, D2=D2, include_fx=True)
        expected = sqrt(nbp_vol ** 2 + nbp2_vol ** 2 - 2 * nbp2_vol * nbp_vol * corr)
        assert_almost_equal(actual, expected)

    def test_lsu_mapping(self):
        lsus = very_big_vol_corr.lsus
        assert lsus == [NBP, UKPOWER_PEAK, UKPOWER_OFFPEAK, EUA, EURGBP]

        actual_simulated_indices, actual_lsu_to_simulated_indices = very_big_vol_corr.lsu_mapping(
            very_big_vol_corr.lsus)
        expected_simulated_indices = [0, 1, 2, 3, 4, 5, 6]
        expected_lsu_to_simulated_indices = {
            NBP: [0, 1],
            UKPOWER_PEAK: [2, 3],
            UKPOWER_OFFPEAK: [6],
            EUA: [4],
            EURGBP: [5]
        }
        assert expected_simulated_indices == actual_simulated_indices
        assert expected_lsu_to_simulated_indices == actual_lsu_to_simulated_indices

    def test_lsu_mapping_proxies(self):
        actual_simulated_indices, actual_lsu_to_simulated_indices = proxy_vol_corr.lsu_mapping(
            proxy_vol_corr.lsus)
        expected_simulated_indices = [0, 1, 2, 3]
        expected_lsu_to_simulated_indices = {
            EURGBP: [0],
            NBP: [1, 2],
            TTF: [1, 3]
        }
        assert expected_simulated_indices == actual_simulated_indices
        assert expected_lsu_to_simulated_indices == actual_lsu_to_simulated_indices

    def test_proxied_vol(self):
        # TTF LT factor is proxied from NBP - expect LT vol to be the same
        nbp_vol = proxy_vol_corr.calc_vol(NBP, 0, 5, 1 / 365)
        ttf_vol = proxy_vol_corr.calc_vol(TTF, 0, 5, 1 / 365)
        assert_almost_equal(nbp_vol, ttf_vol)

        # ST vol should be different
        nbp_vol = proxy_vol_corr.calc_vol(NBP, 0, 1, 1 / 365)
        ttf_vol = proxy_vol_corr.calc_vol(TTF, 0, 1, 1 / 365)
        assert round(abs(nbp_vol-ttf_vol), 7) != 0

        # LT corr should be 1
        lt_corr = proxy_vol_corr.calc_inst_corr(NBP, TTF, 3, 1 / 365)
        assert_almost_equal(1, lt_corr)

        # ST corr should not be 1
        st_corr = proxy_vol_corr.calc_inst_corr(NBP, TTF, 0.1, 1 / 365)
        assert round(abs(1-st_corr), 7) != 0

    def test_vol_in_limit_t_inf_1factor(self):
        """
        Create a 1Factor VC
        Test that vol in limit is as per analytical soln
        """
        lsu = NBP
        # Non-zero alpha -> vol tends to limit
        # Test with D non-zero for coverage - use D = 1 in variance targetting calibration
        for sigma, alpha, D in (
            (0.5, 0.0001, 1.0),
            (0.001, 0.001, 1.0),
            (0.001, 0.001, 0.001),
            (0.5, 1e-18, 1.0),
        ):
            # See VC DDD: A.5 Probability condition - calibrating a long term \alpha
            alpha_scale = (1 - exp(-alpha * D)) / (alpha * D)
            term_var = sigma * sigma / (2 * alpha) * alpha_scale * alpha_scale

            vol_corr = VolCorr(
                MeanRevertingProcesses(
                    sigma=[sigma],
                    alpha=[alpha],
                    rho=[[1.0]],
                    underlying=[lsu],
                ),
            )

            t = float("inf")
            T = float("inf")
            actual = vol_corr.calc_var(lsu, t, T, D)
            expected = term_var if alpha > 1e-10 else float("inf")
            assert_almost_equal(expected, actual)

            # Cross check to behaviour at large t and T
            almost_inf = 1e6
            expected = term_var if alpha > 1e-10 else almost_inf * sigma * sigma  # t * sigma**2
            actual_limit = vol_corr.calc_var(lsu, t=almost_inf, T=almost_inf, D=D)
            assert_almost_equal(expected, actual_limit)

    def test_vol_in_limit_t_inf_2factor(self):
        """ Checks that calc_var for large t converges to that for inf """
        lsu = NBP

        # Non-zero alpha -> vol tends to limit
        for sigmas, alphas, rhos, D in (
            ([0.2, 0.5], [0.0001, 1.0], [[1.0, 0.0], [0.0, 1.0]], 1),  # Non-zero alpha, no correlation
            ([0.2, 0.5], [0.0001, 1.0], [[1.0, 0.1], [0.1, 1.0]], 1),  # Non-zero alpha, with correlation
            (
                [0.2, 0.5, 1.0],
                [0.0001, 0.1, 1.0],
                [[1.0, 0.1, 0.01], [0.1, 1.0, 0.2], [0.01, 0.2, 1.0]],
                1
            ),  # 3 factor, Non-zero alpha, with correlation
        ):
            vol_corr = VolCorr(
                MeanRevertingProcesses(
                    sigma=sigmas,
                    alpha=alphas,
                    rho=rhos,
                    underlying=[lsu for _ in sigmas],
                ),
            )

            t = float("inf")
            T = float("inf")
            actual_limit = vol_corr.calc_var(lsu, t, T, D)

            # Cross check to behaviour at large t and T
            large_t = 1e6
            actual_large_t = vol_corr.calc_var(lsu, t=large_t, T=large_t, D=D)
            assert_almost_equal(actual_large_t, actual_limit)


class TestSeasonalVolCorr(TestVolCorr):
    """The Seasonal VolCorr should behave like a VolCorr hence the test case sub class."""

    def setup_method(self):
        self.dated_spot_vol_ratios = DatedSpotVolRatiosFactory.build()
        self.mean_rev_docs = mean_rev_procs
        self.equivalent_vol_corr = VolCorr(mean_rev_procs)
        self.vol_corr = VolCorr(self.mean_rev_docs, spot_vol_ratios=self.dated_spot_vol_ratios)

    def test_vol_ratio_2f(self):
        t = 0.2
        T = 0.25
        D = 0.1
        sigma = self.vol_corr.calc_vol(NBP, t, T, D)
        equiv_sigma = self.equivalent_vol_corr.calc_vol(NBP, t, T, D)
        assert equiv_sigma == sigma

    def test_vol_ratio_3fs(self):
        # TODO: test the seasonal vol / spot ratios behaviour
        equivalent_vol_corr = VolCorr(power_mean_rev_procs)
        vol_corr = VolCorr(power_mean_rev_procs, spot_vol_ratios=self.dated_spot_vol_ratios)

        t = 0.2
        T = 0.25
        D = 0.1
        sigma = vol_corr.calc_vol(UKPOWER_PEAK, t, T, D)
        equiv_sigma = equivalent_vol_corr.calc_vol(UKPOWER_PEAK, t, T, D)
        # PricingDate is 15/1/2018, and ratios are > 1.0 for start of year -> sigma goes up
        assert equiv_sigma < sigma

        sigma = vol_corr.calc_vol(UKPOWER_OFFPEAK, t, T, D)
        equiv_sigma = equivalent_vol_corr.calc_vol(UKPOWER_OFFPEAK, t, T, D)
        assert equiv_sigma == sigma  # No ratios for OFFPEAK

    def test_spot_vol_ratios(self, subtests):
        equivalent_vol_corr = VolCorr(power_mean_rev_procs)
        vol_corr = VolCorr(power_mean_rev_procs, spot_vol_ratios=self.dated_spot_vol_ratios)

        test_cases = [
            ((UKPOWER_PEAK, 0.2, 0.3), lambda x, y: (x < y) or pytest.fail(f"{x} !< {y}")),  # Start of year - ratios > 1
            ((UKPOWER_PEAK, 0.7, 0.8), lambda x, y: (x > y) or pytest.fail(f"{x} !> {y}")),  # Start of year - ratios < 1
            ((UKPOWER_OFFPEAK, 0.7, 0.8), assert_almost_equal),  # No ratios so equal
        ]
        for (lsu, t, T), assert_comp_func in test_cases:
            with subtests.test(f"{lsu}, {t}, {T}"):
                equivalent_M_B = equivalent_vol_corr._M_B_no_sigmaT(lsu, t, T)
                actual_M_B = vol_corr._M_B_no_sigmaT(lsu, t, T)
                np.testing.assert_equal(equivalent_M_B[:-1], actual_M_B[:-1])
                assert_comp_func(equivalent_M_B[-1], actual_M_B[-1])

    def test_to_numeraire_currency(self):
        sigma_t_function = SigmaTFunction(
            PiecewiseConstantDict({
                0: CalibratedSingleExpirySigmaT({UKPOWER_PEAK: 0.5})
            }, extrapolate_left=True),
        )
        underlying = [GBPUSD, EURGBP, UKPOWER_PEAK, UKPOWER_PEAK]
        sigma = [0.2, 0.3, 0.4, 0.4]
        alpha = [0, 0, 0.2, 2.0]
        rho = [
            [1, 0.3, 0.5, -0.5],
            [0.3, 1, 0.1, 0.25],
            [0.5, 0.1, 1, 0.2],
            [-0.5, 0.25, 0.2, 1],
        ]
        mr_procs = MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=alpha, rho=rho)
        vol_corr = VolCorr(mr_procs, sigmaT_funcs=SigmaTFunctionDict({UKPOWER: sigma_t_function}))

        expected_vol_corr = VolCorr(mr_procs.to_numeraire_currency(GBP),
                                    sigmaT_funcs=SigmaTFunctionDict({UKPOWER: sigma_t_function}),
                                    numeraire_ccy=GBP)
        gbp_vol_corr = vol_corr.to_numeraire_currency(GBP)
        assert gbp_vol_corr.mr_processes == expected_vol_corr.mr_processes
        assert gbp_vol_corr.sigmaT_funcs == expected_vol_corr.sigmaT_funcs
        assert gbp_vol_corr.numeraire_ccy == expected_vol_corr.numeraire_ccy

        expected_vol_corr = VolCorr(mr_procs.to_numeraire_currency(EUR),
                                    sigmaT_funcs=SigmaTFunctionDict({UKPOWER: sigma_t_function}),
                                    numeraire_ccy=EUR)
        eur_vol_corr = vol_corr.to_numeraire_currency(EUR)
        assert_almost_equal(eur_vol_corr.mr_processes, expected_vol_corr.mr_processes)
        assert eur_vol_corr.sigmaT_funcs == expected_vol_corr.sigmaT_funcs
        assert eur_vol_corr.numeraire_ccy == expected_vol_corr.numeraire_ccy


class TestSpotVolRatios:

    def setup_method(self):
        self.spot_vol_ratios = spot_vol_ratios
        self.data = self.spot_vol_ratios.lsu_ratios_dict

    def test_from_table(self):
        headers = ["", f"{UKPOWER_PEAK}", f"{NBP}", f"{TTF}"]
        body = [[i + 1, self.data[UKPOWER_PEAK][i], self.data[NBP][i], self.data[TTF][i]] for i in range(12)]
        table = [headers] + body
        svrs_from_table = SpotVolRatios.from_table(table)
        assert self.spot_vol_ratios == svrs_from_table

    def test_from_table_with_filter(self):
        headers = ["", f"{UKPOWER_PEAK}", f"{NBP}", f"{TTF}"]
        body = [[i + 1, self.data[UKPOWER_PEAK][i], self.data[NBP][i], self.data[TTF][i]] for i in range(12)]
        table = [headers] + body
        svrs_from_table = SpotVolRatios.from_table(table, [TTF])
        expected = SpotVolRatios({TTF: [1] * 12})
        assert expected == svrs_from_table

    def test_from_and_to_table(self):
        table = self.spot_vol_ratios.to_table()
        svrs_from_table = SpotVolRatios.from_table(table)
        assert self.spot_vol_ratios == svrs_from_table

    def test_reshape(self):
        reshaped_spot_vol_ratios = self.spot_vol_ratios.reshape([NBP, TTF])
        assert 2 == len(reshaped_spot_vol_ratios.lsu_ratios_dict)
        assert list(range(1, 13)) == reshaped_spot_vol_ratios.idx
        assert self.spot_vol_ratios.lsu_ratios_dict[NBP] == \
                         reshaped_spot_vol_ratios.lsu_ratios_dict[NBP]


class TestDatedSpotVolRatios:

    def setup_method(self):
        self.spot_vol_ratios = spot_vol_ratios
        self.data = self.spot_vol_ratios.lsu_ratios_dict

        self.pricing_date = date(2018, 1, 15)
        self.dated_spot_vol_ratios = DatedSpotVolRatios(self.pricing_date, self.spot_vol_ratios)

    def test_construction(self):
        expected_taus = [
                            time_between(self.pricing_date, date(2018, i, 1)) - DatedSpotVolRatios.tol
                            for i in range(1, 13)
                        ] + [time_between(self.pricing_date, date(2019, 1, 1)) - DatedSpotVolRatios.tol]
        assert_almost_equal(expected_taus, self.dated_spot_vol_ratios._taus)

        expected_ratios = {lsu: self.data[lsu] + [self.data[lsu][0]] for lsu in self.data}
        assert_almost_equal(expected_ratios, self.dated_spot_vol_ratios._ratios)

        # Tol is less than 10 to the minus 5
        assert DatedSpotVolRatios.tol < 1e-5

    def test_call(self):
        for d in range(1, 750):
            call_date = self.pricing_date + timedelta(d)
            T = time_between(self.pricing_date, call_date)
            expected_ratios = {
                lsu: self.data[lsu][call_date.month - 1]
                for lsu in self.data
            }
            actual_ratios = {
                lsu: self.dated_spot_vol_ratios(lsu, T)
                for lsu in self.data
            }
            assert_almost_equal(expected_ratios, actual_ratios)


