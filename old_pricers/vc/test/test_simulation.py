from datetime import date, timedelta
from unittest.mock import Mock

from numpy import mean, exp, sqrt, array, std, var, ones, cov, log, corrcoef
from scipy.integrate import quad

from pytest_quants import assert_almost_equal
from time_period import month

from thorn.core.market.volatility.vc.simulation import VcSimulation, LocalVolSimulationBase
from thorn.core.market.dynamic.vc_mkt_dynamic import VcSimulationCache
from thorn.core.market.volatility.vc.vol_corr import VolCorr
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from underlying.fixtures.underlyings import NBP, UKPOWER_PEAK, TTF
from thorn.core.market.volatility.vc.test.fixtures import mean_rev_procs, vol_corr, proxy_vol_corr

path_count = 35000


class TestSimulation:
    """
    Check the following for a variety of different JumpVolCorr parameterisations of increasing complexity:

    In relation to prices in the Simulation:
    1. The mean is 1
    2. The variance of log prices matches an analytical formula defined in each test case

    Finally test looks at the approximation used for swap prices
    """

    def z_test(self, expected_mean, sample):
        std_err = std(sample) / sqrt(path_count)
        if std_err > 1e-4:
            z = (mean(sample) - expected_mean) / std_err
            assert -3.5 < z < 3.5, z  # some of the distributions are heavily skewed
        else:
            assert_almost_equal(expected_mean / mean(sample), 1, 2)

    def check_moments(self, analytical_variance_fn, *args, **kwargs):
        lsu = NBP
        underlying = [NBP] * len(kwargs['sigma'])
        mod = VolCorr(MeanRevertingProcesses(underlying=underlying, *args, **kwargs))
        sim = VcSimulation(mod, path_count, seed=8743, simulated_lsus=mod.lsus)
        T = 1.55
        for t in [1., 1.5]:
            sim.advance_to(t)
            ln_returns = sim._ln_returns(lsu, T)
            price = exp(ln_returns)
            # Check first moment
            self.z_test(1, price)

            # Check second moment
            simulated_variance = var(log(price))
            if analytical_variance_fn is not None:
                expected_variance = analytical_variance_fn(t, T)
                assert_almost_equal(simulated_variance / expected_variance, 1, atol=0.1)

            estimated_variance = mod.calc_covariance(lsu, lsu, t, T1=T, D1=0)
            assert estimated_variance / simulated_variance < 1.1
            assert estimated_variance / simulated_variance > 0.9

    def test_gbm(self):
        """Dead simple case where the price process is GBM"""
        sigma = array([0.2])
        alpha = array([0])
        rho = array([[1.000000]])

        def analytical_variance_fn(t, T):
            return sigma[0] * sigma[0] * t

        self.check_moments(analytical_variance_fn, sigma=sigma, alpha=alpha, rho=rho)

    def test_ou(self):
        """Introduce a decay, spot is OU process"""
        sigma = array([0.3])
        decay = array([2.])
        rho = array([[1.0]])

        def analytical_variance_fn(t, T):
            return sigma[0] * sigma[0] * exp(-2 * decay[0] * T) / (2 * decay[0]) * (exp(2 * decay[0] * t) - 1)

        self.check_moments(analytical_variance_fn, sigma=sigma, alpha=decay, rho=rho)

    def test_2_correlated_gbm(self):
        """Now curve is driven by two correlated GBM"""
        sigma = array([0.2, 0.1])
        decay = array([0, 0])
        rho = array([[1.0, 0.8],
                     [0.8, 1.0]])

        def analytical_variance_fn(t, T):
            return (sigma[0] * sigma[0] + sigma[1] * sigma[1] + 2 * rho[0][1] * sigma[0] * sigma[1]) * t

        self.check_moments(analytical_variance_fn, sigma=sigma, alpha=decay, rho=rho)

    def test_2_uncorrelated_ou(self):
        """Curve is driven by two uncorrelated OU processess"""
        sigma = array([0.2, 0.1])
        decay = array([0.1, 0.5])
        rho = array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        def analytical_variance_fn(t, T):
            def integrand(t):
                def int_e_d_T_minus_t(d):
                    """Integral of exp(-d*(T-t))"""
                    return exp(-d * (T - t)) / d

                term0 = sigma[0] * sigma[0] * int_e_d_T_minus_t(2 * decay[0])
                term2 = sigma[1] * sigma[1] * int_e_d_T_minus_t(2 * decay[1])
                return term0 + term2

            return integrand(t) - integrand(0)

        self.check_moments(analytical_variance_fn, sigma=sigma, alpha=decay, rho=rho)

    def test_2_correlated_ou(self):
        """Curve is driven by two correlated OU processess"""
        sigma = array([0.2, 0.1])
        decay = array([0.1, 0.5])
        rho = array([[1.0, 0.8],
                     [0.8, 1.0]])

        def analytical_variance_fn(t, T):
            def integrand(t):
                def int_e_d_T_minus_t(d):
                    """Integral of exp(-d*(T-t))"""
                    return exp(-d * (T - t)) / d

                term0 = sigma[0] * sigma[0] * int_e_d_T_minus_t(2 * decay[0])
                term1 = 2 * rho[0][1] * sigma[0] * sigma[1] * int_e_d_T_minus_t(decay[0] + decay[1])
                term2 = sigma[1] * sigma[1] * int_e_d_T_minus_t(2 * decay[1])
                return term0 + term1 + term2

            return integrand(t) - integrand(0)

        self.check_moments(analytical_variance_fn, sigma=sigma, alpha=decay, rho=rho)

    def test_swap(self):
        """
        Check the swap approximation both for Monte Carlo and Moment Generating Function
        Curve is driven by two correlated OU processess plus jumps
        """
        sigma = array([0.2, 0.1])
        decay = array([0.1, 0.5])
        rho = array([[1.0, 0.8],
                     [0.8, 1.0]])
        lsu = NBP
        underlying = [lsu] * 2
        mod = VolCorr(MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=decay, rho=rho))
        sim = VcSimulation(mod, path_count, seed=1234, simulated_lsus=mod.lsus)
        t = 1.2
        T = 1.5
        D = 0.5
        sim.advance_to(t)

        # Check the swap approximation for Monte Carlo
        def integrand(T):
            return sim.get_price(lsu, T)[0]

        swap_price_path_0 = quad(integrand, T, T + D)[0] / D
        approx_swap_price = sim.get_price(lsu, T, D)
        assert_almost_equal(swap_price_path_0 / approx_swap_price[0], 1, atol=1e-2)

        # check martingale
        self.z_test(1., approx_swap_price)

        # Check the estimated variance
        simulated_variance = var(log(approx_swap_price))
        estimated_variance = mod.calc_covariance(lsu, lsu, t, T, D)
        assert estimated_variance / simulated_variance < 1.1
        assert estimated_variance / simulated_variance > 0.9

    def test_swap_martingale(self):
        """Check the swap approximation is a martingale
        """

        npath = 10000

        sigma = array([1.2, 1.1])
        decay = array([10.1, 5.5])
        rho = array([[1.0, 0.1],
                     [0.1, 1.0]])
        lsu = NBP
        underlying = [lsu] * 2
        mod = VolCorr(MeanRevertingProcesses(underlying=underlying, sigma=sigma, alpha=decay, rho=rho))
        sim = VcSimulation(mod, npath, seed=1234, simulated_lsus=mod.lsus)

        t = 1.2
        T1 = 1.2
        T2 = 1.5
        sim.advance_to(t)

        approx_swap_price = sim.get_price(lsu, T1, T2)
        self.z_test(1., approx_swap_price)

    def test_t_0(self):
        lsu = NBP
        mod = VolCorr(MeanRevertingProcesses(underlying=[NBP]))
        sim = VcSimulation(mod, path_count, seed=8743, simulated_lsus=mod.lsus)
        assert_almost_equal(sim.get_price(lsu, 1, 2), ones(path_count))

    def test_multi_lsu(self):
        sim = VcSimulation(vol_corr, 500000, seed=8724, simulated_lsus=vol_corr.lsus)
        lsu1 = NBP
        lsu2 = UKPOWER_PEAK
        t = 5
        T1 = t + 0.01
        T2 = t + 0.02
        sim.advance_to(t)
        ln_returns1 = sim._ln_returns(lsu1, T1)
        ln_returns2 = sim._ln_returns(lsu2, T2)
        cov_mat = cov(ln_returns1, ln_returns2)
        assert_almost_equal(cov_mat[0][0] / vol_corr.calc_covariance(lsu1, lsu1, t, T1, 0), 1, atol=1e-2)
        assert_almost_equal(cov_mat[1][1] / vol_corr.calc_covariance(lsu2, lsu2, t, T2, 0), 1, atol=1e-2)
        assert_almost_equal(cov_mat[0][1] / vol_corr.calc_covariance(lsu1, lsu2, t, T1, 0, T2, 0), 1, atol=1e-2)

    def test_proxy_lsu_normal_caching(self):
        surface = Mock()
        surface.lsu = TTF
        dt = 1.0 / 365.0
        sim1 = LocalVolSimulationBase(
            proxy_vol_corr, 500000, seed=8724, simulated_lsus=proxy_vol_corr.lsus, surface=surface,
        )
        normals = {}
        for i in range(10):
            normals[i] = sim1._normal(dt)
            sim1.t += dt

        sim2 = LocalVolSimulationBase(
            proxy_vol_corr, 500000, seed=8724, simulated_lsus=proxy_vol_corr.lsus, surface=surface,
            noise_cache=sim1.noise_cache,
        )

        for i in range(10):
            assert_almost_equal(normals[i], sim2._normal(dt))
            sim2.t += dt

    def test_theta_random_numbers(self):
        paths = 20000
        seed = 8743
        pricing_date = date(2018, 1, 10)
        monte_carlo_start_date = date(2018, 1, 1)
        tp = month(2018, 3)
        lsu = UKPOWER_PEAK
        original_sim_cache = VcSimulationCache(monte_carlo_start_date, paths, seed, vol_corr, simulated_lsus=vol_corr.lsus)
        sim_cache = VcSimulationCache(pricing_date, paths, seed, vol_corr, simulated_lsus=vol_corr.lsus)
        sim_cache_adjusted = VcSimulationCache(pricing_date, paths, seed, vol_corr, simulated_lsus=vol_corr.lsus,
                                               monte_carlo_start_date=monte_carlo_start_date)

        # check distribution are similar
        unadjusted_price_moves = sim_cache.get_price_moves(lsu, 30, tp)
        adjusted_price_moves = sim_cache_adjusted.get_price_moves(lsu, 30, tp)
        assert_almost_equal(unadjusted_price_moves.mean(), adjusted_price_moves.mean(), atol=1e-4)
        assert_almost_equal(std(unadjusted_price_moves), std(adjusted_price_moves), atol=1e-3)

        # check more correlation to adjusted one
        original_price_moves = original_sim_cache.get_price_moves(lsu, 39, tp)
        unadj_corr = corrcoef(original_price_moves, unadjusted_price_moves, rowvar=0)[0, 1]
        adj_corr = corrcoef(original_price_moves, adjusted_price_moves, rowvar=0)[0, 1]
        assert_almost_equal(unadj_corr, 0.87001193)
        assert_almost_equal(adj_corr, 0.88032319)

        # check if pricing_date before original (eg backtest)
        sim_cache2 = VcSimulationCache(monte_carlo_start_date - timedelta(10), paths, seed, vol_corr,
                                       simulated_lsus=vol_corr.lsus, monte_carlo_start_date=monte_carlo_start_date)
        assert_almost_equal(0, sim_cache2.max_t_in_days)

    def test_simulated_lsus(self):
        sim_path_count = 20000
        seed = 8742
        pricing_date = date(2018, 1, 1)
        lsu = UKPOWER_PEAK
        sim_cache_all = VcSimulationCache(pricing_date, sim_path_count, seed, vol_corr, simulated_lsus=vol_corr.lsus)
        sim_cache_lsu = VcSimulationCache(pricing_date, sim_path_count, seed, vol_corr, simulated_lsus=[lsu])
        sim_all = sim_cache_all.simulation
        sim_lsu = sim_cache_lsu.simulation
        sim_lsu_indices, lsu_to_simulated_indices = vol_corr.lsu_mapping([lsu])
        sim_all_indices, _ = vol_corr.lsu_mapping(mean_rev_procs.lsus)
        assert sim_lsu_indices == [2, 3]
        assert lsu_to_simulated_indices == {lsu: [0, 1]}

        t = 5
        T = t + 0.5

        assert sim_all.xi.shape == (4, sim_path_count / 2)
        assert sim_lsu.xi.shape == (2, sim_path_count / 2)
        sim_all.advance_to(t)
        sim_lsu.advance_to(t)

        inverse_sim_all_indices = [sim_all_indices.index(i) for i in range(len(sim_all_indices))]

        # diff between sim_all and sim_lsu should only be due to noise
        assert_almost_equal(1.25679716, std(sim_all.xi[inverse_sim_all_indices][sim_lsu_indices]))
        assert_almost_equal(1.27121469, std(sim_lsu.xi))
        assert_almost_equal(-0.00181632, sim_all.xi[inverse_sim_all_indices][sim_lsu_indices].mean())
        assert_almost_equal(0.00915887, sim_lsu.xi.mean())
        assert_almost_equal(0.99951787, sim_all.get_price(lsu, T).mean())
        assert_almost_equal(1.00084659, sim_lsu.get_price(lsu, T).mean())

    def test_antithetics(self):
        # Even path count
        sim = VcSimulation(VolCorr(mean_rev_procs), 20, seed=8743, simulated_lsus=mean_rev_procs.lsus)
        t = 5
        T = t + 0.5
        D = T + 0.5
        sim.advance_to(t)
        ln_returns_with_drift = sim._ln_returns_with_drift(NBP, T, D)
        zeros = ln_returns_with_drift[:10] + ln_returns_with_drift[10:]
        assert_almost_equal(min(zeros), 0)
        assert_almost_equal(max(zeros), 0)

        # Odd path count
        sim = VcSimulation(VolCorr(mean_rev_procs), 19, seed=8743, simulated_lsus=mean_rev_procs.lsus)
        ln_returns_with_drift = sim._ln_returns_with_drift(NBP, T, D)
        zeros = ln_returns_with_drift[:9] + ln_returns_with_drift[10:]
        assert_almost_equal(min(zeros), 0)
        assert_almost_equal(max(zeros), 0)

    def test_normal_price(self):
        sim = VcSimulation(VolCorr(mean_rev_procs), 1000, seed=8743, simulated_lsus=mean_rev_procs.lsus)
        sim.advance_to(1.1)
        normal_prices = sim.get_normal_price(NBP, 1, 1/365)
        log_normal_prices = sim.get_price(NBP, 1, 1 / 365)
        assert_almost_equal(min(exp(normal_prices)/log_normal_prices),
                               max(exp(normal_prices)/log_normal_prices))


