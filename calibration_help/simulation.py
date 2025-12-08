from abc import abstractmethod
from copy import deepcopy

import numpy as np
from numpy import einsum, zeros
from quantity import DAYS_PER_YEAR


class Simulation:
    def __init__(self, vol_corr, path_count, seed, simulated_lsus, start_time=0):
        self.path_count = path_count
        self.half_path_count = (self.path_count + 1) // 2
        self.path_count_is_even = (self.half_path_count * 2 == self.path_count)
        self.negated_path_count = self.half_path_count - (0 if self.path_count_is_even else 1)
        self.seed = seed
        self.vol_corr = vol_corr
        self.simulated_lsus = simulated_lsus
        self.simulated_indices, self.lsu_to_simulated_indices = self.vol_corr.lsu_mapping(self.simulated_lsus)
        self.t = start_time  # observation time in years
        self.random_state = np.random.default_rng(seed=seed)

    def __eq__(self, other):
        return (
                isinstance(other, type(self))
                and self.path_count == other.path_count
                and self.half_path_count == other.half_path_count
                and self.path_count_is_even == other.path_count_is_even
                and self.negated_path_count == other.negated_path_count
                and self.seed == other.seed
                and self.vol_corr == other.vol_corr
                and self.simulated_lsus == other.simulated_lsus
                and self.simulated_indices == other.simulated_indices
                and self.lsu_to_simulated_indices == other.lsu_to_simulated_indices
                and self.t == other.t
        )

    def _get_antithetics(self, positive):
        negative = np.negative(positive)[:self.negated_path_count]
        return np.append(positive, negative, axis=-1)

    @abstractmethod
    def create_memento(self):
        """ create a cache item """

    @abstractmethod
    def set_memento(self, memento):
        """ retrieve a cache item """

    @abstractmethod
    def _brownian_increments(self, dt):
        """ A brownian increment for a time-step of size dt """

    def standard_normal(self):
        return self.random_state.normal(0, 1, size=(self.vol_corr.factor_count, self.half_path_count))

    @staticmethod
    def t_in_days(t):
        return round(t * DAYS_PER_YEAR)  # Use round as int truncates towards zero


class VcSimulation(Simulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xi = zeros((len(self.simulated_indices), self.half_path_count))
        self._cholesky_cache = {}

    def __eq__(self, other):
        return super().__eq__(other) and np.array_equal(self.xi, other.xi, equal_nan=True)

    def _ln_returns_with_drift(self, lsu, T, D):
        lsu_indices = self.lsu_to_simulated_indices[lsu]
        X_B = einsum('i,ip->p', self.vol_corr.M_B(lsu, self.t, T, D), self.xi[lsu_indices])
        return self._get_antithetics(X_B)

    def _ln_returns(self, lsu, T, D=0):
        X_B = self._ln_returns_with_drift(lsu, T, D)
        D_B = -0.5 * self.vol_corr.calc_covariance(lsu, lsu, t=self.t, T1=T, D1=D)
        return X_B + D_B

    def create_memento(self):
        return deepcopy((self.xi, self.t))

    def set_memento(self, memento):
        self.xi, self.t = memento

    def _brownian_increments(self, dt):
        normal = self.standard_normal()
        return np.dot(self._cholesky_decomposition(dt), normal)

    def _cholesky_decomposition(self, dt):
        key = self.t_in_days(dt)
        try:
            return self._cholesky_cache[key]
        except KeyError:
            _value = self._cholesky_cache[key] = np.linalg.cholesky(self.vol_corr.C(dt))
            return _value

    def advance_to(self, t, in_the_future=True):
        # Advance state variables for Brownians
        delta_t = t - self.t
        assert delta_t > 0, f"Time increment ({delta_t}) should be > 0"
        xi_of_delta_t = self._brownian_increments(delta_t)[self.simulated_indices]
        if in_the_future:
            self.xi = np.exp(-self.vol_corr.alpha * delta_t)[self.simulated_indices, None] * self.xi + xi_of_delta_t
        self.t = t

    def get_price(self, lsu, T, D=0, fx_lsu=None):
        ln_returns = self._ln_returns(lsu, T, D)
        fx_drift = 0 if not fx_lsu else self.vol_corr.calc_covariance(lsu, fx_lsu, self.t, T, D)
        return np.exp(ln_returns) / np.exp(fx_drift)

    def get_normal_price(self, lsu, T, D=0, fx_lsu=None):
        return self._ln_returns_with_drift(lsu, T, D)


class LvcNoiseCache(dict):

    @property
    def max_t_in_days(self):
        return max(self, default=None)


class LvcSimulation(Simulation):
    """ Intermediate layer that takes care of xvs for lvc simulations"""

    def __init__(self, *args, surface=None, noise_cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface = surface
        self.noise_cache = LvcNoiseCache() if noise_cache is None else noise_cache
        if not self.noise_cache:  # Initialise cache with random state for t_0
            self.noise_cache[self.t_in_days(self.t)] = (None, self.random_state.bit_generator.state)

        self.G = np.zeros(self.path_count)  # G = log(F / K)
        self._cholesky_decomposition = np.linalg.cholesky(self.vol_corr.rho)

        # Step parameters
        self.dt = 1.0 / DAYS_PER_YEAR
        self.dt_tol = self.dt * 0.1  # Tolerance to use when compating simulation_time (t)
        self.coarse_step_threshold = 2.0 * 30 / DAYS_PER_YEAR  # far-from-expiry threshold
        self.coarse_step = 7.0 / DAYS_PER_YEAR  # weekly time step when far-from-expiry

    def clone(self, surface=None):
        new = self.__class__(
            self.vol_corr, self.path_count, self.seed, self.simulated_lsus, start_time=self.t,
            surface=surface or self.surface,
            noise_cache=self.noise_cache,
        )
        new.random_state.bit_generator.state = self.random_state.bit_generator.state
        new.G = deepcopy(self.G)
        return new

    def create_memento(self):
        _normal = self.standard_normal()
        return (_normal, self.random_state.bit_generator.state)

    def set_memento(self, memento):
        _, random_state = memento
        self.random_state.bit_generator.state = random_state

    def _normal(self, dt):
        """ Standard normal, averaged over the interval dt (one or many days) """
        t_in_days = self.t_in_days(self.t)
        target_t_in_days = self.t_in_days(self.t + dt)
        normals = []
        while t_in_days < target_t_in_days:
            t_in_days += 1
            memento = self.noise_cache.get(t_in_days)
            if memento is None:
                self.set_memento(self.noise_cache.get(t_in_days - 1))
                memento = self.noise_cache[t_in_days] = self.create_memento()
            normals.append(memento[0])

        assert normals, f"Cannot call normal with 0 dt: {dt}, {dt * DAYS_PER_YEAR}, {self.t}"
        return sum(normals) / np.sqrt(len(normals))

    def _brownian_increments(self, dt):
        normal = self._normal(dt)
        return np.dot(self._cholesky_decomposition * np.sqrt(dt), normal)


class LocalVolSimulationBase(LvcSimulation):
    '''
    Local Vol Corr simulation

    Uses Brownians class as is, these produce a set of correlated increments that are suitable for use with the LVC
    in particular in both cases dW_1(t) dW_2(t) = \rho dt

    Local Vol Function that works directly from the implied vol surface

    Uses with differentials of sigma with respect to T and K

    see http://www.frouah.com/finance%20notes/Dupire%20Local%20Volatility.pdf, page 1, eq. (3b),
    note that this derivation is wrong, and needs correcting as per implementation below

    Note: innovations are performed in log-space
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dstrike = 1e-4
        assert self.surface is not None, f"{self.__class__.__name__} requires a surface"
        self.simulated_indices = [
            self.simulated_indices[i]
            for i in self.lsu_to_simulated_indices[self.surface.lsu]
        ]

    @property
    @abstractmethod
    def _coordinates(self):
        ''' Implemented in derived classes '''

    @abstractmethod
    def _local_vol_function(self, T, K, dt):
        ''' Implemented in derived classes '''

    def _advance(self, dt):
        """ Advance one step of the Euler discretization """
        # Historical simulation
        if self.t < -self.dt_tol:  # t < 0 c.f. "in_the_future"
            past_dt = min(dt, -self.t)
            _ = self._brownian_increments(past_dt)
            dt -= past_dt
            self.t += past_dt

        # Forward simulation - if required
        if dt > self.dt_tol:
            dW = self._brownian_increments(dt)
            lsu_dW = dW[self.simulated_indices]
            P = self.vol_corr.calc_P(self.surface.lsu, self.surface.T - self.t, self.surface.D)
            brownian_increment = np.dot(P, lsu_dW)

            # Calculate the log-price move using the local_vol, correlated innovation and ito-correction
            local_vol = self._local_vol_function(self.t, self._coordinates, dt)
            dG = local_vol * self._get_antithetics(brownian_increment) - 0.5 * local_vol * local_vol * dt

            # Increment the log prices and time
            self.G += dG
            self.t += dt

    def advance_to(self, T):
        while self.t < T - self.dt_tol:
            time_left = T - self.t
            if time_left < self.coarse_step_threshold + self.dt_tol:
                self._advance(self.dt)
            else:
                self._advance(min(time_left, self.coarse_step))

    def get_price_moves(self, T):
        self.advance_to(T)
        return np.exp(self.G)


