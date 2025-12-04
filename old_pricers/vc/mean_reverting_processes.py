import typing
from copy import deepcopy

import math
import numpy as np

from general_utils import memoize_property
from numpy.linalg import LinAlgError
from time_period import BASE
from underlying import FX, Underlying
from value_object import Attr, List, Object, un_Collection


def _get_fxs_and_fx_vol_pivot_currencies(lsus):
    '''
    Returns:

    - the list of FX underlyings
    - the currencies present in all the FX pairs:
        - There should be only one except if there is only one FX
        - If there is more than one then the correl matrix is either arbitrable or non positive => Raise
    '''
    fx_list = [lsu for lsu in lsus if isinstance(lsu, FX)]
    pivot_ccys = set.intersection(*[{u.currency, u.foreign_currency} for u in fx_list]) if fx_list else set()

    assert not (len(fx_list) > 0 and len(pivot_ccys) == 0), \
        "Did not find a common currency in all the FX couples (%s)" % str(fx_list)
    assert not (len(fx_list) > 1 and len(pivot_ccys) > 1), \
        "Found more than one common currency in all the FX couples (%s). Correlation matrix corrupted." % \
        str(fx_list)

    return fx_list, pivot_ccys


def _invert_fx(lsus, rho, fx):
    '''
    Update the lsu_to_indices and rho to refer to fx.inverse.
    '''
    # Create new FX
    new_fx = fx.inverse
    lsus = [new_fx if u == fx else u for u in lsus]

    # Get FX factor
    indices = [i for i, u in enumerate(lsus) if u == new_fx]
    assert len(indices) == 1, f"Only one index supported: {indices}"
    index = indices[0]

    rho[index, :] = -rho[index, :]
    rho[:, index] = -rho[:, index]
    return lsus, rho


def _change_fx_currency(lsus, rho, sigma, fx, numeraire_ccy):
    new_fx = FX.get(currency=numeraire_ccy, foreign_currency=fx.foreign_currency)

    orig_indices = [i for i, u in enumerate(lsus) if u == fx]
    lsus = [new_fx if u == fx else u for u in lsus]

    # c1=fx.foreign_ccy, c2=fx.currency, c3 = numeraire_ccy
    assert len(orig_indices) == 1, f"Only one index supported ({orig_indices})"
    c1c2 = orig_indices[0]
    sigma_c1c2 = sigma[c1c2]
    cross_fx = FX.get(currency=numeraire_ccy, foreign_currency=fx.currency)
    c2c3 = [i for i, u in enumerate(lsus) if u == cross_fx][0]
    sigma_c2c3 = sigma[c2c3]
    rho_c1c2_c2c3 = rho[c1c2, c2c3]
    sigma_c1c3 = math.sqrt(
        sigma_c1c2 * sigma_c1c2 +
        sigma_c2c3 * sigma_c2c3 +
        2 * rho_c1c2_c2c3 * sigma_c1c2 * sigma_c2c3
    )
    c1c3 = c1c2
    sigma[c1c3] = sigma_c1c3
    old_rho = deepcopy(rho)

    for m in range(len(sigma)):
        rho_c1c2_m = old_rho[c1c2, m]
        rho_c2c3_m = old_rho[c2c3, m]
        rho_c1c3_m = (rho_c1c2_m * sigma_c1c2 + rho_c2c3_m * sigma_c2c3) / sigma_c1c3
        if m != c1c3:
            rho[c1c3, m] = rho_c1c3_m
            rho[m, c1c3] = rho_c1c3_m
    return lsus, rho, sigma


class MeanRevertingProcesses(Object):
    """Describes MR processes for a collection of underlyings.

    Used by VolCorr, however there may be other data used to completely determine the paths generated -
    e.g. SpotVolRatiosData """
    underlying: List((Underlying, List(Underlying))) = Attr([])
    sigma: List(float) = Attr([0])
    alpha: List(float) = Attr([0])
    rho: List(List(float)) = Attr([[1]])

    def post_init(self):
        # Check underlyings are valid
        if len(self.underlying) > 0:
            if len(self.underlying) != len(self.sigma):
                raise AssertionError(f"Len(underlying) != len(sigma), (len({self.underlying}) != len({self.sigma}))")
            if len(self.underlying) != len(self.alpha):
                raise AssertionError(f"Len(underlying) != len(alpha), (len({self.underlying}) != len({self.alpha}))")

        # Check rho is valid
        _rho = np.array(self.rho)
        tol = 1e-6
        if not (abs(_rho.T - _rho) < tol).all():
            raise AssertionError("Rho not symmetric")
        if not (abs(np.diag(_rho) - 1) < tol).all():
            raise AssertionError("diag(rho) != 1")
        if len(self.underlying) > 1:
            try:
                np.linalg.cholesky(_rho)
            except LinAlgError as e:
                raise LinAlgError(f"{e} --> Matrix (_rho):\n{_rho}") from e

    @memoize_property
    def lsu_to_indices(self):
        lsu_to_indices = {}  # Don't use a defaultdict so that we'll get a KeyError rather than an empty list
        _underlying = [np.atleast_1d(u) for u in self.underlying]
        for i, underlying_list in enumerate(_underlying):
            for u in underlying_list:
                lsu_to_indices.setdefault(u, []).append(i)

        return {
            u: sorted(indices, key=lambda i: self.alpha[i])
            for u, indices in lsu_to_indices.items()
        }

    @memoize_property
    def lsus(self):
        return sorted(self.lsu_to_indices.keys())

    @memoize_property
    def lsus_by_base_underlying(self):
        _lsus_by_base_underlying = {}
        for lsu in self.lsus:
            _lsus_by_base_underlying.setdefault(lsu.base_underlying, []).append(lsu)
        return _lsus_by_base_underlying

    def get_intersecting_lsus(self, lsu: Underlying) -> typing.List[Underlying]:
        """ The modelled LoadShapedUnderlyings required to calculate the volatility of the requested LSU """
        if lsu.load_shape == BASE:  # Optimisation
            return self.lsus_by_base_underlying.get(lsu.vol_model_underlying, ())
        return [
            _lsu for _lsu in self.lsus_by_base_underlying.get(lsu.base_underlying.vol_model_underlying, ())
            if _lsu.load_shape.intersects(lsu.load_shape)
        ]

    @property
    def factor_count(self):
        return len(self.sigma)

    @classmethod
    def from_table(cls, data) -> 'MeanRevertingProcesses':
        """Convert a list of rows to a vol corr instance.
        underlying_filter: list[Underlyings]
            used to filter the table to the set of underlyings required. If empty or None no filtering
            will be applied
        """
        list_of_rows = list(data)  # handle iterators

        start_idx_of_first_underlying = 4
        if len(list_of_rows) < start_idx_of_first_underlying:
            raise AssertionError(
                f"Insufficent Data: not enough rows to create a volcorr representation"
            )

        def crop_list(row_or_column):
            # Only parse until first empty column
            valid_pos = [
                i for i, field in enumerate(row_or_column[start_idx_of_first_underlying:])
                if field not in (None, "", [])
            ]
            max_size = max([i + 1 for i in valid_pos if max(i - 1, 0) in valid_pos], default=0)
            return row_or_column[start_idx_of_first_underlying:start_idx_of_first_underlying + max_size]

        # Split data into header rows / columns and Rho matrix,
        #  handle various combinations and "long" and "short" rows/cols
        data = [crop_list(row) for row in list_of_rows]
        lsus, factors, sigmas, alphas, *rhos = data
        lsu_column, factor_column, sigma_column, alpha_column = [
            crop_list([row[i] if i < len(row) else None for row in list_of_rows])
            for i in range(start_idx_of_first_underlying)
        ]
        rhos = [row for row in rhos if row]

        # Check consistency between row and column headings
        if not len(lsus) == len(factors) == len(sigmas) == len(alphas):
            raise AssertionError(
                f"Length mismatch: Underlyings ({len(lsus)}) != Factors ({len(factors)}) != "
                f" Sigmas ({len(sigmas)}) != Alphas ({len(alphas)})"
            )

        if not len(lsu_column) == len(factor_column) == len(sigma_column) == len(alpha_column):
            raise AssertionError(
                f"Column Length mismatch: Underlyings ({len(lsu_column)}) != Factors ({len(factor_column)}) != "
                f" Sigmas ({len(sigma_column)}) != Alphas ({len(alpha_column)})"
            )

        for i, _ in enumerate(lsus):
            assert lsu_column[i] == lsus[i], f"Shape mismatch at {i}: {lsu_column[i]} != {lsus[i]}"
            assert factor_column[i] == factors[i], f"Shape mismatch at {i}: {factor_column[i]} != {factors[i]}"
            assert sigma_column[i] == sigmas[i], f"Shape mismatch at {i}: {sigma_column[i]} != {sigmas[i]}"
            assert alpha_column[i] == alphas[i], f"Shape mismatch at {i}: {alpha_column[i]} != {alphas[i]}"

        # Split underlyings, and return built object
        load_shaped_underlyings = [ul if len(ul.split(";")) == 1 else ul.split(";") for ul in lsus]
        MeanRevertingProcesses._check_proxied_factors(load_shaped_underlyings, factors)
        return cls(underlying=load_shaped_underlyings, sigma=sigmas, alpha=alphas, rho=rhos)

    @staticmethod
    def _check_proxied_factors(underlyings, factors):
        ul_list = [np.atleast_1d(u) for u in underlyings]
        ul_factor_tuples = [(uls, f) for uls, f in zip(ul_list, factors)]
        for i, (underlyings1, f1) in enumerate(ul_factor_tuples[:-1]):
            for (underlyings2, f2) in ul_factor_tuples[i + 1:]:
                if any(ul in underlyings2 for ul in underlyings1) and f1 == f2:
                    ul = [ul for ul in underlyings1 if ul in underlyings2]
                    raise AssertionError(f"Underlying(s) {ul} has multiple entries for factor {f1}")

    def to_table(self):
        """Build the volcorr table showing Sigma, Alpha and Rho for each underlying."""
        heading = np.array([['', '', '', 'Underlying'],
                            ['', '', '', 'Factor Number'],
                            ['', '', '', 'Sigma'],
                            ['Underlying', 'Factor Number', 'Sigma', 'Alpha']])

        vol_corr_len = len(self.sigma)
        factors = [''] * vol_corr_len
        lsu_sets = [set() for _ in range(vol_corr_len)]
        for lsu in self.lsu_to_indices:
            for factor, index in enumerate(self.lsu_to_indices[lsu]):
                lsu_sets[index].add(lsu)
                factors[index] = factor

        # Ensure the order of any proxied underlyings in as per the original argument
        original_underlying_sets = [set(np.atleast_1d(u)) for u in self.underlying]
        lsu_lists = [self.underlying[original_underlying_sets.index(lsu_set)] for lsu_set in lsu_sets]
        lsu_names = [";".join([str(u) for u in np.atleast_1d(lsus)]) for lsus in lsu_lists]

        if self.sigma:
            vol = np.array([lsu_names, factors, self.sigma, self.alpha])
            a = np.hstack((heading, vol))
            b = np.hstack((vol.transpose(), self.rho))
            output = np.vstack((a, b))
        else:
            output = heading
        return output.tolist()

    @staticmethod
    def _should_include(underlying_or_list, underlyings_to_include):
        if isinstance(underlying_or_list, Underlying):
            return underlying_or_list in underlyings_to_include
        else:
            return any(MeanRevertingProcesses._should_include(u, underlyings_to_include) for u in underlying_or_list)

    def override_sigma(self, underlying, factor, override):
        if underlying not in self.lsu_to_indices:
            raise AssertionError(f"Cannot override an underlying ({underlying}) not already in VolCorr.")
        assert 0 <= factor <= len(self.lsu_to_indices[underlying]), \
            f"Cannot override factor {factor} of {underlying}."
        new_sigma = un_Collection(self.sigma)
        new_sigma[self.lsu_to_indices[underlying][factor]] = override
        return self.clone(sigma=new_sigma)

    def reshape(self, underlyings, ir_pivot_ccy=None):
        """
        Given a list of underlyings return a new reduced VolCorr object with
        only those underlyings plus any FXs as required
        """
        if ir_pivot_ccy is not None:
            required_fxs = [
                FX.get(ir_pivot_ccy, u.currency) for u in underlyings
                if ir_pivot_ccy != u.currency
            ]
            underlyings = required_fxs + [fx.inverse for fx in required_fxs] + underlyings

        idx = {i for i, underlying_or_list in enumerate(self.underlying) if
               self._should_include(underlying_or_list, underlyings)}

        def reshape_ul(ul_or_list):
            if isinstance(ul_or_list, Underlying):
                return ul_or_list
            else:
                uls = [_ul for _ul in ul_or_list if _ul in underlyings]
                if len(uls) == 1:
                    return uls[0]
                return uls

        kwargs = {
            'underlying': [reshape_ul(ul) for i, ul in enumerate(self.underlying) if i in idx],
            'sigma': [s for i, s in enumerate(self.sigma) if i in idx],
            'alpha': [a for i, a in enumerate(self.alpha) if i in idx],
            'rho': [[r for i, r in enumerate(ul_rho) if i in idx]
                    for i, ul_rho in enumerate(self.rho) if i in idx]
        }
        return self.clone(**kwargs)

    @memoize_property
    def fx_and_pivot_ccys(self):
        return _get_fxs_and_fx_vol_pivot_currencies(self.lsus)

    def to_numeraire_currency(self, numeraire_ccy):
        '''
        Returns a new object with numeraire_ccy as a base currency
        This means that all the FXs described in the calc_adaptor have the shape: FX(numeraire_ccy, XXX)

        For the moment we assume that we have either XXX/REF of REF/XXX in the price processes, for all XXX
        If not we raise (assertion error)
        Next step: be able to re-base the matrix even if OLDREF != NEWREF
        '''
        fx_list, pivot_ccys = self.fx_and_pivot_ccys

        # Create a new VolCorr with the re-based currencies (where required)
        lsus = self.underlying.copy()
        rho = np.array(self.rho)
        sigma = np.array(self.sigma)

        # Stage1 invert all fx where the foreign currency is the numeraire_ccy
        for fx in fx_list:
            if numeraire_ccy == fx.foreign_currency:
                lsus, rho = _invert_fx(lsus, rho, fx)

        # Stage2 find all currency where the denominator is not the numeraire ccy and change to the numeraire ccy
        fx_list = [lsu for lsu in lsus if isinstance(lsu, FX)]
        for fx in fx_list:
            if numeraire_ccy != fx.currency:
                potential_new_fx = FX.get(currency=numeraire_ccy, foreign_currency=fx.foreign_currency)
                if potential_new_fx in lsus:
                    lsus, rho = _invert_fx(lsus, rho, fx)
                    fx_to_replace = fx.inverse
                else:
                    fx_to_replace = fx
                lsus, rho, sigma = _change_fx_currency(lsus, rho, sigma, fx_to_replace, numeraire_ccy)

        return self.clone(underlying=lsus, rho=rho, sigma=sigma)


