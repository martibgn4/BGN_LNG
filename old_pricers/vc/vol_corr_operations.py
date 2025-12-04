from value_object import un_Collection

from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses


def max_factor_num(vol_corr):
    return max(len(ind) for ind in vol_corr.lsu_to_indices.values()) if vol_corr.lsu_to_indices else 0


def mean_rev_2f(mean_rev_procs):
    """ Creates a Two Factor vol_corr from a general form """
    _valid_factors = ("", "Factor Number", "0", "1", 0, 1)
    original_data = mean_rev_procs.to_table()
    # Check dat ais as expected, then
    assert "Factor" in original_data[1][3], "Expected 'Factor Number' in Cell(1, 3), found: {vc2f_data[1][3]}"
    # Filter out the rows for factors > 2
    vc2f_data = [row for row in original_data if row[1] in _valid_factors]

    # Filter out the columns for factors > 2
    cols_to_delete = [i for i, factor_num in enumerate(vc2f_data[1]) if factor_num not in _valid_factors]
    vc2f_data = [
        [x for i, x in enumerate(row) if i not in cols_to_delete]
        for row in vc2f_data
    ]
    return MeanRevertingProcesses.from_table(vc2f_data)


def scale_sigma(mean_rev_procs, lsu, factor):
    scaled_sigma = un_Collection(mean_rev_procs.sigma)
    for idx in mean_rev_procs.lsu_to_indices[lsu]:
        scaled_sigma[idx] *= factor
    return mean_rev_procs.clone(sigma=scaled_sigma)


def bump_down_correlations(mean_rev_procs, lsu1, lsu2, shift):
    # correlation_cap = 0.9999 in the covariance_calculator.py
    proxy_threshold = 0.9998
    bumped_rho = un_Collection(mean_rev_procs.rho)

    def _shift(eq_cls1, eq_cls2):
        # set1 and set2 are equivalent classes ie they re either the same or disjoint
        # if the same: no bumping, else all clear
        if eq_cls1 == eq_cls2:
            return
        for i1 in eq_cls1:
            for i2 in eq_cls2:
                bumped_rho[i1][i2] -= shift
                bumped_rho[i2][i1] -= shift

    def _eq_cls(idx):
        # idx along with its proxies (ie correl = 0.9999)
        return {k for k, rho in enumerate(bumped_rho[idx]) if rho > proxy_threshold}

    for i in mean_rev_procs.lsu_to_indices[lsu1]:
        i_eq_cls = _eq_cls(i)
        for j in mean_rev_procs.lsu_to_indices[lsu2]:
            j_eq_cls = _eq_cls(j)
            _shift(i_eq_cls, j_eq_cls)

    return mean_rev_procs.clone(rho=bumped_rho)


def bump_up_vs_sigma(mean_rev_procs, lsu, shift):
    vs_factor = 2
    indices = mean_rev_procs.lsu_to_indices[lsu]
    if len(indices) <= vs_factor:
        return mean_rev_procs
    prev_sigma = mean_rev_procs.sigma[indices[vs_factor]]
    return mean_rev_procs.override_sigma(lsu, vs_factor, prev_sigma * (1 + shift))


def correl_shift(mean_rev_procs, shift, numeraire_ccy):
    """shift from original to 100% correlation matrix, eg. shift of 0.5 (50%): 0.2 becomes 0.6"""
    num_ccy_mrp = mean_rev_procs.to_numeraire_currency(numeraire_ccy)
    shifter = lambda old_value: old_value * (1 - shift) + shift
    shifted_rho = [list(map(shifter, row)) for row in num_ccy_mrp.rho]
    return num_ccy_mrp.clone(rho=shifted_rho)


