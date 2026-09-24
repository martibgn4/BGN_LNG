"""Vol over the trade's horizon, from a vol to the option expiry.

An ATM implied vol is an average over [today, option expiry]. The trade needs
the average over [today, t_d], and for every leg but M those windows differ:
M+1's option includes the month when M+1 is the front contract, the most
volatile part of its life, which the trade never lives through.

With a time-homogeneous Samuelson shape phi(tau) (instantaneous vol as a
function of time to fixing, calibrated as realised vol by days-to-expiry
bucket), the implied total variance is spread over time in proportion to
phi^2, and the horizon share is kept:

    sigma_h^2 h = sigma_imp^2 T_opt * W(T_fix - h, T_fix) / W(T_fix - T_opt, T_fix)
    W(a, b)     = integral of phi(tau)^2 over tau in [a, b]

The level of phi cancels; only its shape matters. For the M leg (h ~ T_opt)
this returns the implied vol; for M+1 it strips the front-month period out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def horizon_vol(vol_implied: float, t_opt_days: float, t_fix_days: float, h_days: float,
                profile: pd.DataFrame) -> float:
    """Annualised vol over [0, h] for a contract fixing in t_fix days.

    `profile` is one hub's rows of vol_profile.csv. Days throughout: the
    year fractions cancel in the ratio.
    """
    if h_days <= 0 or t_opt_days <= 0:
        return vol_implied
    t_opt_days = min(t_opt_days, t_fix_days)
    num = _w(profile, max(t_fix_days - h_days, 0.0), t_fix_days)
    den = _w(profile, t_fix_days - t_opt_days, t_fix_days)
    if not den > 0:
        # An empty or zeroed profile would otherwise give 0/0 = NaN vol, silently.
        raise ValueError("no usable vol profile for this hub - check vol_profile.csv, "
                         "or price with vol_horizon: implied")
    return float(vol_implied * np.sqrt(t_opt_days / h_days * num / den))


def profile_vol(profile: pd.DataFrame, t_days: float) -> float:
    """Realised-curve vol of a contract from today to its expiry in t_days:
    the root mean of phi^2 over tau in [0, t]."""
    if t_days <= 0:
        return float(profile.sort_values("tau_lo_days")["realised_vol"].iloc[0])
    return float(np.sqrt(_w(profile, 0.0, t_days) / t_days))


def _w(profile: pd.DataFrame, a: float, b: float) -> float:
    """Integral of phi^2 over tau in [a, b]; flat beyond the last bucket."""
    lo = profile["tau_lo_days"].to_numpy(float)
    hi = profile["tau_hi_days"].to_numpy(float)
    var = profile["realised_vol"].to_numpy(float) ** 2
    hi = np.where(np.isfinite(hi), hi, np.inf)
    overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return float((overlap * var).sum())
