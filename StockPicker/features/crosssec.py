"""
============================================================
features/crosssec.py  --  Cross-sectional standardisation
============================================================
Stock selection is a *relative* problem: we never ask "will Apple
go up", we ask "will Apple beat the other 499 names this week".
Every raw signal therefore has to be converted into a rank within
the cross-section on each date before it can be compared or blended.

Three things happen here, in order, and the order matters:

1. **Winsorise**  -- clip extreme tails. A single name with a
   1,000% trailing return would otherwise dominate a z-score and
   turn a diversified signal into a one-stock bet.
2. **Neutralise** -- optionally demean within sector. Without this,
   a "momentum" score in 2020 is mostly a bet on being in Tech;
   we want the name that beat *its own sector*, because the sector
   bet is a separate decision handled by portfolio construction.
3. **Standardise** -- convert to a z-score or a uniform rank so
   signals measured in different units (a P/E, a return, a
   sentiment score) can be averaged on one scale.

Rank-based standardisation is the default: it is robust to the
fat tails and regime-dependent dispersion that wreck raw z-scores
in equity data.
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, pct: float = 0.02) -> pd.Series:
    """Clip a series to its [pct, 1-pct] quantiles."""
    if s.notna().sum() < 5 or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score; returns zeros if the cross-section is degenerate."""
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def rank_normal(s: pd.Series) -> pd.Series:
    """Map to uniform ranks in (0,1), then to a standard normal.

    This is the workhorse transform. It makes the signal distribution
    identical on every date, so a blend does not accidentally weight
    high-dispersion dates more heavily, and it is immune to outliers.
    """
    valid = s.notna()
    n = int(valid.sum())
    if n < 5:
        return pd.Series(np.nan, index=s.index)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ranks = s[valid].rank(method="average")
    uniform = (ranks - 0.5) / n
    # Inverse normal CDF via the error function -- avoids a scipy import here.
    from math import sqrt

    try:
        from scipy.special import ndtri

        out[valid] = ndtri(uniform.to_numpy())
    except ImportError:  # pragma: no cover
        out[valid] = np.sqrt(2.0) * _erfinv(2 * uniform.to_numpy() - 1) * sqrt(1.0)
    return out


def _erfinv(x: np.ndarray) -> np.ndarray:  # pragma: no cover - scipy fallback
    a = 0.147
    ln = np.log(1 - x**2)
    t1 = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(t1**2 - ln / a) - t1)


def neutralize(s: pd.Series, groups: pd.Series) -> pd.Series:
    """Demean within group (sector), so the score is relative to peers."""
    aligned = groups.reindex(s.index).fillna("Unknown")
    return s - s.groupby(aligned).transform("mean")


def standardize_panel(
    panel: pd.DataFrame,
    cols: list[str],
    sector_col: str | None = None,
    method: str = "rank",
    winsor_pct: float = 0.02,
    date_col: str = "date",
    min_names: int = 30,
) -> pd.DataFrame:
    """Standardise raw feature columns cross-sectionally, date by date.

    Parameters
    ----------
    method
        ``"rank"`` -> rank-normal (robust, recommended)
        ``"z"``    -> winsorised z-score
    sector_col
        If given, scores are demeaned within sector *before* standardising,
        so what survives is "beat your own sector", not "be in a hot sector".
    min_names
        Dates with fewer valid names than this are set to NaN. Standardising
        a 6-name cross-section produces noise that looks like signal.

    Returns a copy of ``panel`` with ``<col>_z`` columns appended.
    """
    out = panel.copy()
    cols = [c for c in cols if c in out.columns]
    if not cols:
        return out

    # Vectorised across the whole panel rather than a Python loop over dates.
    # On an 800k-row panel with ~50 features the groupby-apply version spends
    # nearly all its time rebuilding per-date DataFrames; these transforms do
    # the same arithmetic in a handful of passes.
    try:
        from scipy.special import ndtri
    except ImportError:  # pragma: no cover
        ndtri = None

    date_grp = out.groupby(date_col)
    valid_counts = date_grp[cols].transform("count")

    num = out[cols].apply(pd.to_numeric, errors="coerce")

    # ---- winsorise within date ------------------------------------------
    if winsor_pct and winsor_pct > 0:
        lo = num.groupby(out[date_col]).transform(lambda s: s.quantile(winsor_pct))
        hi = num.groupby(out[date_col]).transform(lambda s: s.quantile(1 - winsor_pct))
        num = num.clip(lower=lo, upper=hi)

    # ---- sector-neutralise ----------------------------------------------
    if sector_col and sector_col in out.columns:
        sect = out[sector_col].fillna("Unknown")
        num = num - num.groupby([out[date_col], sect]).transform("mean")

    # ---- standardise -----------------------------------------------------
    if method == "rank":
        n = date_grp[cols].transform("count")
        ranks = num.groupby(out[date_col]).rank(method="average")
        uniform = (ranks - 0.5) / n.replace(0, np.nan)
        if ndtri is not None:
            arr = uniform.to_numpy(dtype=float)
            mask = np.isfinite(arr)
            res = np.full(arr.shape, np.nan)
            res[mask] = ndtri(arr[mask])
            scored = pd.DataFrame(res, index=num.index, columns=cols)
        else:  # pragma: no cover
            scored = pd.DataFrame(
                np.sqrt(2.0) * _erfinv(2 * uniform.to_numpy() - 1),
                index=num.index, columns=cols,
            )
    else:
        mean = num.groupby(out[date_col]).transform("mean")
        sd = num.groupby(out[date_col]).transform("std")
        scored = (num - mean) / sd.replace(0, np.nan)

    # Dates with too thin a cross-section produce noise that looks like
    # signal, so they are blanked rather than standardised.
    scored = scored.where(valid_counts >= min_names)

    for c in cols:
        out[f"{c}_z"] = scored[c]
    return out


def information_coefficient(
    panel: pd.DataFrame,
    signal_col: str,
    forward_col: str,
    date_col: str = "date",
    method: str = "spearman",
) -> pd.Series:
    """Per-date rank correlation between a signal and forward returns.

    The IC time series is the honest measure of whether a signal works.
    A mean IC of 0.02-0.05 with a stable sign is a genuinely useful equity
    signal; anything above 0.10 sustained should be treated as a bug or a
    look-ahead leak until proven otherwise.
    """
    def _ic(g: pd.DataFrame) -> float:
        sub = g[[signal_col, forward_col]].dropna()
        if len(sub) < 20:
            return np.nan
        return sub[signal_col].corr(sub[forward_col], method=method)

    return panel.groupby(date_col).apply(_ic, include_groups=False)


def ic_summary(ic: pd.Series) -> dict:
    """Mean IC, its t-stat and the information ratio of the signal."""
    ic = ic.dropna()
    if len(ic) < 8:
        return {"n": len(ic), "mean_ic": np.nan, "ic_ir": np.nan, "t_stat": np.nan, "hit_rate": np.nan}
    mean, sd = ic.mean(), ic.std()
    ir = mean / sd if sd > 0 else np.nan
    return {
        "n": len(ic),
        "mean_ic": mean,
        "ic_ir": ir,
        "t_stat": ir * np.sqrt(len(ic)) if np.isfinite(ir) else np.nan,
        "hit_rate": (ic > 0).mean(),
    }
