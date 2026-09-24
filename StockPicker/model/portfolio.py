"""
============================================================
model/portfolio.py  --  Selection and position sizing
============================================================
Converts a ranked score into an actual weekly book: which five
names, and how much of the capital in each.

The brief was "maximise returns while minimising volatility",
and with only five positions those two goals genuinely fight each
other. Four mechanisms do the reconciling:

1. **Inverse-volatility weighting.** Equal-weighting five names
   means the riskiest name contributes far more than a fifth of
   portfolio risk. Sizing by 1/vol equalises *risk* contribution
   instead of dollars, which is what actually lowers portfolio
   volatility without forecasting anything.

2. **Sector cap.** At most two names per sector. Five names drawn
   from one hot sector is a sector bet wearing a stock-picking
   costume, and it is where concentrated books blow up.

3. **Replacement buffer.** A challenger must beat the incumbent's
   score by a margin before it displaces it. Weekly rebalancing of
   five names can otherwise churn 200%+ a year swapping between
   names ranked 5th and 6th, where the score difference is noise
   but the cost is real.

4. **Volatility targeting.** Gross exposure scales so that
   *forecast* portfolio volatility tracks a target; the residual
   sits in cash. This is what stops the book running 35% vol into
   a crisis, and it is applied with a forecast built only from
   trailing data.
============================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SECTOR_COL, STRATEGY

log = logging.getLogger(__name__)


def select_names(
    cross_section: pd.DataFrame,
    score_col: str = "score",
    n: int = None,
    sector_col: str = SECTOR_COL,
    max_per_sector: int = None,
    incumbents: list[str] | None = None,
    buffer: float = None,
) -> pd.DataFrame:
    """Pick the book for one rebalance date.

    Greedy descent through the ranking, honouring the sector cap, then a
    turnover-aware pass that keeps an incumbent unless a challenger clears it
    by ``buffer`` standard deviations of score.
    """
    cfg = STRATEGY
    n = n or cfg.n_positions
    max_per_sector = max_per_sector if max_per_sector is not None else cfg.max_per_sector
    buffer = buffer if buffer is not None else cfg.replacement_buffer

    df = cross_section.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if df.empty:
        return df

    # ---- greedy selection under the sector cap ---------------------------
    picked: list[int] = []
    sector_count: dict[str, int] = {}
    for idx, row in df.iterrows():
        sec = row.get(sector_col, "Unknown")
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        picked.append(idx)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picked) >= n:
            break
    book = df.loc[picked].copy()

    # ---- turnover damping: a rank buffer ---------------------------------
    # Without this the book turned over ~97% a week -- effectively a fresh
    # portfolio every Monday, costing >4% a year in fees for reshuffles that
    # are mostly noise at the 5th-vs-6th rank boundary.
    #
    # A rank buffer is the standard fix and works far better than comparing
    # scores: a held name is retained while it remains anywhere in the top
    # ``hold_rank`` of the cross-section, and is only sold once it genuinely
    # falls out of contention. Buying requires a top-N rank, selling requires
    # dropping below top-``hold_rank`` -- the gap between the two thresholds
    # is what stops the churn.
    if incumbents:
        df = df.copy()
        df["_rank"] = df[score_col].rank(ascending=False, method="first")
        hold_rank = max(n * STRATEGY.hold_rank_multiple, n + 1)
        by_sec = df.set_index("security")

        retained = [
            s for s in incumbents
            if s in by_sec.index and by_sec.at[s, "_rank"] <= hold_rank
        ]
        if retained:
            # Start from the retained names, then fill the remaining slots from
            # the fresh ranking. Sector caps are enforced across both.
            keep = df[df["security"].isin(retained)].sort_values(score_col, ascending=False)
            keep = keep.head(n)
            sector_count = keep[sector_col].value_counts().to_dict() if sector_col in keep.columns else {}
            # Drop retained names that would breach the sector cap.
            trimmed, counts = [], {}
            for idx, row in keep.iterrows():
                sec = row.get(sector_col, "Unknown")
                if counts.get(sec, 0) >= max_per_sector:
                    continue
                trimmed.append(idx)
                counts[sec] = counts.get(sec, 0) + 1
            keep = keep.loc[trimmed]

            held_secs = set(keep["security"])
            for idx, row in df.sort_values(score_col, ascending=False).iterrows():
                if len(keep) >= n:
                    break
                if row["security"] in held_secs:
                    continue
                sec = row.get(sector_col, "Unknown")
                if counts.get(sec, 0) >= max_per_sector:
                    continue
                keep = pd.concat([keep, df.loc[[idx]]])
                counts[sec] = counts.get(sec, 0) + 1
                held_secs.add(row["security"])
            book = keep

    return book.sort_values(score_col, ascending=False).reset_index(drop=True)


def compute_weights(
    book: pd.DataFrame,
    vol_col: str = "vol_60",
    method: str = None,
    max_weight: float = None,
    min_weight: float = None,
) -> pd.Series:
    """Position weights summing to 1.0 before any vol-target scaling."""
    cfg = STRATEGY
    method = method or cfg.weighting
    max_weight = max_weight if max_weight is not None else cfg.max_weight
    min_weight = min_weight if min_weight is not None else cfg.min_weight

    n = len(book)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=book.index)

    if method == "equal":
        w = pd.Series(1.0 / n, index=book.index)
    else:
        vol = pd.to_numeric(book.get(vol_col), errors="coerce")
        # A missing vol estimate must not win the auction for capital, so it
        # is imputed at the book's median rather than treated as zero risk.
        vol = vol.replace(0, np.nan).fillna(vol.median() if vol.notna().any() else 0.25)
        inv = 1.0 / (vol**2) if method == "inverse_var" else 1.0 / vol
        w = inv / inv.sum()

    # Iteratively enforce the weight band; capping one name redistributes to
    # the others, which can push a second name over the cap.
    for _ in range(20):
        w = w.clip(lower=min_weight, upper=max_weight)
        total = w.sum()
        if total == 0:
            return pd.Series(1.0 / n, index=book.index)
        w = w / total
        if (w <= max_weight + 1e-9).all() and (w >= min_weight - 1e-9).all():
            break
    return w


def volatility_scalar(
    book: pd.DataFrame,
    weights: pd.Series,
    corr_assumption: float = 0.35,
    vol_col: str = "vol_60",
    target: float | None = None,
    max_leverage: float = None,
) -> float:
    """Scale gross exposure so forecast portfolio vol meets the target.

    Portfolio variance is estimated from individual vols plus a single
    average pairwise correlation. A full covariance matrix estimated from
    60 daily observations on 5 names is noisier than this crude but stable
    approximation, and the error in a covariance estimate propagates
    straight into position sizes.

    Returns a multiplier in [0, max_leverage]. Being long-only and
    unlevered, the multiplier only ever *reduces* exposure; the remainder
    is held in cash.
    """
    cfg = STRATEGY
    target = target if target is not None else cfg.vol_target
    max_leverage = max_leverage if max_leverage is not None else cfg.max_leverage
    if target is None or book.empty:
        return min(1.0, max_leverage)

    vol = pd.to_numeric(book.get(vol_col), errors="coerce")
    vol = vol.replace(0, np.nan).fillna(vol.median() if vol.notna().any() else 0.25)
    w = weights.reindex(vol.index).fillna(0.0)

    # var = sum w_i^2 v_i^2 + rho * sum_{i != j} w_i w_j v_i v_j
    wv = (w * vol).to_numpy()
    var = float((wv**2).sum() + corr_assumption * (wv.sum() ** 2 - (wv**2).sum()))
    port_vol = np.sqrt(max(var, 1e-12))
    if not np.isfinite(port_vol) or port_vol <= 0:
        return min(1.0, max_leverage)
    return float(np.clip(target / port_vol, 0.0, max_leverage))


def build_book(
    cross_section: pd.DataFrame,
    score_col: str = "score",
    incumbents: list[str] | None = None,
    apply_vol_target: bool = True,
    n: int | None = None,
    max_per_sector: int | None = None,
) -> pd.DataFrame:
    """End-to-end: rank -> select -> weight -> vol-scale. Returns the book.

    ``n`` must be threaded all the way down to ``select_names``; truncating a
    5-name book afterwards silently makes every larger portfolio size
    identical, which is what made the concentration sensitivity table
    meaningless.
    """
    n = n or STRATEGY.n_positions
    # The sector cap has to scale with the book, otherwise a 30-name portfolio
    # is still capped at 2 per sector and cannot be filled at all.
    if max_per_sector is None:
        max_per_sector = max(STRATEGY.max_per_sector, int(np.ceil(n / 4)))

    book = select_names(
        cross_section, score_col=score_col, n=n,
        max_per_sector=max_per_sector, incumbents=incumbents,
    )
    if book.empty:
        return book
    # Weight bounds must also scale: a 20-name book cannot satisfy a 5% floor
    # and a 35% cap simultaneously in any meaningful way.
    w = compute_weights(book, max_weight=max(STRATEGY.max_weight, 1.5 / n),
                        min_weight=min(STRATEGY.min_weight, 0.5 / n))
    scalar = volatility_scalar(book, w) if apply_vol_target else 1.0
    book = book.copy()
    book["raw_weight"] = w.values
    book["vol_scalar"] = scalar
    book["weight"] = (w * scalar).values
    book["cash_weight"] = 1.0 - book["weight"].sum()
    return book
