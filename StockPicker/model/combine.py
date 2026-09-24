"""
============================================================
model/combine.py  --  Signal blending, walk-forward
============================================================
Turns a wide panel of standardised features into a single score
per stock per week, using only information that existed at the
time of the prediction.

Two blenders are provided and both are honest about the same
constraint: at rebalance date t we may use forward returns only
up to t-1, because the return from t-1 to t is the first one that
has actually been observed by the morning of t.

1. ``ic_weighted`` (default)
   Weight each feature by its own exponentially-decayed historical
   information coefficient. A feature that has predicted well
   recently gets more weight; one that has stopped working decays
   out of the blend on its own. This is deliberately close to a
   linear model: with ~1,000 weekly cross-sections and a signal
   that is 2-4% correlated with next week's return, anything more
   flexible mostly fits noise.

   The IC estimate is *shrunk* toward zero by its own standard
   error, so a feature with a high but unstable IC does not get a
   large weight on the strength of a few lucky quarters.

2. ``lightgbm``
   Gradient-boosted trees retrained on an expanding window. Can
   find interactions the linear blend cannot (e.g. momentum only
   works when volatility is low). Heavily regularised, shallow,
   and validated on the same walk-forward split. Use it as a
   challenger, not a default.
============================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import MODEL

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# family aggregation
# ----------------------------------------------------------------------
def build_family_scores(
    panel: pd.DataFrame,
    date_col: str = "date",
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse ~50 feature columns into ~11 family scores.

    Blending 50 standardised features directly has two problems that together
    were destroying the signal:

    * **Multicollinearity.** ``mom_12_1``, ``mom_6_1``, ``mom_3_1``,
      ``mom_x_fip``, ``mom_x_adx`` and ``mom_sharpe`` are six views of one
      idea. Weighted individually, momentum gets six votes while a genuinely
      independent signal like news sentiment gets one -- so the blend is a
      momentum bet whether or not that was intended.
    * **Estimation noise.** Fifty IC weights estimated from a series whose
      true IC is ~0.015 means most of the estimated weight is sampling error.
      Averaging that many noisy weights dilutes the few real signals, which is
      exactly what the diagnostics showed: composite IC 0.006 against 0.015
      for the best single feature.

    Averaging within family first fixes both: each *idea* gets one score, the
    within-family average cancels idiosyncratic feature noise, and only ~11
    weights have to be estimated. It also makes the report's factor
    attribution and the model's actual mechanics the same thing.

    Features whose expected sign is negative are flipped first, so that a
    higher family score always means "more attractive" and the within-family
    average does not cancel itself out.

    Returns ``(panel_with_family_cols, family_col_names)``.
    """
    from ..features.registry import FEATURE_FAMILIES, expected_signs

    out = panel.copy()
    signs = expected_signs()
    fam_cols: list[str] = []

    for fam, feats in FEATURE_FAMILIES.items():
        members = []
        for f in feats:
            zc = f"{f}_z"
            if zc not in out.columns:
                continue
            # Skip near-empty columns: a feature present for 5% of rows adds
            # noise to the family mean without adding information.
            if out[zc].notna().mean() < min_coverage:
                continue
            sign = signs.get(f, 0)
            # A zero expected sign means the direction is genuinely unknown;
            # include it unflipped and let the family IC weight sort it out.
            members.append(out[zc] * (-1.0 if sign < 0 else 1.0))
        if not members:
            continue
        col = f"fam_{fam}"
        out[col] = pd.concat(members, axis=1).mean(axis=1, skipna=True)
        fam_cols.append(col)

    # Re-standardise the family scores cross-sectionally so each family enters
    # the blend on an identical scale regardless of how many features it has.
    for c in fam_cols:
        g = out.groupby(date_col)[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)

    log.info("Built %s family scores: %s", len(fam_cols), ", ".join(c[4:] for c in fam_cols))
    return out, fam_cols


# ----------------------------------------------------------------------
# rolling IC estimation
# ----------------------------------------------------------------------
def rolling_ic_weights(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_ret_1w",
    date_col: str = "date",
    halflife: int = 260,
    min_periods: int = 104,   # 2y before any weight is trusted
    shrink: bool = True,
) -> pd.DataFrame:
    """Exponentially-weighted historical IC per feature, per date.

    The returned frame is indexed by date; the row for date t contains
    weights derived **only** from cross-sections strictly before t whose
    forward return was already observable. It is directly usable as the
    blend weight on date t with no further lagging.
    """
    ic_rows = []
    for dt_, g in panel.groupby(date_col, sort=True):
        y = pd.to_numeric(g[target_col], errors="coerce")
        rec = {date_col: dt_}
        for c in feature_cols:
            x = pd.to_numeric(g[c], errors="coerce")
            sub = pd.concat([x, y], axis=1).dropna()
            rec[c] = sub.iloc[:, 0].corr(sub.iloc[:, 1], method="spearman") if len(sub) >= 20 else np.nan
        ic_rows.append(rec)

    ic = pd.DataFrame(ic_rows).set_index(date_col).sort_index()

    # ---- expanding, not short-window, IC estimates ------------------------
    # A 52-week EWM was the default here and it did not work: with an IC
    # information ratio around 0.1-0.2, one year of observations gives a
    # t-statistic near 0.7, so the "weights" were almost entirely sampling
    # noise. The symptom was unmistakable -- the sector family (IC 0.004,
    # t=0.9) received the largest weight while crowding (IC 0.012, t=6.4)
    # received nearly the smallest.
    #
    # Estimating on an expanding window instead means the weight on each
    # family reflects its entire observed history, which is the only sample
    # large enough to separate these signals. A long EWM is blended in at a
    # modest fraction so the model can still adapt to a factor genuinely
    # decaying, without handing recent noise the steering wheel.
    exp_mean = ic.expanding(min_periods=min_periods).mean()
    exp_std = ic.expanding(min_periods=min_periods).std()
    count = ic.notna().expanding().sum()

    if halflife and halflife > 0:
        slow = ic.ewm(halflife=max(halflife, 260), min_periods=min_periods).mean()
        mean_ic = 0.75 * exp_mean + 0.25 * slow
    else:
        mean_ic = exp_mean
    std_ic = exp_std

    if shrink:
        # Shrink toward zero by the estimate's own standard error:
        #   w = IC * t^2 / (1 + t^2),  t = IC / (sd / sqrt(n))
        # A family whose IC is indistinguishable from noise is pulled to ~0;
        # one with a stable IC keeps nearly its full value. This is what stops
        # the blend chasing a factor that had two good quarters.
        se = std_ic / np.sqrt(count.clip(lower=1))
        t = mean_ic / se.replace(0, np.nan)
        weights = mean_ic * (t**2 / (1.0 + t**2))
    else:
        weights = mean_ic

    # Shift by TWO periods, not one.
    #
    # The IC stamped on date d is computed from the return running from d to
    # d+1, which is only complete at the *close* of d+1. We rebalance at the
    # VWAP of date t, i.e. during the day, so the newest IC we could actually
    # have in hand is the one for d = t-2:
    #
    #   IC(t-2)  needs the return t-2 -> t-1, known at the close of t-1.  OK
    #   IC(t-1)  needs the return t-1 -> t,   known at the close of t.    TOO LATE
    #
    # A single shift would let the blend weights peek one week ahead. The
    # effect is small -- the weights are an EWM with a 52-week halflife, so one
    # extra observation barely moves them -- but it is a genuine leak and the
    # whole point of this pipeline is that there are none.
    weights = weights.shift(2)
    return weights.fillna(0.0)


def ic_weighted_score(
    panel: pd.DataFrame,
    feature_cols: list[str],
    weights: pd.DataFrame,
    date_col: str = "date",
    normalise: bool = True,
) -> pd.Series:
    """Blend standardised features into one score using per-date IC weights."""
    scores = pd.Series(np.nan, index=panel.index, dtype=float)
    for dt_, g in panel.groupby(date_col, sort=True):
        if dt_ not in weights.index:
            continue
        w = weights.loc[dt_, [c for c in feature_cols if c in weights.columns]]
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if normalise:
            denom = w.abs().sum()
            if denom == 0:
                continue
            w = w / denom
        X = g[w.index].apply(pd.to_numeric, errors="coerce")
        # Mean-impute within the cross-section: a missing feature should be
        # neutral for that name, not silently drop it from the ranking.
        X = X.fillna(X.mean())
        scores.loc[g.index] = X.mul(w, axis=1).sum(axis=1)
    return scores


# ----------------------------------------------------------------------
# LightGBM challenger
# ----------------------------------------------------------------------
def walk_forward_lgbm(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_ret_1w",
    date_col: str = "date",
    min_train: int = None,
    retrain_every: int = None,
    params: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window gradient boosting, retrained periodically.

    Returns (predictions, feature_importance_history).

    The target is the *cross-sectionally demeaned* forward return, so the
    model learns relative performance rather than the market direction --
    predicting the market is a different problem and would swamp the
    stock-selection signal we are after.
    """
    try:
        import lightgbm as lgb
    except ImportError:  # pragma: no cover
        raise ImportError("lightgbm required for the ML blender: pip install lightgbm")

    min_train = min_train or MODEL.min_train_weeks
    retrain_every = retrain_every or MODEL.retrain_every_weeks
    params = params or MODEL.lgbm_params

    df = panel.sort_values(date_col).copy()
    df["_y"] = df.groupby(date_col)[target_col].transform(lambda s: s - s.mean())

    dates = np.sort(df[date_col].unique())
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    importances: list[dict] = []

    model = None
    last_train_idx = -10**9

    for i, dt_ in enumerate(dates):
        if i < min_train:
            continue
        if model is None or (i - last_train_idx) >= retrain_every:
            # Train on everything strictly before this date whose target is
            # observable. dates[i-1] is excluded because its forward return
            # only completes on dates[i].
            train = df[df[date_col] < dates[i - 1]].dropna(subset=feature_cols + ["_y"], how="any")
            if len(train) < 5000:
                continue
            model = lgb.LGBMRegressor(**params)
            model.fit(train[feature_cols], train["_y"])
            last_train_idx = i
            importances.append(
                {"date": dt_, **dict(zip(feature_cols, model.feature_importances_))}
            )
            log.info("  retrained @ %s on %s rows", pd.Timestamp(dt_).date(), len(train))

        if model is None:
            continue
        cur = df[df[date_col] == dt_]
        X = cur[feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean())
        if X.isna().all().all():
            continue
        preds.loc[cur.index] = model.predict(X)

    imp = pd.DataFrame(importances).set_index("date") if importances else pd.DataFrame()
    return preds.reindex(panel.index), imp


# ----------------------------------------------------------------------
# feature diagnostics
# ----------------------------------------------------------------------
def feature_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_ret_1w",
    date_col: str = "date",
) -> pd.DataFrame:
    """Full-sample IC diagnostics per feature.

    This is *in-sample* by construction and is a research diagnostic only --
    it must never be used to pick features for the live model, or the whole
    walk-forward discipline is undone. Its purpose is to show which signals
    carry information at all, and with what sign.
    """
    from ..features.crosssec import ic_summary, information_coefficient

    rows = []
    for c in feature_cols:
        ic = information_coefficient(panel, c, target_col, date_col=date_col)
        s = ic_summary(ic)
        s["feature"] = c
        # Split-half stability: does the sign survive out of its own era?
        mid = len(ic.dropna()) // 2
        first, second = ic.dropna().iloc[:mid], ic.dropna().iloc[mid:]
        s["ic_first_half"] = first.mean() if len(first) else np.nan
        s["ic_second_half"] = second.mean() if len(second) else np.nan
        s["sign_stable"] = bool(np.sign(s["ic_first_half"]) == np.sign(s["ic_second_half"]))
        rows.append(s)

    out = pd.DataFrame(rows).set_index("feature")
    return out.sort_values("mean_ic", key=abs, ascending=False)
