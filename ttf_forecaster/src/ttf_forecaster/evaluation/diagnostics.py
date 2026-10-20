"""Model-risk diagnostics: ablation, sensitivity, feature-shift, stability, label-overlap audits and
the criterion-3 positive control.

Owner: model-risk agent. Everything here *re-uses* the evaluation stack (``evaluation.splits``,
``evaluation.backtest.walk_forward``, ``evaluation.statistical_tests``); nothing re-implements a
split or a test. The functions answer "would this conclusion survive?", not "how good is it?".
They never tune: every comparison is run on the same folds and the same rows as the base run, and
every variant is reported, including the ones that make the model look worse.

Vocabulary
----------
* ``Experiment`` - a feature matrix, a TARGET_SCHEMA frame of one target x horizon, a split
  configuration and a model name. ``run_experiment`` evaluates it with ``walk_forward``.
* *nested comparison* - a large model (more feature families) against a small one fitted with the
  same estimator on the same folds. Reported with HLN-DM (does the large model have lower loss?)
  and Clark-West (is the small model's MSPE inflated by noise only?). DM is the gate (D-020);
  CW is supplementary and, with shrinkage estimators, can be oversized (see
  ``nested_test_size``).
* *effective overlap* - the number of later origins whose label window overlaps an origin's label
  (``label_overlap_rows``). For T3 it equals ``h - 1`` on a contiguous session grid; for T4/T5 the
  TARGET_SCHEMA ``horizon`` is a month count and the overlap is tens of sessions. Model inner-CV
  gaps and HAC lags must use the overlap, not the month count.

Everything run on ``data/synthetic.py`` output is SYNTHETIC and is never evidence about TTF.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import MARKET_TZ, DataBundle
from ttf_forecaster.evaluation.backtest import (
    TradingConventions,
    simulate_trading,
    walk_forward,
)
from ttf_forecaster.evaluation.metrics import oos_r2, paired_frame
from ttf_forecaster.evaluation.splits import Fold, SplitConfig, make_splits
from ttf_forecaster.evaluation.statistical_tests import clark_west, diebold_mariano
from ttf_forecaster.features.registry import PRICE_ONLY_FAMILIES, family_of
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.models.interface import make_model
from ttf_forecaster.targets.common import label_overlap_rows, session_horizon  # noqa: F401

log = get_logger(__name__)

ZERO_ID = "zero"
ABLATION_FAMILIES: tuple[str, ...] = (
    "price", "curve", "cross_commodity", "storage", "flows", "lng", "weather", "power",
    "calendar", "interactions", "text",
)


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------


@dataclass
class Experiment:
    """One target x horizon evaluation problem.

    Attributes:
        X: feature matrix indexed by ``origin_time`` (tz-aware UTC), point-in-time.
        targets: TARGET_SCHEMA rows of exactly one ``(target_id, horizon)``.
        horizon: overlap length in sessions used for model inner-CV gaps and HAC lags. For T3
            this is ``h``; for month-based targets use ``label_overlap_rows(targets) + 1``.
        split_cfg: outer-fold configuration (``None`` -> ``configs/backtests.yaml``).
        model: ``make_model`` name.
        model_params: extra model parameters (never tuned on test folds).
        dropna_features: passed to ``walk_forward`` (False for models that impute internally).
    """

    X: pd.DataFrame
    targets: pd.DataFrame
    horizon: int
    split_cfg: SplitConfig | None = None
    model: str = "ridge"
    model_params: dict[str, Any] = field(default_factory=dict)
    dropna_features: bool = False
    _folds: list[Fold] | None = field(default=None, repr=False)

    def folds(self) -> list[Fold]:
        """Outer folds over origins with a label and a feature row (cached)."""
        if self._folds is None:
            self._folds = make_splits(self.targets, horizon=self.horizon, cfg=self.split_cfg,
                                      valid_origins=self.X.index)
        return self._folds

    def with_(self, **changes: Any) -> Experiment:
        """Copy with changes; folds are recomputed unless ``X`` and targets are unchanged."""
        keep_folds = not ({"targets", "split_cfg", "horizon"} & set(changes)) and \
            ("X" not in changes or changes["X"].index.equals(self.X.index))
        new = replace(self, **changes)
        new._folds = self._folds if keep_folds else None
        return new

    def factory(self) -> Callable[[], Any]:
        name, params, h = self.model, dict(self.model_params), int(self.horizon)
        return lambda: make_model(name, horizon=h, **params)


def run_experiment(exp: Experiment, columns: Sequence[str] | None = None, *,
                   model_id: str = "model", folds: Sequence[Fold] | None = None) -> pd.DataFrame:
    """Walk-forward predictions of ``exp.model`` on ``columns`` plus the zero benchmark.

    Returns the long ``walk_forward`` prediction frame (model_id rows and ``zero`` rows). An empty
    column list is evaluated as the zero forecast itself (the model cannot be fitted on nothing).
    """
    X = exp.X if columns is None else exp.X.loc[:, list(columns)]
    folds = list(folds) if folds is not None else exp.folds()
    if X.shape[1] == 0:
        res = walk_forward(lambda: make_model("zero_return", horizon=exp.horizon),
                           exp.X.iloc[:, :0].assign(_c=0.0), exp.targets, folds, None,
                           model_id=model_id, is_synthetic=None, dropna_features=False)
        return res.predictions
    res = walk_forward(exp.factory(), X, exp.targets, folds, None, model_id=model_id,
                       dropna_features=exp.dropna_features)
    return res.predictions


def _model_rows(preds: pd.DataFrame, model_id: str) -> pd.DataFrame:
    return preds.loc[preds["model_id"] == model_id]


def skill_vs_zero(preds: pd.DataFrame, model_id: str, h: int) -> dict[str, float]:
    """OOS R^2 vs the zero (forward-curve) benchmark, HLN-DM and CW p-values, fold win share."""
    p = paired_frame(preds, model_id, ZERO_ID).dropna(subset=["actual", "model", "bench"])
    if p.empty:
        return {"oos_r2": np.nan, "dm_p": np.nan, "cw_p": np.nan, "fold_win_share": np.nan,
                "n": 0}
    dm = diebold_mariano(p["actual"] - p["model"], p["actual"] - p["bench"], h)
    cw = clark_west(p["actual"], p["bench"], p["model"], h)
    wins = p.groupby("fold_id").apply(
        lambda g: float(((g["actual"] - g["bench"]) ** 2).mean()
                        - ((g["actual"] - g["model"]) ** 2).mean()) > 0, include_groups=False)
    return {"oos_r2": oos_r2(p["actual"], p["model"], p["bench"]), "dm_p": dm.p_value,
            "cw_p": cw.p_value, "fold_win_share": float(wins.mean()), "n": len(p)}


def compare_nested(large: pd.DataFrame, small: pd.DataFrame, h: int, *,
                   large_id: str = "model", small_id: str = "model") -> dict[str, float]:
    """Large model (more families) vs small model, on identical origins.

    Returns OOS R^2 of each vs zero, ``delta_oos_r2 = r2_small - r2_large`` (negative = the extra
    families helped), HLN-DM p (H1: large has lower MSE), CW p (small nested in large) and the
    share of folds where the large model has lower MSE.

    Raises:
        ValueError: the two runs do not cover the same origins.
    """
    a = _model_rows(large, large_id).set_index("origin_time")
    b = _model_rows(small, small_id).set_index("origin_time")
    if not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError("large and small runs cover different origins")
    j = pd.DataFrame({"actual": a["actual"], "large": a["prediction"],
                      "small": b["prediction"].reindex(a.index), "fold_id": a["fold_id"]})
    j = j.sort_index().dropna()
    r2_l = skill_vs_zero(large, large_id, h)["oos_r2"]
    r2_s = skill_vs_zero(small, small_id, h)["oos_r2"]
    if len(j) < 2 * h + 10:
        return {"oos_r2_large": r2_l, "oos_r2_small": r2_s, "delta_oos_r2": r2_s - r2_l,
                "dm_stat": np.nan, "dm_p": np.nan, "cw_stat": np.nan, "cw_p": np.nan,
                "fold_win_share_large": np.nan, "n": len(j)}
    dm = diebold_mariano(j["actual"] - j["large"], j["actual"] - j["small"], h)
    cw = clark_west(j["actual"], j["small"], j["large"], h)
    wins = j.groupby("fold_id").apply(
        lambda g: float(((g["actual"] - g["small"]) ** 2).mean()
                        - ((g["actual"] - g["large"]) ** 2).mean()) > 0, include_groups=False)
    return {"oos_r2_large": r2_l, "oos_r2_small": r2_s, "delta_oos_r2": r2_s - r2_l,
            "dm_stat": dm.statistic, "dm_p": dm.p_value, "cw_stat": cw.statistic,
            "cw_p": cw.p_value, "fold_win_share_large": float(wins.mean()), "n": len(j)}


def columns_of(X: pd.DataFrame, families: Iterable[str]) -> list[str]:
    fams = set(families)
    return [c for c in X.columns if family_of(c) in fams]


def families_present(X: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for c in X.columns:
        f = family_of(c)
        if f not in seen:
            seen.append(f)
    return seen


# ---------------------------------------------------------------------------------------------
# (a) Family ablation
# ---------------------------------------------------------------------------------------------


def family_ablation(exp: Experiment, families: Sequence[str] | None = None, *,
                    base_families: Sequence[str] = PRICE_ONLY_FAMILIES,
                    modes: Sequence[str] = ("drop", "add")) -> pd.DataFrame:
    """Drop-one-family and add-one-family-to-price-only ablations, all on the full run's folds.

    * ``drop``: full model vs full minus family F. ``delta_oos_r2`` = R^2(without F) - R^2(full);
      < 0 means F helped. DM/CW test whether the full model beats the reduced one.
    * ``add``: (price-only + F) vs price-only (``base_families``) - criterion 3 one family at a
      time. ``delta_oos_r2`` = R^2(price-only) - R^2(price-only + F); < 0 means F adds skill.

    Families absent from ``exp.X`` are reported with ``n_cols = 0`` and NaN statistics, never
    silently skipped. Caveat: ``interactions`` columns built from a dropped parent family are NOT
    dropped with it (they are their own family); read the ``interactions`` row with that in mind.
    """
    fams = list(families) if families is not None else list(ABLATION_FAMILIES)
    folds = exp.folds()
    present = families_present(exp.X)
    full = run_experiment(exp, model_id="model", folds=folds)
    base_cols = columns_of(exp.X, base_families)
    base = run_experiment(exp, base_cols, folds=folds) if "add" in modes else None
    rows: list[dict[str, Any]] = []
    h = exp.horizon
    full_skill = skill_vs_zero(full, "model", h)
    rows.append({"family": "<full>", "mode": "reference", "n_cols": exp.X.shape[1],
                 "oos_r2_variant": full_skill["oos_r2"], "oos_r2_reference": np.nan,
                 "delta_oos_r2": 0.0, "dm_p": full_skill["dm_p"], "cw_p": full_skill["cw_p"],
                 "fold_win_share": full_skill["fold_win_share"], "n": full_skill["n"]})
    for fam in fams:
        cols = columns_of(exp.X, [fam])
        if "drop" in modes:
            if not cols or fam not in present:
                rows.append(_absent_row(fam, "drop"))
            else:
                keep = [c for c in exp.X.columns if c not in set(cols)]
                red = run_experiment(exp, keep, folds=folds)
                c = compare_nested(full, red, h)
                rows.append({"family": fam, "mode": "drop", "n_cols": len(cols),
                             "oos_r2_variant": c["oos_r2_small"],
                             "oos_r2_reference": c["oos_r2_large"],
                             "delta_oos_r2": c["delta_oos_r2"], "dm_p": c["dm_p"],
                             "cw_p": c["cw_p"], "fold_win_share": c["fold_win_share_large"],
                             "n": c["n"]})
        if "add" in modes and base is not None:
            if not cols or fam in base_families:
                rows.append(_absent_row(fam, "add", in_base=fam in base_families))
            else:
                big = run_experiment(exp, base_cols + cols, folds=folds)
                c = compare_nested(big, base, h)
                rows.append({"family": fam, "mode": "add", "n_cols": len(cols),
                             "oos_r2_variant": c["oos_r2_large"],
                             "oos_r2_reference": c["oos_r2_small"],
                             "delta_oos_r2": c["oos_r2_small"] - c["oos_r2_large"],
                             "dm_p": c["dm_p"], "cw_p": c["cw_p"],
                             "fold_win_share": c["fold_win_share_large"], "n": c["n"]})
    out = pd.DataFrame(rows)
    out.attrs["note"] = ("drop: variant = full minus family, reference = full; add: variant = "
                         "price-only + family, reference = price-only. delta_oos_r2 = R^2(model "
                         "WITHOUT the family) - R^2(model WITH it): negative = the family helped. "
                         "dm_p / cw_p test the model WITH the family against the one without.")
    return out


def _absent_row(fam: str, mode: str, in_base: bool = False) -> dict[str, Any]:
    return {"family": fam, "mode": mode, "n_cols": 0, "oos_r2_variant": np.nan,
            "oos_r2_reference": np.nan, "delta_oos_r2": np.nan, "dm_p": np.nan, "cw_p": np.nan,
            "fold_win_share": np.nan, "n": 0,
            "note": "in base set" if in_base else "family absent from X"}


def price_only_test(exp: Experiment,
                    price_families: Sequence[str] = PRICE_ONLY_FAMILIES) -> dict[str, Any]:
    """Spec §8 criterion 3: full model vs its price-only twin (same estimator, same folds).

    ``passed`` = HLN-DM p < 0.10 (the spec's threshold). CW is reported, never gates (D-020).
    """
    folds = exp.folds()
    full = run_experiment(exp, folds=folds)
    po = run_experiment(exp, columns_of(exp.X, price_families), folds=folds)
    c = compare_nested(full, po, exp.horizon)
    c["passed"] = bool(np.isfinite(c["dm_p"]) and c["dm_p"] < 0.10)
    c["full_vs_zero"] = skill_vs_zero(full, "model", exp.horizon)
    c["price_only_vs_zero"] = skill_vs_zero(po, "model", exp.horizon)
    return c


# ---------------------------------------------------------------------------------------------
# (c) Feature-shift test
# ---------------------------------------------------------------------------------------------


def feature_shift_test(exp: Experiment, columns: Sequence[str],
                       shifts: Sequence[int] = (-1, 0, 1, 2), *,
                       implausible_r2: float = 0.05,
                       min_retention: float = 0.2) -> pd.DataFrame:
    """Re-run the model with ``columns`` shifted by ``k`` rows of the origin grid.

    ``k > 0`` lags the feature (origin t sees the value of origin t-k: strictly *less*
    information); ``k < 0`` peeks (origin t sees origin t+k's value: deliberate leakage, a
    reference point). Other columns are left alone. Rows are origins, so ``k`` is in sessions only
    on a contiguous session grid.

    Flags (heuristics - the leakage canary is the gate, this is the cross-check of PIT policy §6):

    * ``suspect_leak``: skill at k=0 is implausibly high for a return target
      (``oos_r2 > implausible_r2``) **and** collapses when lagged one session
      (``r2(+1) / r2(0) < min_retention``). A genuine but fresh signal also loses skill when
      lagged, so the implausibility prior does the discriminating; read the profile, not the flag.
    * ``peek_gain``: ``r2(-1) - r2(0)``. Large positive values mean the target is coincident with
      the feature's *next* value: a one-session error in the publication-time rule would turn this
      feature into a leak, so its lag rule needs evidence.
    """
    if 0 not in shifts:
        shifts = (0, *shifts)
    folds = exp.folds()
    rows = []
    for k in sorted(set(shifts)):
        X = exp.X.copy()
        X[list(columns)] = X[list(columns)].shift(k)
        p = run_experiment(exp.with_(X=X), folds=folds)
        s = skill_vs_zero(p, "model", exp.horizon)
        rows.append({"shift": k, **s})
    out = pd.DataFrame(rows).set_index("shift")
    r0 = float(out.loc[0, "oos_r2"])
    r1 = float(out.loc[1, "oos_r2"]) if 1 in out.index else np.nan
    rm = float(out.loc[-1, "oos_r2"]) if -1 in out.index else np.nan
    retention = r1 / r0 if r0 > 0 else np.nan
    out.attrs["retention_lag1"] = retention
    out.attrs["peek_gain"] = rm - r0
    out.attrs["suspect_leak"] = bool(r0 > implausible_r2 and np.isfinite(retention)
                                     and retention < min_retention)
    return out


# ---------------------------------------------------------------------------------------------
# (d) Stability
# ---------------------------------------------------------------------------------------------


def _importance(model: Any) -> pd.Series:
    for attr in ("coef_", "feature_importance_"):
        v = getattr(model, attr, None)
        if isinstance(v, pd.Series):
            return v.drop(labels=[x for x in v.index if str(x) in ("const", "intercept")],
                          errors="ignore").astype(float)
    raise TypeError(f"{type(model).__name__} exposes neither coef_ nor feature_importance_")


def coefficient_stability(exp: Experiment, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-fold coefficients (or importances) of the first refit of each outer fold.

    Returns a frame with one row per feature: ``n_folds``, ``share_positive``,
    ``sign_agreement`` = |mean(sign)| (1 = same sign in every fold), ``mean``, ``std`` and
    ``abs_rank_mean``; ``attrs["per_fold"]`` holds the fold x feature table. Coefficients are on
    the model's internal (standardised) scale. A fundamental effect whose sign flips across folds
    fails spec §8 criterion 6 regardless of its pooled significance.
    """
    X = exp.X if columns is None else exp.X.loc[:, list(columns)]
    t = exp.targets.set_index(pd.DatetimeIndex(exp.targets["origin_time"]))
    per: dict[str, pd.Series] = {}
    for fold in exp.folds():
        seg = fold.segments[0]
        tr = seg.train_origins[seg.train_origins.isin(X.index)]
        y = t.loc[tr, "value"].astype(float)
        ok = y.notna().to_numpy()
        m = exp.factory()()
        m.fit(X.loc[tr[ok]], y.loc[ok])
        per[fold.fold_id] = _importance(m)
    tab = pd.DataFrame(per).T
    sgn = np.sign(tab)
    ranks = tab.abs().rank(axis=1, ascending=False)
    summ = pd.DataFrame({
        "n_folds": tab.notna().sum(),
        "share_positive": (sgn > 0).sum() / tab.notna().sum(),
        "sign_agreement": sgn.mean().abs(),
        "mean": tab.mean(),
        "std": tab.std(ddof=1),
        "abs_rank_mean": ranks.mean(),
    }).sort_values("abs_rank_mean")
    summ.attrs["per_fold"] = tab
    return summ


def skill_concentration(paired: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Where the MSE reduction vs the benchmark comes from.

    ``paired`` is ``metrics.paired_frame`` output. ``by`` = ``"year"`` (local calendar year of the
    origin) or any column of ``paired``. Per group: n, OOS R^2 in the group, the group's share of
    the total squared-error reduction, and the pooled OOS R^2 with the group left out
    (``oos_r2_ex_group``). ``attrs``: ``max_share`` and ``fragile`` (True if leaving out one group
    turns pooled OOS R^2 non-positive while the full-sample R^2 is positive - spec §9's "one
    exceptional year" stop criterion).
    """
    p = paired.dropna(subset=["actual", "model", "bench"]).copy()
    if by == "year":
        p["year"] = pd.DatetimeIndex(p["origin_time"]).tz_convert(MARKET_TZ).year
    gain = (p["actual"] - p["bench"]) ** 2 - (p["actual"] - p["model"]) ** 2
    total = float(gain.sum())
    rows = []
    for key, g in p.groupby(by, sort=True):
        rest = p.loc[p[by] != key]
        rows.append({
            by: key, "n": len(g),
            "oos_r2_group": oos_r2(g["actual"], g["model"], g["bench"]),
            "gain_share": float(gain.loc[g.index].sum()) / total if total else np.nan,
            "oos_r2_ex_group": oos_r2(rest["actual"], rest["model"], rest["bench"])
            if len(rest) else np.nan,
        })
    out = pd.DataFrame(rows)
    r2_all = oos_r2(p["actual"], p["model"], p["bench"])
    out.attrs["oos_r2_all"] = r2_all
    out.attrs["max_share"] = float(out["gain_share"].max()) if len(out) else np.nan
    out.attrs["fragile"] = bool(r2_all > 0 and (out["oos_r2_ex_group"] <= 0).any())
    return out


def pnl_concentration(daily: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    """Net P&L by local calendar year from ``simulate_trading(...).daily``, with each year's share
    of total net P&L and the total with that year removed."""
    d = daily.copy()
    d["year"] = pd.DatetimeIndex(d["trade_date"]).tz_convert(MARKET_TZ).year
    tot = float(d["net_pnl"].sum())
    g = d.groupby(by)["net_pnl"].sum()
    return pd.DataFrame({"net_pnl": g, "share": g / tot if tot else np.nan,
                         "total_ex_group": tot - g})


# ---------------------------------------------------------------------------------------------
# Label-overlap audits (D-011 / D-021)
# ---------------------------------------------------------------------------------------------


def inner_gap_violations(targets: pd.DataFrame, gap: int) -> int:
    """Training rows whose label is NOT known at the first validation origin, summed over every
    possible validation start, for a row-gap ``gap`` (``models.interface.inner_time_splits``).

    Row j trains for a validation block starting at row b iff j < b - gap. Violation:
    ``label_known_time[j] > origin[b]``. Returns the count over b (0 = gap sufficient).
    """
    t = targets.dropna(subset=["value"]).sort_values("origin_time")
    o = pd.DatetimeIndex(t["origin_time"]).asi8
    k = pd.DatetimeIndex(t["label_known_time"]).asi8
    n = len(o)
    run_max = np.maximum.accumulate(k)
    bad = 0
    for b in range(gap + 1, n):
        j = b - gap - 1  # last training row
        if run_max[j] > o[b]:
            bad += int((k[: j + 1] > o[b]).sum())
    return bad


def target_lag_violations(targets: pd.DataFrame, horizon: int, p: int = 1) -> int:
    """Rows where ``models.baselines.make_target_lags(y, horizon=horizon, p=p)`` hands the model a
    lagged label that is not yet known at the origin (``label_known_time`` > origin)."""
    t = targets.dropna(subset=["value"]).sort_values("origin_time").reset_index(drop=True)
    o = pd.DatetimeIndex(t["origin_time"])
    k = pd.Series(pd.DatetimeIndex(t["label_known_time"]))
    bad = 0
    for lag in range(1, p + 1):
        lk = k.shift(horizon + lag - 1)
        bad += int((lk > pd.Series(o)).sum())
    return bad


# ---------------------------------------------------------------------------------------------
# (b) Sensitivity harness
# ---------------------------------------------------------------------------------------------


def shift_publication(bundle: DataBundle, days: int, *, grades: Iterable[str] = ("B",),
                      tables: Iterable[str] = ("observations", "forecasts")) -> DataBundle:
    """Copy of ``bundle`` with ``publication_time`` (and ``ingestion_time``) of rows of the given
    PIT grades moved by ``days`` calendar days (PIT policy §4: +1/+3 on grade-B series)."""
    g = set(grades)
    delta = pd.Timedelta(days=int(days))

    def one(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out
        m = out["pit_grade"].isin(g)
        out.loc[m, "publication_time"] = out.loc[m, "publication_time"] + delta
        if "ingestion_time" in out:
            out.loc[m, "ingestion_time"] = out.loc[m, "ingestion_time"] + delta
        return out

    t = set(tables)
    return DataBundle(
        one(bundle.observations) if "observations" in t else bundle.observations.copy(),
        one(bundle.forecasts) if "forecasts" in t else bundle.forecasts.copy(),
        one(bundle.curves) if "curves" in t else bundle.curves.copy(),
        bundle.contracts.copy())


def delay_targets(targets: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Executable-label variant: the label of origin t becomes the label of origin t+delay
    (``r(t+d, t+d+h)`` on the contract held from t+d), with that origin's ``label_known_time``.

    Criteria 1-3 score ``r(t, t+h)``, which starts at the settlement that produced the signal; no
    order can be filled there (D-021 fills at t+1). With ``delay = 1`` the statistical test and the
    trading convention measure the same thing. Rows are shifted on the session grid of the frame.
    """
    if delay < 0:
        raise ValueError("delay must be >= 0")
    t = targets.sort_values("origin_time").reset_index(drop=True).copy()
    if delay == 0:
        return t
    for c in ("value", "label_known_time", "contract_id", "quality_flag", "construction"):
        if c in t:
            t[c] = t[c].shift(-delay)
    t = t.dropna(subset=["label_known_time"]).reset_index(drop=True)
    t["label_known_time"] = pd.DatetimeIndex(t["label_known_time"]).tz_convert("UTC")
    return t


def execution_cost_grid(forecasts: pd.Series, targets: pd.DataFrame, curves: pd.DataFrame, *,
                        delays: Sequence[int] = (1, 2), cost_multipliers: Sequence[float] = (
                            0.0, 1.0, 2.0, 4.0), rebalance_every: int | None = None
                        ) -> pd.DataFrame:
    """Trading summary for every (execution delay, cost multiplier). Delay 0 is never run."""
    base = TradingConventions.from_config()
    if rebalance_every is not None:
        base = base.with_(rebalance_every=int(rebalance_every))
    rows = []
    for d in delays:
        if d < 1:
            raise ValueError("execution delay must be >= 1 (same-settlement fills are refused)")
        for m in cost_multipliers:
            tr = simulate_trading(forecasts, targets, curves,
                                  base.with_(execution_delay=int(d), cost_multiplier=float(m)))
            rows.append({"execution_delay": d, "cost_multiplier": m, **tr.summary})
    return pd.DataFrame(rows)


@dataclass
class Variant:
    """One sensitivity: how to transform the base ``Experiment``.

    ``exp_fn`` receives the base experiment and returns the variant. Use the helpers below.
    """

    name: str
    exp_fn: Callable[[Experiment], Experiment]
    group: str = ""


def sensitivity_table(base: Experiment, variants: Sequence[Variant], *,
                      price_families: Sequence[str] | None = PRICE_ONLY_FAMILIES) -> pd.DataFrame:
    """Evaluate the base and each variant: skill vs zero and (optionally) criterion 3.

    Each variant gets its own folds (targets, splits or rows may differ). Columns: ``variant``,
    ``group``, ``n``, ``oos_r2``, ``dm_p``, ``cw_p``, ``fold_win_share`` and, if
    ``price_families``, ``c3_dm_p`` / ``c3_delta_oos_r2`` (full vs price-only twin).
    """
    rows = []
    for v in [Variant("<base>", lambda e: e, "base"), *variants]:
        e = v.exp_fn(base)
        p = run_experiment(e)
        row: dict[str, Any] = {"variant": v.name, "group": v.group,
                               **skill_vs_zero(p, "model", e.horizon)}
        if price_families is not None:
            pc = columns_of(e.X, price_families)
            if pc and len(pc) < e.X.shape[1]:
                po = run_experiment(e, pc)
                c = compare_nested(p, po, e.horizon)
                row["c3_dm_p"] = c["dm_p"]
                row["c3_cw_p"] = c["cw_p"]
                row["c3_delta_oos_r2"] = c["delta_oos_r2"]
        rows.append(row)
    return pd.DataFrame(rows)


def variant_split(name: str, **cfg: Any) -> Variant:
    """Split-config variant (e.g. ``scheme="rolling", rolling_window_years=3``)."""
    def fn(e: Experiment) -> Experiment:
        sc = e.split_cfg or SplitConfig.from_config()
        return e.with_(split_cfg=replace(sc, **cfg))
    return Variant(name, fn, "training_window")


def variant_execution_delay(delay: int) -> Variant:
    """Statistical evaluation on the executable label ``r(t+d, t+d+h)``."""
    return Variant(f"label_delay_{delay}", lambda e: e.with_(targets=delay_targets(e.targets,
                                                                                    delay)),
                   "execution")


def variant_rebuild(name: str, group: str,
                    build: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
                    horizon: int | None = None) -> Variant:
    """Variant whose features and/or targets are rebuilt by ``build() -> (X, targets)`` - for
    publication-lag shifts, roll rules and alternative targets."""
    def fn(e: Experiment) -> Experiment:
        X, t = build()
        changes: dict[str, Any] = {"X": X, "targets": t}
        if horizon is not None:
            changes["horizon"] = int(horizon)
        return e.with_(**changes)
    return Variant(name, fn, group)


# ---------------------------------------------------------------------------------------------
# Building experiments from a bundle (used by the sensitivity variants and the positive control)
# ---------------------------------------------------------------------------------------------


def build_targets(curves: pd.DataFrame, horizon: int, *, target: str = "m1_logret",
                  roll_days: int | None = None) -> pd.DataFrame:
    """T3 (``m1_logret``) or T5a log (``delivery_excess_fut_log``) targets on fixed contracts."""
    from ttf_forecaster.contracts.rolls import RollRule
    from ttf_forecaster.targets import futures_targets
    from ttf_forecaster.targets.build import CurveContext

    rule = RollRule.from_config(roll_days=roll_days)
    ctx = CurveContext.from_curve(curves, rule=rule)
    if target == "m1_logret":
        return futures_targets.build_m1_logret(ctx, [horizon])
    if target in ("delivery_excess_fut_log", "delivery_excess_fut"):
        t = futures_targets.build_delivery_excess_futures(ctx, [horizon])
        return t.loc[t["target_id"] == target].reset_index(drop=True)
    raise ValueError(f"unsupported target {target!r}")


def build_features(bundle: DataBundle, origins: pd.DatetimeIndex, families: Sequence[str], *,
                   family_params: Mapping[str, Mapping[str, Any]] | None = None,
                   common: Mapping[str, Any] | None = None,
                   extra_families: Mapping[str, Mapping[str, Any]] | None = None,
                   min_coverage: float = 0.2) -> pd.DataFrame:
    """Feature matrix for ``families`` through ``pipelines.build_features`` (the sanctioned path),
    with columns below ``min_coverage`` non-NaN share dropped (as ``synthetic_e2e``)."""
    from ttf_forecaster.pipelines.build_features import build_feature_matrix, load_feature_config

    fc = load_feature_config(None)
    for name, spec in (extra_families or {}).items():
        fc["families"][name] = copy.deepcopy(dict(spec))
    for fam in families:
        fc["families"].setdefault(fam, {})["enabled"] = True
    for fam, prm in (family_params or {}).items():
        fc["families"].setdefault(fam, {}).setdefault("params", {}).update(dict(prm))
    if common:
        fc.setdefault("common", {}).update(dict(common))
    X = build_feature_matrix(bundle, origins, fc, families=list(families))
    return X.loc[:, X.notna().mean() > float(min_coverage)]


# ---------------------------------------------------------------------------------------------
# (e) Positive control for criterion 3 (D-026)
# ---------------------------------------------------------------------------------------------


def white_storage_observations(cfg: synthetic.SyntheticConfig, *, step_sd: float = 3.0,
                               seed_offset: int = 101) -> pd.DataFrame:
    """``synthetic.storage_observations`` rows with the inventory replaced by a seasonal-free
    random walk, so its day-on-day change is white noise (SYNTHETIC).

    Publication times, revision structure, grades and provenance are kept from the generator; the
    revision noise (N(0, 2) TWh, published 3 days later) is re-drawn on the new first release.
    """
    obs = synthetic.storage_observations(cfg)
    rng = np.random.default_rng(cfg.seed + seed_offset)
    first = obs.loc[obs["revision"] == 0].sort_values("event_time")
    walk = 650.0 + np.cumsum(rng.normal(0.0, step_sd, len(first)))
    level = pd.Series(walk, index=pd.DatetimeIndex(first["event_time"]))
    out = obs.copy()
    out["value"] = level.reindex(pd.DatetimeIndex(out["event_time"])).to_numpy()
    rev = out["revision"] > 0
    out.loc[rev, "value"] = out.loc[rev, "value"] + rng.normal(0.0, 2.0, int(rev.sum()))
    return out


@dataclass
class ControlWorld:
    """A synthetic bundle plus the ingredients of a criterion-3 test."""

    bundle: DataBundle
    targets: pd.DataFrame
    X: pd.DataFrame
    horizon: int
    beta: float


def positive_control_world(*, start: str = "2019-01-02", end: str = "2021-12-31",
                           first_origin: str = "2019-07-01", horizon: int = 1,
                           beta: float = 0.01, seed: int = 11,
                           families: Sequence[str] = ("price", "curve", "calendar", "storage"),
                           ) -> ControlWorld:
    """World in which a WHITE storage-only driver moves TTF returns (``beta = 0`` -> null twin).

    The planted driver is ``synthetic._storage_surprise_by_day`` with ``planted_driver="change"``
    applied to ``white_storage_observations``: the standardised first-release inventory change,
    published at D+2 07:00, moves the return of the next session. Because the change is white,
    past returns carry no information about it and a price-only model cannot proxy it - so the
    full model must beat its price-only twin (criterion 3) when ``beta > 0``.

    Storage features are built with the projection disabled and no working-gas-volume series
    (only ``storage__level_twh`` / ``storage__net_injection*`` survive); that keeps the control
    fast and makes the storage block exactly the planted information plus noise columns.
    """
    cfg = synthetic.SyntheticConfig(start=start, end=end, seed=seed,
                                    world="planted" if beta else "null",
                                    planted_beta=float(beta), planted_driver="change",
                                    contract_noise=0.002)
    storage = white_storage_observations(cfg)
    curves = synthetic.ttf_curve(cfg, storage=storage)
    b0 = DataBundle.empty()
    bundle = DataBundle(storage, b0.forecasts, curves, b0.contracts)
    targets = build_targets(curves, horizon)
    targets = targets.loc[targets["origin_time"] >= pd.Timestamp(first_origin, tz=MARKET_TZ)
                          ].reset_index(drop=True)
    origins = pd.DatetimeIndex(targets["origin_time"].unique())
    fp = {"storage": {"projection_enabled": False}}
    X = build_features(bundle, origins, list(families), family_params=fp)
    X = X.loc[:, [c for c in X.columns if not c.startswith("storage__normal_")]]
    return ControlWorld(bundle, targets, X, horizon, float(beta))


def control_split_config() -> SplitConfig:
    """Short-window outer folds for the control worlds (quarterly blocks; holdout after the data)."""
    return SplitConfig(test_block_months=3, min_train_origins=120, min_test_origins=20,
                       holdout_start="2030-01-01")


# ---------------------------------------------------------------------------------------------
# Size of the nested tests under the null (why CW never gates)
# ---------------------------------------------------------------------------------------------


def nested_test_size(*, n_reps: int = 40, n: int = 600, k_small: int = 3, k_extra: int = 10,
                     h: int = 1, model: str = "ridge", seed: int = 0,
                     model_params: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Monte Carlo rejection rates (5 %) of HLN-DM and CW for full-vs-small nested models when the
    extra regressors are pure noise and y is white (the null of criterion 3).

    Uses the project's model factory and an expanding walk-forward with ``n // 6`` row blocks
    after a ``n // 2`` warm-up. Nominal size is 0.05; a rate well above it means the test
    over-rejects for this estimator and must not be read as evidence.
    """
    rng = np.random.default_rng(seed)
    prm = dict(model_params or {})
    dm_rej = cw_rej = 0
    for _ in range(n_reps):
        Xa = rng.normal(size=(n, k_small + k_extra))
        y = rng.normal(size=n)
        start, block = n // 2, max(n // 6, 20)
        f_s = np.full(n, np.nan)
        f_l = np.full(n, np.nan)
        idx = (pd.bdate_range("2020-01-02", periods=n) + pd.Timedelta(hours=18)).tz_localize("UTC")
        for b0 in range(start, n, block):
            tr = slice(0, b0 - h + 1)
            te = slice(b0, min(b0 + block, n))
            for cols, out in ((list(range(k_small)), f_s),
                              (list(range(k_small + k_extra)), f_l)):
                Xtr = pd.DataFrame(Xa[tr][:, cols], index=idx[tr],
                                   columns=[f"x__{c}" for c in cols])
                Xte = pd.DataFrame(Xa[te][:, cols], index=idx[te], columns=Xtr.columns)
                m = make_model(model, horizon=h, **prm)
                m.fit(Xtr, pd.Series(y[tr], index=idx[tr]))
                out[te] = np.asarray(m.predict(Xte), dtype=float)
        ok = np.isfinite(f_s) & np.isfinite(f_l)
        dm = diebold_mariano(y[ok] - f_l[ok], y[ok] - f_s[ok], h)
        cw = clark_west(y[ok], f_s[ok], f_l[ok], h)
        dm_rej += int(dm.p_value < 0.05)
        cw_rej += int(cw.p_value < 0.05)
    return {"dm_rejection_rate": dm_rej / n_reps, "cw_rejection_rate": cw_rej / n_reps,
            "n_reps": n_reps, "nominal": 0.05}
