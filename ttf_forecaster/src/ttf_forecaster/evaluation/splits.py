"""Walk-forward splits with purging, embargo, inner folds and a locked holdout.

Implements ``research_specification.md`` §7 and ``point_in_time_policy.md`` §6. Every split here is
**time-ordered**; random K-fold, shuffled indices and interleaved train/test sets are refused with
``NonMonotoneSplitError``.

Vocabulary
----------
* **origin** - a forecast origin time (tz-aware UTC; the close of a trading date after settlement
  publication). Splits are over origins of ONE target x horizon.
* **label_known_time** - when the realised label of an origin becomes known (``TARGET_SCHEMA``).
* **refit time** - the as-of time of a model fit: the first origin it predicts. A row may train a
  model refitted at ``r`` only if ``origin < r`` **and** ``label_known_time <= r`` (the purge).
* **outer fold** - one contiguous block of test origins (6 months by default) plus the training
  rows available at each refit point inside it (``RefitSegment``).
* **inner fold** - a walk-forward split *inside* an outer fold's training set, for tuning only.
* **embargo** - removal of the ``n`` origins immediately after a test block from any training set
  that lies *after* that block (de Prado). In a strictly forward walk-forward, training never lies
  after the block being evaluated, so the embargo removes nothing there; it is enforced by
  ``purge_and_embargo`` for any two-sided split and verified in tests.

Retraining frequency
--------------------
``block`` refits once at the start of each test block (spec default). ``daily`` / ``weekly`` /
``monthly`` / ``quarterly`` refit at the first origin of each such local (Europe/Amsterdam) period
within the block; earlier test origins of the same block re-enter training once their labels are
known at the refit time, exactly as they would live.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.settings import ConfigError, load_config

log = get_logger(__name__)

SCHEMES: tuple[str, ...] = ("expanding", "rolling")
RETRAIN_FREQUENCIES: tuple[str, ...] = ("block", "daily", "weekly", "monthly", "quarterly")
FOLD_KINDS: tuple[str, ...] = ("outer", "inner", "holdout")


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class SplitError(ValueError):
    """A split cannot be built or violates the validation design."""


class NonMonotoneSplitError(SplitError):
    """A split is shuffled, interleaved, non-contiguous or otherwise not time-ordered."""


class HoldoutLockedError(SplitError):
    """Evaluation on the locked final holdout was requested without ``unlock_holdout=True``."""


class HoldoutUnlockedWarning(UserWarning):
    """The locked final holdout has been unlocked. Its single evaluation must be reported as is."""


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Split parameters (``configs/backtests.yaml`` sections ``splits`` and ``holdout``)."""

    scheme: str = "expanding"
    rolling_window_years: float = 3.0
    test_block_months: int = 6
    first_test_start: str | None = None
    min_train_origins: int = 250
    min_test_origins: int = 20
    embargo_trading_days: int | None = None
    retrain_frequency: str = "block"
    inner_validation_fraction: float = 0.25
    inner_n_blocks: int = 3
    holdout_start: str | None = None
    holdout_months: int = 12

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ConfigError(f"splits.scheme {self.scheme!r} not in {SCHEMES}")
        if self.retrain_frequency not in RETRAIN_FREQUENCIES:
            raise ConfigError(
                f"splits.retrain_frequency {self.retrain_frequency!r} not in {RETRAIN_FREQUENCIES}"
            )
        if self.test_block_months < 1:
            raise ConfigError("splits.test_block_months must be >= 1")
        if not 0.0 < self.inner_validation_fraction < 1.0:
            raise ConfigError("splits.inner.validation_fraction must be in (0, 1)")
        if self.inner_n_blocks < 1:
            raise ConfigError("splits.inner.n_blocks must be >= 1")
        if self.embargo_trading_days is not None and self.embargo_trading_days < 0:
            raise ConfigError("splits.embargo_trading_days must be >= 0 or null")
        if self.holdout_months < 1:
            raise ConfigError("holdout.months must be >= 1")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None) -> SplitConfig:
        """Build from the ``backtests`` config mapping (loaded from YAML when ``cfg`` is None)."""
        cfg = load_config("backtests") if cfg is None else cfg
        s = dict(cfg.get("splits", {}))
        inner = dict(s.pop("inner", {}) or {})
        h = dict(cfg.get("holdout", {}) or {})
        known = {f for f in cls.__dataclass_fields__}
        kwargs: dict[str, Any] = {k: v for k, v in s.items() if k in known}
        unknown = sorted(set(s) - known)
        if unknown:
            raise ConfigError(f"Unknown keys in backtests.splits: {unknown}")
        if "validation_fraction" in inner:
            kwargs["inner_validation_fraction"] = float(inner["validation_fraction"])
        if "n_blocks" in inner:
            kwargs["inner_n_blocks"] = int(inner["n_blocks"])
        if "start" in h:
            kwargs["holdout_start"] = None if h["start"] is None else str(h["start"])
        if "months" in h:
            kwargs["holdout_months"] = int(h["months"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RefitSegment:
    """One model fit inside a fold: train on ``train_origins``, predict ``predict_origins``.

    Invariants (checked at construction by the builders, re-checked by ``walk_forward``):
    ``max(train_origins) < refit_time == min(predict_origins)`` and every training row has
    ``label_known_time <= refit_time``.
    """

    refit_time: pd.Timestamp
    train_origins: pd.DatetimeIndex
    predict_origins: pd.DatetimeIndex


@dataclass(frozen=True, eq=False)
class Fold:
    """A contiguous test block of origins and the refit segments that predict it."""

    fold_id: str
    kind: str
    test_origins: pd.DatetimeIndex
    segments: tuple[RefitSegment, ...]
    n_purged: int
    n_embargoed: int
    horizon: int
    parent_fold_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in FOLD_KINDS:
            raise SplitError(f"Fold kind {self.kind!r} not in {FOLD_KINDS}")

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_origins[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_origins[-1]

    @property
    def train_origins(self) -> pd.DatetimeIndex:
        """Training origins of the FIRST refit (the fold's 'training set' for inner folds)."""
        return self.segments[0].train_origins

    def describe(self) -> dict[str, Any]:
        """Flat summary for logs and reports."""
        return {
            "fold_id": self.fold_id,
            "kind": self.kind,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_test": len(self.test_origins),
            "n_train_first_refit": len(self.train_origins),
            "n_refits": len(self.segments),
            "n_purged": self.n_purged,
            "n_embargoed": self.n_embargoed,
        }


# ---------------------------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------------------------


def as_origin_index(origins: Sequence[Any] | pd.Index, name: str = "origins") -> pd.DatetimeIndex:
    """Return ``origins`` as a tz-aware UTC DatetimeIndex; refuse unsorted or duplicated input.

    Raises:
        SplitError: naive or non-UTC timestamps.
        NonMonotoneSplitError: not strictly increasing (shuffled, duplicated).
    """
    idx = pd.DatetimeIndex(origins)
    if idx.tz is None:
        raise SplitError(f"{name}: timestamps must be tz-aware UTC, got naive")
    if str(idx.tz) != "UTC":
        raise SplitError(f"{name}: timestamps must be UTC, got {idx.tz}")
    if len(idx) > 1 and not (np.diff(idx.asi8) > 0).all():
        raise NonMonotoneSplitError(
            f"{name}: origins must be strictly increasing (no shuffling, no duplicates)"
        )
    return idx


def validate_forward_split(
    universe: pd.DatetimeIndex, train: pd.DatetimeIndex, test: pd.DatetimeIndex
) -> None:
    """Refuse any split that is not a forward, contiguous, time-ordered split.

    Checks: both sides strictly increasing and inside ``universe``; ``test`` non-empty and
    contiguous within ``universe`` (no gaps, i.e. not a random sample); every training origin
    strictly before the first test origin.

    Raises:
        NonMonotoneSplitError: on any violation, naming it.
    """
    universe = as_origin_index(universe, "universe")
    train = as_origin_index(train, "train")
    test = as_origin_index(test, "test")
    if len(test) == 0:
        raise SplitError("test set is empty")
    missing = (~train.isin(universe)).sum() + (~test.isin(universe)).sum()
    if missing:
        raise SplitError(f"{int(missing)} split origins are not in the origin universe")
    pos = universe.get_indexer(test)
    if not (np.diff(pos) == 1).all():
        raise NonMonotoneSplitError(
            "test origins are not a contiguous block of the origin universe (random/interleaved "
            "test sets are forbidden)"
        )
    if len(train) and train[-1] >= test[0]:
        raise NonMonotoneSplitError(
            f"training origin {train[-1]} is not before the first test origin {test[0]}: "
            "forward splits only (random K-fold is forbidden)"
        )


def refuse_random_splitter(splitter: object) -> None:
    """Refuse sklearn-style splitters that shuffle or are not time-series splitters.

    Call this on any externally supplied splitter object before use. Only objects whose class name
    contains ``TimeSeries`` and that do not shuffle are accepted; this module's own folds are
    preferred in all cases.
    """
    if getattr(splitter, "shuffle", False):
        raise NonMonotoneSplitError(f"{type(splitter).__name__} shuffles: forbidden")
    if "TimeSeries" not in type(splitter).__name__:
        raise NonMonotoneSplitError(
            f"{type(splitter).__name__} is not a time-ordered splitter; use evaluation.splits"
        )


def _lkt_for(origins: pd.DatetimeIndex, label_known_time: pd.Series) -> pd.Series:
    lkt = label_known_time.reindex(origins)
    if lkt.isna().any():
        raise SplitError(f"label_known_time missing for {int(lkt.isna().sum())} origins")
    return lkt


def purge(
    train: pd.DatetimeIndex, label_known_time: pd.Series, cutoff: pd.Timestamp
) -> pd.DatetimeIndex:
    """Training origins whose label is known at ``cutoff`` (``label_known_time <= cutoff``)."""
    lkt = _lkt_for(train, label_known_time)
    return train[(lkt <= cutoff).to_numpy()]


def purge_and_embargo(
    universe: pd.DatetimeIndex,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    label_known_time: pd.Series,
    embargo: int,
) -> tuple[pd.DatetimeIndex, int, int]:
    """General two-sided purge + embargo (de Prado), for splits with training on both sides.

    * Training rows **before** the test block are purged if their label is not known at the first
      test origin (``label_known_time > test[0]``): their label window overlaps the test period.
    * Training rows **after** the test block are purged if their origin falls before the last test
      label becomes known (``origin <= max(label_known_time[test])``), then the next ``embargo``
      universe origins after the purged zone are dropped as well (serial correlation of features).

    Returns:
        (kept training origins, n_purged, n_embargoed).
    """
    universe = as_origin_index(universe, "universe")
    train = as_origin_index(train, "train")
    test = as_origin_index(test, "test")
    if train.isin(test).any():
        raise NonMonotoneSplitError("train and test overlap")
    before = train[train < test[0]]
    after = train[train > test[-1]]
    kept_before = purge(before, label_known_time, test[0])
    test_label_end = _lkt_for(test, label_known_time).max()
    after_purged = after[after > test_label_end]
    n_purged = (len(before) - len(kept_before)) + (len(after) - len(after_purged))
    later = universe[universe > max(test[-1], test_label_end)]
    embargo_zone = later[:embargo]
    kept_after = after_purged[~after_purged.isin(embargo_zone)]
    n_embargoed = len(after_purged) - len(kept_after)
    return kept_before.append(kept_after), int(n_purged), int(n_embargoed)


# ---------------------------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------------------------


def _local_date(ts: pd.Timestamp | str) -> pd.Timestamp:
    """A config date string or tz-aware timestamp -> local (Europe/Amsterdam) midnight in UTC."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.normalize().tz_localize(MARKET_TZ).tz_convert("UTC")
    return t.tz_convert(MARKET_TZ).normalize().tz_convert("UTC")


def refit_points(test_origins: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Refit times inside a test block: first origin of each local period of ``frequency``."""
    if frequency not in RETRAIN_FREQUENCIES:
        raise ConfigError(f"retrain frequency {frequency!r} not in {RETRAIN_FREQUENCIES}")
    test_origins = as_origin_index(test_origins, "test")
    if frequency == "block" or len(test_origins) == 0:
        return test_origins[:1]
    if frequency == "daily":
        return test_origins
    local = test_origins.tz_convert(MARKET_TZ)
    if frequency == "weekly":
        iso = local.isocalendar()
        key = iso["year"].astype(int).to_numpy() * 100 + iso["week"].astype(int).to_numpy()
    elif frequency == "monthly":
        key = local.year.to_numpy() * 100 + local.month.to_numpy()
    else:  # quarterly
        key = local.year.to_numpy() * 10 + local.quarter.to_numpy()
    first = np.r_[True, key[1:] != key[:-1]]
    return test_origins[first]


def _segments(
    universe: pd.DatetimeIndex,
    base_train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    label_known_time: pd.Series,
    frequency: str,
    window_start: pd.Timestamp | None,
) -> tuple[tuple[RefitSegment, ...], int]:
    """Refit segments for a test block. Training at refit r = base_train + earlier test origins,
    restricted to ``origin >= window_start`` (rolling) and purged on ``label_known_time <= r``."""
    points = refit_points(test, frequency)
    segs: list[RefitSegment] = []
    n_purged_first = 0
    for i, r in enumerate(points):
        nxt = points[i + 1] if i + 1 < len(points) else None
        pred = test[(test >= r) & (test < nxt)] if nxt is not None else test[test >= r]
        candidates = base_train.append(test[test < r])
        if window_start is not None:
            candidates = candidates[candidates >= window_start]
        kept = purge(candidates, label_known_time, r)
        if i == 0:
            n_purged_first = len(candidates) - len(kept)
        validate_forward_split(universe, kept, pred)
        segs.append(RefitSegment(refit_time=r, train_origins=kept, predict_origins=pred))
    return tuple(segs), n_purged_first


# ---------------------------------------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------------------------------------


def holdout_start(origins: pd.DatetimeIndex, cfg: SplitConfig) -> pd.Timestamp:
    """Start of the locked holdout (UTC). Fixed config date, else last origin - ``holdout_months``.

    A derived start moves as data accrues; it is logged and warned as not a valid final lock.
    """
    origins = as_origin_index(origins)
    if cfg.holdout_start is not None:
        return _local_date(cfg.holdout_start)
    derived = _local_date(origins[-1] - pd.DateOffset(months=cfg.holdout_months))
    msg = (
        f"holdout.start is null; derived holdout start {derived.date()} from the data range. "
        "Fix holdout.start in configs/backtests.yaml before any final claim."
    )
    log.warning("holdout_start_derived", extra={"fields": {"holdout_start": str(derived)}})
    warnings.warn(msg, UserWarning, stacklevel=2)
    return derived


def holdout_fold(
    origins: pd.DatetimeIndex,
    label_known_time: pd.Series,
    horizon: int,
    cfg: SplitConfig | None = None,
    *,
    unlock_holdout: bool = False,
) -> Fold:
    """The single evaluation fold on the locked holdout. Refused unless ``unlock_holdout=True``.

    Training uses every pre-holdout origin whose label is known at each refit time (same refit
    frequency as the outer folds).

    Raises:
        HoldoutLockedError: if not unlocked.
    """
    cfg = cfg or SplitConfig.from_config()
    if not unlock_holdout:
        raise HoldoutLockedError(
            "The final holdout is locked until the model specification is frozen and signed off "
            "(research_specification.md §7). Pass unlock_holdout=True to evaluate it ONCE."
        )
    origins = as_origin_index(origins)
    start = holdout_start(origins, cfg)
    log.warning("holdout_unlocked", extra={"fields": {"holdout_start": str(start)}})
    warnings.warn(
        f"Locked holdout (origins >= {start}) unlocked: report this single evaluation as is.",
        HoldoutUnlockedWarning,
        stacklevel=2,
    )
    test = origins[origins >= start]
    if len(test) == 0:
        raise SplitError("holdout contains no origins")
    base = origins[origins < start]
    window = _rolling_window_start(test[0], cfg)
    segs, n_purged = _segments(
        origins, base, test, label_known_time, cfg.retrain_frequency, window
    )
    return Fold("holdout", "holdout", test, segs, n_purged, 0, horizon)


# ---------------------------------------------------------------------------------------------
# Outer and inner folds
# ---------------------------------------------------------------------------------------------


def embargo_length(horizon: int, cfg: SplitConfig | None = None) -> int:
    """Embargo in trading days: ``splits.embargo_trading_days`` or, when null, the horizon."""
    cfg = cfg or SplitConfig.from_config()
    return horizon if cfg.embargo_trading_days is None else cfg.embargo_trading_days


def _rolling_window_start(test_start: pd.Timestamp, cfg: SplitConfig) -> pd.Timestamp | None:
    if cfg.scheme == "expanding":
        return None
    months = int(round(cfg.rolling_window_years * 12))
    return test_start - pd.DateOffset(months=months)


def _first_block_start(origins: pd.DatetimeIndex, cfg: SplitConfig) -> pd.Timestamp:
    if cfg.first_test_start is not None:
        return _local_date(cfg.first_test_start)
    if len(origins) <= cfg.min_train_origins:
        raise SplitError(
            f"{len(origins)} origins available; need more than min_train_origins="
            f"{cfg.min_train_origins} before the first test block"
        )
    after = origins[cfg.min_train_origins].tz_convert(MARKET_TZ)
    # Align to the next half-year boundary (1 Jan / 1 Jul) so blocks are calendar half-years.
    year, month = after.year, after.month
    if month < 7:
        aligned = pd.Timestamp(year=year, month=7, day=1)
    else:
        aligned = pd.Timestamp(year=year + 1, month=1, day=1)
    return _local_date(aligned)


def outer_folds(
    origins: pd.DatetimeIndex,
    label_known_time: pd.Series,
    horizon: int,
    cfg: SplitConfig | None = None,
) -> list[Fold]:
    """Outer walk-forward folds over pre-holdout origins of one target x horizon.

    Args:
        origins: candidate origins (strictly increasing, tz-aware UTC), typically the origins with
            a non-NaN label and a feature row.
        label_known_time: Series indexed by origin (``TARGET_SCHEMA.label_known_time``).
        horizon: target horizon in trading days; default embargo length.
        cfg: split configuration (YAML when None).

    Returns:
        Folds in time order, ids ``outer_00``, ``outer_01``... Test origins whose label is not known
        before the holdout start are dropped from evaluation (their labels live in the holdout).
    """
    cfg = cfg or SplitConfig.from_config()
    if horizon < 1:
        raise SplitError("horizon must be >= 1 trading day")
    origins = as_origin_index(origins)
    hstart = holdout_start(origins, cfg)
    pre = origins[origins < hstart]
    lkt_pre = _lkt_for(pre, label_known_time)
    pre = pre[(lkt_pre < hstart).to_numpy()]
    start = _first_block_start(pre, cfg)
    folds: list[Fold] = []
    k = 0
    while start <= pre[-1]:
        end = _local_date(start.tz_convert(MARKET_TZ).tz_localize(None)
                          + pd.DateOffset(months=cfg.test_block_months))
        test = pre[(pre >= start) & (pre < end)]
        start = end
        if len(test) < cfg.min_test_origins:
            continue
        window = _rolling_window_start(test[0], cfg)
        base = pre[pre < test[0]]
        segs, n_purged = _segments(
            origins, base, test, label_known_time, cfg.retrain_frequency, window
        )
        if len(segs[0].train_origins) < cfg.min_train_origins:
            log.info(
                "fold_skipped_short_train",
                extra={"fields": {"test_start": str(test[0]), "n": len(segs[0].train_origins)}},
            )
            continue
        # Forward split: no training row lies after this block, so the embargo removes nothing.
        folds.append(Fold(f"outer_{k:02d}", "outer", test, segs, n_purged, 0, horizon))
        k += 1
    if not folds:
        raise SplitError("no outer fold could be built; check data length and split config")
    return folds


def inner_folds(
    outer: Fold,
    label_known_time: pd.Series,
    cfg: SplitConfig | None = None,
) -> list[Fold]:
    """Walk-forward tuning folds inside ``outer``'s first-refit training set.

    The last ``inner_validation_fraction`` of the training origins is cut into ``inner_n_blocks``
    contiguous validation blocks; each is predicted by one fit on the earlier training origins whose
    labels are known at the block start. Validation labels are all known at the outer refit time
    (the outer purge guarantees it), so tuning never sees the outer test period.
    """
    cfg = cfg or SplitConfig.from_config()
    if outer.kind != "outer":
        raise SplitError("inner folds are built only inside outer folds")
    train = outer.train_origins
    n_val = int(np.floor(len(train) * cfg.inner_validation_fraction))
    if n_val < cfg.inner_n_blocks:
        raise SplitError(f"outer fold {outer.fold_id}: too few training rows for inner folds")
    val = train[len(train) - n_val:]
    blocks = np.array_split(np.arange(n_val), cfg.inner_n_blocks)
    folds: list[Fold] = []
    for j, b in enumerate(blocks):
        test = val[b]
        base = train[train < test[0]]
        kept = purge(base, label_known_time, test[0])
        validate_forward_split(train, kept, test)
        seg = RefitSegment(refit_time=test[0], train_origins=kept, predict_origins=test)
        folds.append(
            Fold(
                f"{outer.fold_id}_inner_{j}",
                "inner",
                test,
                (seg,),
                len(base) - len(kept),
                0,
                outer.horizon,
                parent_fold_id=outer.fold_id,
            )
        )
    return folds


def make_splits(
    targets: pd.DataFrame,
    horizon: int | None = None,
    cfg: SplitConfig | None = None,
    valid_origins: pd.DatetimeIndex | None = None,
) -> list[Fold]:
    """Outer folds straight from a TARGET_SCHEMA frame of one target x horizon.

    Origins with a NaN label are excluded. ``valid_origins`` (e.g. the feature matrix index)
    further restricts the universe.
    """
    if targets[["target_id", "horizon"]].drop_duplicates().shape[0] != 1:
        raise SplitError("make_splits expects exactly one (target_id, horizon)")
    h = int(targets["horizon"].iloc[0]) if horizon is None else horizon
    t = targets.loc[targets["value"].notna()].sort_values("origin_time")
    origins = pd.DatetimeIndex(t["origin_time"])
    if valid_origins is not None:
        origins = origins[origins.isin(valid_origins)]
    lkt = pd.Series(t["label_known_time"].to_numpy(), index=pd.DatetimeIndex(t["origin_time"]))
    return outer_folds(origins, lkt, h, cfg)
