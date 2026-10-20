"""Splits: purge, embargo, holdout lock, refusal of random splits, retrain frequency. SYNTHETIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from tests.unit.evaluation.conftest import settle_utc
from ttf_forecaster.evaluation.splits import (
    HoldoutLockedError,
    HoldoutUnlockedWarning,
    NonMonotoneSplitError,
    SplitConfig,
    SplitError,
    as_origin_index,
    holdout_fold,
    inner_folds,
    outer_folds,
    purge_and_embargo,
    refit_points,
    refuse_random_splitter,
    validate_forward_split,
)

H = 5


@pytest.fixture()
def origins() -> pd.DatetimeIndex:
    return settle_utc(pd.bdate_range("2019-01-01", "2024-12-31"))


@pytest.fixture()
def lkt(origins: pd.DatetimeIndex) -> pd.Series:
    """Overlapping h=5 labels: known at the settlement 5 trading days later."""
    tail = pd.DatetimeIndex([origins[-1] + pd.Timedelta(days=7 * k) for k in range(1, H + 1)])
    return pd.Series(origins[H:].append(tail), index=origins)


def test_purge_removes_overlapping_labels(origins, lkt, split_cfg) -> None:
    folds = outer_folds(origins, lkt, H, split_cfg)
    assert len(folds) >= 4
    for f in folds:
        for seg in f.segments:
            assert seg.train_origins.max() < seg.refit_time == seg.predict_origins[0]
            assert (lkt.reindex(seg.train_origins) <= seg.refit_time).all()
        # exactly the h-1 origins just before the block have labels still open at its start
        before = origins[origins < f.test_start]
        assert f.n_purged == H - 1
        assert set(before[-(H - 1):]).isdisjoint(f.train_origins)
        assert before[-H] in f.train_origins


def test_blocks_are_contiguous_six_months(origins, lkt, split_cfg) -> None:
    folds = outer_folds(origins, lkt, H, split_cfg)
    for a, b in zip(folds, folds[1:], strict=False):
        validate_forward_split(origins, a.test_origins[:0], b.test_origins)
        assert a.test_end < b.test_start
        months = (b.test_start.tz_convert("Europe/Amsterdam").tz_localize(None).to_period("M")
                  - a.test_start.tz_convert("Europe/Amsterdam").tz_localize(None).to_period("M")).n
        assert months == 6


def test_holdout_excluded_and_locked(origins, lkt, split_cfg) -> None:
    hstart = pd.Timestamp("2024-07-01", tz="Europe/Amsterdam").tz_convert("UTC")
    folds = outer_folds(origins, lkt, H, split_cfg)
    for f in folds:
        assert (f.test_origins < hstart).all()
        assert (lkt.reindex(f.test_origins) < hstart).all()  # no test label lives in holdout
        for seg in f.segments:
            assert (seg.train_origins < hstart).all()
    with pytest.raises(HoldoutLockedError):
        holdout_fold(origins, lkt, H, split_cfg)
    with pytest.warns(HoldoutUnlockedWarning):
        hf = holdout_fold(origins, lkt, H, split_cfg, unlock_holdout=True)
    assert hf.kind == "holdout" and hf.test_start >= hstart
    assert (lkt.reindex(hf.train_origins) <= hf.test_start).all()


def test_derived_holdout_warns(origins, lkt) -> None:
    with pytest.warns(UserWarning, match="holdout.start is null"):
        outer_folds(origins, lkt, H, SplitConfig(holdout_start=None))


@pytest.mark.parametrize("bad", ["shuffled", "duplicated", "naive"])
def test_non_monotone_origins_refused(origins, bad) -> None:
    if bad == "shuffled":
        idx = origins[np.random.default_rng(0).permutation(len(origins))]
        with pytest.raises(NonMonotoneSplitError):
            as_origin_index(idx)
    elif bad == "duplicated":
        with pytest.raises(NonMonotoneSplitError):
            as_origin_index(origins[:10].append(origins[9:10]))
    else:
        with pytest.raises(SplitError):
            as_origin_index(origins.tz_localize(None))


def test_random_split_refused(origins) -> None:
    rng = np.random.default_rng(1)
    pos = np.sort(rng.choice(len(origins), 200, replace=False))
    with pytest.raises(NonMonotoneSplitError, match="contiguous"):
        validate_forward_split(origins, origins[:0], origins[pos])  # random test sample
    with pytest.raises(NonMonotoneSplitError, match="forward"):
        validate_forward_split(origins, origins[500:600], origins[100:200])  # train after test
    with pytest.raises(NonMonotoneSplitError):
        refuse_random_splitter(KFold(n_splits=5, shuffle=True, random_state=0))
    with pytest.raises(NonMonotoneSplitError):
        refuse_random_splitter(KFold(n_splits=5))
    refuse_random_splitter(TimeSeriesSplit(n_splits=5))


def test_two_sided_purge_and_embargo(origins, lkt) -> None:
    test = origins[1000:1100]
    train = origins[:1000].append(origins[1100:])
    kept, n_purged, n_emb = purge_and_embargo(origins, train, test, lkt, embargo=H)
    assert n_purged == (H - 1) + H  # 4 open labels before; 5 origins before last test label after
    assert n_emb == H
    assert set(origins[996:1000]).isdisjoint(kept)
    assert set(origins[1100:1110]).isdisjoint(kept)
    assert origins[1110] in kept and origins[995] in kept


def test_retrain_frequency_refit_points(origins, lkt) -> None:
    cfg = SplitConfig(min_train_origins=250, holdout_start="2024-07-01",
                      retrain_frequency="monthly")
    f = outer_folds(origins, lkt, H, cfg)[0]
    assert len(f.segments) == 6
    local = pd.DatetimeIndex([s.refit_time for s in f.segments]).tz_convert("Europe/Amsterdam")
    assert local.month.nunique() == 6
    # later refits re-use earlier test origins whose labels are known
    s2 = f.segments[2]
    assert s2.train_origins.max() > f.test_start
    assert (lkt.reindex(s2.train_origins) <= s2.refit_time).all()
    assert len(refit_points(f.test_origins, "daily")) == len(f.test_origins)
    assert len(refit_points(f.test_origins, "quarterly")) == 2


def test_rolling_window(origins, lkt) -> None:
    cfg = SplitConfig(scheme="rolling", rolling_window_years=1, min_train_origins=200,
                      holdout_start="2024-07-01")
    for f in outer_folds(origins, lkt, H, cfg):
        assert f.train_origins[0] >= f.test_start - pd.DateOffset(years=1)


def test_inner_folds_inside_outer_training(origins, lkt, split_cfg) -> None:
    outer = outer_folds(origins, lkt, H, split_cfg)[1]
    inner = inner_folds(outer, lkt, split_cfg)
    assert len(inner) == 3
    n_val = sum(len(f.test_origins) for f in inner)
    assert n_val == int(len(outer.train_origins) * 0.25)
    for f in inner:
        assert f.test_origins.isin(outer.train_origins).all()
        assert (lkt.reindex(f.test_origins) <= outer.test_start).all()
        assert (lkt.reindex(f.train_origins) <= f.test_start).all()


def test_config_from_yaml() -> None:
    cfg = SplitConfig.from_config()
    assert cfg.scheme == "expanding" and cfg.test_block_months == 6
    assert cfg.inner_validation_fraction == 0.25
