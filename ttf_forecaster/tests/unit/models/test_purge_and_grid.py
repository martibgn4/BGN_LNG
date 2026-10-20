"""Mandatory horizon, label_known_time purging of every internal split, and origin-grid checks.

SYNTHETIC data only. ``label_known_time`` here mimics T5 (labels known ~60 days after the origin,
far beyond the 5-row gap), so a row gap alone would leak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.models.conftest import linear_world, origin_index
from ttf_forecaster.models import interface
from ttf_forecaster.models.baselines import ZeroReturn
from ttf_forecaster.models.interface import inner_time_splits, make_model, tail_split
from ttf_forecaster.settings import ConfigError

LAG = pd.Timedelta(days=60)


def _t5_like(n: int = 700) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = linear_world(n=n)
    return X, y, pd.Series(X.index + LAG, index=X.index)


def _ok(train: np.ndarray, first_val: int, lkt: pd.Series, origins: pd.Index) -> bool:
    return bool((lkt.iloc[train] <= origins[first_val]).all())


# ---------------------------------------------------------------------------------------------
# 1. horizon is mandatory
# ---------------------------------------------------------------------------------------------


def test_make_model_without_horizon_fails() -> None:
    with pytest.raises(TypeError, match="horizon"):
        make_model("ridge")
    with pytest.raises(TypeError, match="horizon"):
        ZeroReturn()
    assert make_model("ridge", horizon=20).horizon == 20


def test_config_defaults_cannot_supply_horizon(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text("defaults: {horizon: 5}\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="horizon must be explicit"):
        make_model("ridge", config_dir=str(tmp_path))


def test_config_entry_with_explicit_horizon_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        "models:\n  ridge_h20: {model: ridge, params: {horizon: 20}}\n", encoding="utf-8")
    assert make_model("ridge_h20", config_dir=str(tmp_path)).horizon == 20


# ---------------------------------------------------------------------------------------------
# 2. label_known_time purge
# ---------------------------------------------------------------------------------------------


def test_inner_splits_purge_by_label_time() -> None:
    X, _, lkt = _t5_like()
    plain = inner_time_splits(len(X), gap=5)
    assert any(not _ok(tr, va[0], lkt, X.index) for tr, va in plain)  # row gap alone leaks
    purged = inner_time_splits(len(X), gap=5, label_known_time=lkt, origins=X.index)
    assert all(_ok(tr, va[0], lkt, X.index) for tr, va in purged)
    for (tr0, _), (tr1, _) in zip(plain, purged, strict=True):
        assert len(tr1) < len(tr0)


def test_tail_split_purge_by_label_time() -> None:
    X, _, lkt = _t5_like()
    head, tail = tail_split(len(X), fraction=0.25, gap=5)
    assert not _ok(head, tail[0], lkt, X.index)
    head, tail = tail_split(len(X), fraction=0.25, gap=5, label_known_time=lkt, origins=X.index)
    assert _ok(head, tail[0], lkt, X.index)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, Any, Any]]:
    """Record every inner_time_splits / tail_split call made during a fit."""
    calls: list[tuple[str, Any, Any, Any]] = []
    real_inner, real_tail = interface.inner_time_splits, interface.tail_split

    def inner(*a: Any, **k: Any) -> Any:
        out = real_inner(*a, **k)
        calls.append(("inner", out, k.get("label_known_time"), k.get("origins")))
        return out

    def tail(*a: Any, **k: Any) -> Any:
        out = real_tail(*a, **k)
        calls.append(("tail", [out], k.get("label_known_time"), k.get("origins")))
        return out

    monkeypatch.setattr(interface, "inner_time_splits", inner)
    monkeypatch.setattr(interface, "tail_split", tail)
    return calls


CASES = [
    ("ridge", {}),
    ("lasso", {}),
    ("lightgbm", {}),
    ("lightgbm_quantile", {"levels": [0.1, 0.5, 0.9]}),
    ("conformal", {}),
    ("threshold_logistic", {"threshold": 0.0}),
    ("lightgbm_event", {"threshold": 0.0}),
]


@pytest.mark.parametrize("name,params", CASES)
def test_every_internal_split_is_purged(name: str, params: dict[str, Any],
                                        recorder: list[tuple[str, Any, Any, Any]]) -> None:
    X, y, lkt = _t5_like()
    make_model(name, horizon=5, **params).fit(X, y, label_known_time=lkt)
    assert recorder, f"{name} made no internal split"
    for kind, splits, lk, origins in recorder:
        assert lk is not None, f"{name}: a {kind} split ran without label_known_time"
        for tr, va in splits:
            assert _ok(tr, va[0], lk, origins), f"{name}: {kind} split trains on unknown labels"


def test_conformal_passes_label_time_to_base() -> None:
    X, y, lkt = _t5_like()
    m = make_model("conformal", horizon=5).fit(X, y, label_known_time=lkt)
    assert m.base_.fit_label_known_time_ is not None
    assert m.base_.train_end_ + LAG <= X.index[len(X) - len(m.residuals_)]


def test_fit_without_label_time_keeps_row_gap(recorder: list[tuple[str, Any, Any, Any]]) -> None:
    X, y, _ = _t5_like()
    make_model("ridge", horizon=5).fit(X, y)
    assert all(lk is None for _, _, lk, _ in recorder)


def test_label_known_time_validation() -> None:
    X, y, lkt = _t5_like(200)
    m = make_model("zero_return", horizon=5)
    with pytest.raises(ValueError, match="share X's index"):
        m.fit(X, y, label_known_time=lkt.iloc[1:])
    with pytest.raises(ValueError, match="earlier than its origin"):
        m.fit(X, y, label_known_time=lkt - 2 * LAG)
    with pytest.raises(TypeError, match="tz-aware"):
        m.fit(X, y, label_known_time=lkt.dt.tz_localize(None))


# ---------------------------------------------------------------------------------------------
# 3. origin grid must be one row per trading day
# ---------------------------------------------------------------------------------------------


def _frame(idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    return (pd.DataFrame({"a": rng.normal(size=len(idx))}, index=idx),
            pd.Series(rng.normal(size=len(idx)), index=idx))


def test_calendar_day_grid_refused_without_label_time() -> None:
    idx = pd.date_range("2022-01-03 18:00", periods=200, freq="D", tz="UTC")
    X, y = _frame(idx)
    with pytest.raises(ValueError, match="weekends"):
        make_model("zero_return", horizon=5).fit(X, y)
    with pytest.warns(UserWarning, match="not one row per trading day"):
        make_model("zero_return", horizon=5).fit(
            X, y, label_known_time=pd.Series(idx + LAG, index=idx))


def test_intraday_grid_refused() -> None:
    days = origin_index(100)
    idx = days.append(days + pd.Timedelta(hours=2)).sort_values()
    X, y = _frame(idx)
    with pytest.raises(ValueError, match="more than one row per trading day"):
        make_model("zero_return", horizon=1).fit(X, y)


def test_holiday_gaps_are_fine() -> None:
    idx = origin_index(300)
    idx = idx.delete([10, 11, 50, 120])  # missing trading days only make the row gap safer
    X, y = _frame(idx)
    make_model("zero_return", horizon=5).fit(X, y)


def test_non_datetime_index_refused() -> None:
    X, y = _frame(origin_index(50))
    X.index = y.index = pd.RangeIndex(50)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        make_model("zero_return", horizon=1).fit(X, y)
