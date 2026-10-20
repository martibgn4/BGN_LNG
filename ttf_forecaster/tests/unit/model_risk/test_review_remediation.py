"""Regression tests for the review findings that had to close before any real-data run (D-031,
D-032): session horizon (F1), penalised-linear alpha rule (F5), executable label wiring (F7),
config_dir threading (SQ-04), single-owner source ids (SQ-05) and the roll splice (SQ-06).

SYNTHETIC data only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ttf_forecaster.evaluation import diagnostics as d
from ttf_forecaster.evaluation.statistical_tests import session_hac_lag
from ttf_forecaster.models.baselines import lag_horizon
from ttf_forecaster.models.interface import grid_search_time_cv, make_model
from ttf_forecaster.settings import load_config
from ttf_forecaster.targets.common import label_overlap_rows, session_horizon

from .conftest import toy_world

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"


# ---- F1: session horizon ------------------------------------------------------------------


def _toy_targets(n: int = 300, h: int = 5) -> pd.DataFrame:
    _, t = toy_world(n=n, h=h)
    return t


def test_t3_session_horizon_is_derived_from_label_overlap() -> None:
    t = _toy_targets(h=5)
    assert label_overlap_rows(t) == 4
    assert session_horizon(t) == 5
    assert session_horizon(t, declared=5) == 5
    assert session_hac_lag(t) == 4          # DM / CW HAC lag = h - 1
    assert lag_horizon(t) == 5


def test_month_based_horizon_is_refused() -> None:
    """A month-count horizon whose labels are known ~21 sessions per month later must not be used
    as a session horizon (T4/T5, D-032 F1)."""
    t = _toy_targets(h=1).copy()
    t["horizon"] = 2                         # a month count
    t["label_known_time"] = t["origin_time"] + pd.Timedelta(days=45)
    assert label_overlap_rows(t) > 2
    with pytest.raises(ValueError, match="month-based horizons"):
        session_horizon(t, declared=2)


def test_diagnostics_reexports_the_targets_implementation() -> None:
    assert d.label_overlap_rows is label_overlap_rows


# ---- F5: penalised linear alpha rule --------------------------------------------------------


def test_ridge_config_has_a_penalty_floor() -> None:
    cfg = load_config("models")
    ridge = cfg["models"]["ridge"]["params"]
    assert ridge["min_alpha"] == pytest.approx(1.0e-2)
    assert ridge["selection"] == "min"   # one_se rejected as default (D-033)
    m = make_model("ridge", horizon=5)
    assert m.params["min_alpha"] == pytest.approx(1.0e-2)
    assert all(a >= 1.0e-2 for a in m._alphas())


def test_min_alpha_drops_grid_values_below_the_floor() -> None:
    m = make_model("ridge", horizon=5, alphas=[1e-4, 1e-2, 1.0], min_alpha=1e-2)
    assert m._alphas() == [1.0, 1e-2]
    with pytest.raises(ValueError, match="min_alpha"):
        make_model("ridge", horizon=5, alphas=[1e-4], min_alpha=1e-2)._alphas()


def test_one_se_picks_the_most_regularised_value_within_one_se() -> None:
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n), index=idx)
    grid = [100.0, 10.0, 1.0, 0.1]          # most -> least regularised

    def fit_score(Xt, yt, Xv, yv, v):  # noqa: ANN001
        # validation loss decreases with less regularisation, plus fold noise
        return float(yv.var(ddof=0)) + 0.5 / (1.0 + v) + 0.0 * float(Xt.shape[0])

    best_min, tab_min = grid_search_time_cv(X, y, grid, fit_score, gap=5, n_splits=3,
                                            val_fraction=0.25)
    best_1se, tab = grid_search_time_cv(X, y, grid, fit_score, gap=5, n_splits=3,
                                        val_fraction=0.25, rule="one_se")
    means = tab["mean_loss"].to_numpy()
    se = float(tab["std_loss"].iloc[int(np.argmin(means))]) / np.sqrt(3)
    threshold = means.min() + se
    expected = grid[int(np.flatnonzero(means <= threshold)[0])]
    assert best_1se == expected
    assert best_1se >= best_min             # never less regularised than the minimum rule
    with pytest.raises(ValueError, match="rule"):
        grid_search_time_cv(X, y, grid, fit_score, gap=5, n_splits=3, val_fraction=0.25,
                            rule="other")


# ---- SQ-04: config_dir is honoured by model construction ------------------------------------


def test_make_model_reads_models_from_config_dir(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    shutil.copytree(CONFIGS, cfg_dir)
    p = cfg_dir / "models.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw["models"]["ridge"]["params"]["alphas"] = [7.0]
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    m = make_model("ridge", horizon=5, config_dir=str(cfg_dir))
    assert m.params["alphas"] == [7.0]
    assert make_model("ridge", horizon=5).params["alphas"] != [7.0]


# ---- SQ-05: one owner per source id; power inputs resolve ----------------------------------


def _source_series() -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for f in sorted((CONFIGS / "sources").glob("*.yaml")):
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for sid in (raw.get("series") or {}):
            owners.setdefault(sid, []).append(f.name)
    return owners


def test_every_source_series_id_has_exactly_one_owner() -> None:
    dup = {k: v for k, v in _source_series().items() if len(v) != 1}
    assert not dup, f"series ids defined in more than one source file: {dup}"


def test_power_price_inputs_resolve_to_a_defined_series() -> None:
    power = yaml.safe_load((CONFIGS / "sources" / "power.yaml").read_text(encoding="utf-8"))
    refs = {k: v for k, v in power["features"]["prices"].items() if isinstance(v, str)}
    owners = _source_series()
    missing = {k: v for k, v in refs.items() if v not in owners}
    assert not missing, f"power.yaml price inputs with no defined series: {missing}"
    assert all(owners[v] == ["bloomberg.yaml"] for v in refs.values()), \
        "power price inputs must be owned by bloomberg.yaml (SQ-05)"


# ---- SQ-06: the price family uses the canonical roll; holidays agree with the calendar ------


def test_price_family_uses_the_canonical_roll_splice() -> None:
    fam = load_config("features")["families"]["price"]["params"]
    assert fam["return_series_provider"] == "ttf_forecaster.contracts.rolls:canonical_front_series"


def test_feature_holidays_match_the_instrument_calendar() -> None:
    feat_h = load_config("features")["common"]["holidays"]
    cal = load_config("instruments")["calendars"]["TTF"]
    assert list(feat_h or []) == list(cal.get("future_holidays") or [])
