"""Short SYNTHETIC end-to-end run through ``pipelines.synthetic_e2e.run_world``.

Verifies architecture only (synthetic output is never evidence of skill). Uses a temp copy of
``configs/`` with a shorter synthetic window (see conftest ``short_config_dir``); ``configs/`` is
not modified. The null world is run twice into separate stores to test determinism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from ttf_forecaster.data.base import PREDICTION_SCHEMA, validate_frame
from ttf_forecaster.data.storage import ParquetStore, StorageError
from ttf_forecaster.pipelines.synthetic_e2e import WorldResult, run_world

pytestmark = pytest.mark.slow

RUN_TAG = "itest"


@dataclass
class Run:
    result: WorldResult
    store: ParquetStore
    root: Path


def _run(world: str, root: Path, config_dir: Path) -> Run:
    res = run_world(world, data_root=str(root), config_dir=str(config_dir), run_tag=RUN_TAG)
    return Run(res, ParquetStore(root), root)


@pytest.fixture(scope="module")
def null_runs(tmp_path_factory: pytest.TempPathFactory, short_config_dir: Path
              ) -> tuple[Run, Run]:
    a = _run("null", tmp_path_factory.mktemp("e2e_null_a"), short_config_dir)
    b = _run("null", tmp_path_factory.mktemp("e2e_null_b"), short_config_dir)
    return a, b


def _hashes(store: ParquetStore, layer: str, run_id: str) -> dict[str, str]:
    return {k: v.content_sha256 for k, v in store.read_manifest(layer, run_id).artifacts.items()}


def test_artefacts_written_with_synthetic_manifests(null_runs: tuple[Run, Run]) -> None:
    run = null_runs[0]
    rid = run.result.run_id
    assert rid == f"synthetic_null_{RUN_TAG}"
    for layer, name in (("features", "feature_matrix"), ("predictions", "predictions")):
        assert (run.root / layer / rid / "manifest.json").is_file()
        assert (run.root / layer / rid / f"{name}.parquet").is_file()
        man = run.store.read_manifest(layer, rid)
        assert man.is_synthetic is True
        assert man.config_hash, f"{layer} manifest lacks a config hash"
        assert name in man.artifacts
        df = run.store.read_run_artifact(layer, rid, name)  # verifies the content hash
        assert len(df) == man.artifacts[name].n_rows > 0
    # the raw layer keeps the synthetic vendor payloads it was fed
    assert any((run.root / "raw").rglob("*.parquet"))


def test_predictions_validate_against_schema(null_runs: tuple[Run, Run]) -> None:
    run = null_runs[0]
    p = run.store.read_run_artifact("predictions", run.result.run_id, "predictions")
    validate_frame(p, PREDICTION_SCHEMA)
    assert p["is_synthetic"].all()
    models = set(p["model_id"])
    assert {"ridge", "lightgbm", "ridge_price_only", "zero", "rolling_mean_60"} <= models
    # every model is scored on exactly the same (origin, fold) rows as the principal benchmark
    keys = {m: set(map(tuple, g[["origin_time", "fold_id"]].astype(str).to_numpy()))
            for m, g in p.groupby("model_id")}
    assert all(k == keys["zero"] for k in keys.values()), "models scored on different rows"
    # one forecast per (model, origin, target, horizon)
    assert not p.duplicated(["model_id", "origin_time", "target_id", "horizon"]).any()
    # PIT: train_end is the refit time (information cutoff of the fit); never after the origin
    assert (p["train_end"] <= p["origin_time"]).all()
    assert (p["label_known_time"] > p["origin_time"]).all()
    assert p.groupby("model_id")["feature_hash"].nunique().max() == 1


def test_null_world_fails_criterion_1(null_runs: tuple[Run, Run]) -> None:
    res = null_runs[0].result
    assert res.n_origins > 250 and res.n_features > 5
    assert res.criteria, "no criteria evaluated"
    passed = {m: c["criterion_1"]["passed"] for m, c in res.criteria.items()}
    assert not any(passed.values()), f"null world 'passed' criterion 1: {passed}"


def test_two_runs_are_identical(null_runs: tuple[Run, Run]) -> None:
    a, b = null_runs
    rid = a.result.run_id
    for layer, name in (("predictions", "predictions"), ("features", "feature_matrix")):
        pd.testing.assert_frame_equal(a.store.read_run_artifact(layer, rid, name),
                                      b.store.read_run_artifact(layer, rid, name),
                                      check_exact=True)
        assert a.store.read_manifest(layer, rid).config_hash == \
            b.store.read_manifest(layer, rid).config_hash
        assert _hashes(a.store, layer, rid) == _hashes(b.store, layer, rid)
    assert a.result.feature_columns == b.result.feature_columns
    for m in a.result.criteria:
        assert a.result.criteria[m]["criterion_1"] == b.result.criteria[m]["criterion_1"]


def test_rewrite_of_identical_artefact_is_noop_and_different_raises(
        null_runs: tuple[Run, Run]) -> None:
    """Runs are immutable: identical re-write is a no-op; different content under the same name
    raises instead of silently overwriting."""
    a = null_runs[0]
    rid = a.result.run_id
    before = _hashes(a.store, "predictions", rid)
    man = a.store.read_manifest("predictions", rid)
    p = a.store.read_run_artifact("predictions", rid, "predictions")
    cfg_hash = man.config_hash
    # identical content, no config passed: the run keeps its original config hash
    a.store.write_run_artifact("predictions", rid, "predictions", p, is_synthetic=True)
    assert _hashes(a.store, "predictions", rid) == before
    assert a.store.read_manifest("predictions", rid).config_hash == cfg_hash
    changed = p.copy()
    changed.loc[changed.index[0], "prediction"] = 123.0
    with pytest.raises(StorageError):
        a.store.write_run_artifact("predictions", rid, "predictions", changed, is_synthetic=True)
    assert _hashes(a.store, "predictions", rid) == before
