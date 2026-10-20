"""ParquetStore: raw / normalised / run-artefact layers. SYNTHETIC data only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    OBSERVATION_SCHEMA,
    PREDICTION_SCHEMA,
    SchemaError,
    validate_frame,
)
from ttf_forecaster.data.storage import (
    ParquetStore,
    PathOutsideRootError,
    Store,
    StorageError,
    SyntheticLabelError,
    VintageConflictError,
    frame_content_hash,
)
from ttf_forecaster.features.point_in_time import resolve_as_of, resolve_forecasts_as_of
from ttf_forecaster.settings import Paths

from .helpers import obs_frame

T0 = pd.Timestamp("2026-10-16 08:00", tz="UTC")


@pytest.fixture
def store(tmp_path) -> ParquetStore:
    return ParquetStore(Paths(tmp_path / "data"))


@pytest.fixture(scope="module")
def small_cfg() -> synthetic.SyntheticConfig:
    return synthetic.SyntheticConfig(start="2023-01-02", end="2023-03-31")


class TestProtocolAndLayout:
    def test_is_a_store(self, store: ParquetStore) -> None:
        assert isinstance(store, Store)

    def test_layers_created(self, store: ParquetStore) -> None:
        for layer in ("raw", "normalised", "features", "predictions", "benchmarks"):
            assert (store.root / layer).is_dir()

    def test_accepts_plain_path(self, tmp_path) -> None:
        s = ParquetStore(tmp_path / "d2")
        assert s.root == (tmp_path / "d2").resolve()


class TestPathSafety:
    @pytest.mark.parametrize("bad", ["..", "../x", "a/b", "a\\b", "", ".hidden", "C:", "x" * 300])
    def test_unsafe_source_refused(self, store: ParquetStore, bad: str) -> None:
        with pytest.raises(PathOutsideRootError):
            store.write_raw(bad, "p", b"x", ingestion_time=T0)

    def test_unsafe_run_id_refused(self, store: ParquetStore) -> None:
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(PathOutsideRootError):
            store.write_run_artifact("features", "../escape", "m", df, is_synthetic=True)

    def test_unknown_layer_refused(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="Run layer"):
            store.write_run_artifact("raw", "r1", "m", pd.DataFrame({"a": [1.0]}),
                                     is_synthetic=True)


class TestRaw:
    @pytest.mark.parametrize(
        ("payload", "ext"),
        [(b"\x00\x01", "bin"), ("csv,text\n1,2", "txt"), ({"b": 1, "a": [1, 2]}, "json"),
         ([1, 2, 3], "json")],
    )
    def test_payload_types(self, store: ParquetStore, payload, ext: str) -> None:
        rec = store.write_raw("synthetic", "payload", payload, ingestion_time=T0,
                              metadata={"req": "x"})
        assert rec.written and rec.path.suffix == f".{ext}"
        assert "ingest_date=2026-10-16" in rec.path.parent.name
        assert rec.path.parent.parent.name == "synthetic"
        meta = json.loads(rec.path.with_suffix(rec.path.suffix + ".meta.json").read_text())
        assert meta["sha256"] == rec.sha256 and meta["metadata"] == {"req": "x"}

    def test_frame_payload_round_trips(self, store: ParquetStore) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        rec = store.write_raw("synthetic", "frame", df, ingestion_time=T0)
        back = pd.read_parquet(rec.path)
        pd.testing.assert_frame_equal(back, df)

    def test_identical_payload_not_rewritten(self, store: ParquetStore) -> None:
        a = store.write_raw("synthetic", "p", b"same", ingestion_time=T0)
        b = store.write_raw("synthetic", "p", b"same", ingestion_time=T0 + pd.Timedelta(hours=1))
        assert a.written and not b.written and a.path == b.path

    def test_different_payload_new_file(self, store: ParquetStore) -> None:
        a = store.write_raw("synthetic", "p", b"one", ingestion_time=T0)
        b = store.write_raw("synthetic", "p", b"two", ingestion_time=T0)
        assert a.path != b.path and a.path.exists() and b.path.exists()

    def test_naive_ingestion_time_refused(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="naive"):
            store.write_raw("synthetic", "p", b"x", ingestion_time=pd.Timestamp("2026-01-01"))

    def test_unsupported_payload(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="Unsupported"):
            store.write_raw("synthetic", "p", 3.14, ingestion_time=T0)  # type: ignore[arg-type]


class TestNormalised:
    def test_round_trip_validates(self, store: ParquetStore, small_cfg) -> None:
        obs = synthetic.storage_observations(small_cfg)
        res = store.append_normalised(obs, "observation")
        assert res.n_written == len(obs) and res.n_duplicates_skipped == 0
        back = store.read_normalised("observation")
        validate_frame(back, OBSERVATION_SCHEMA)
        key = ["series_id", "event_time", "revision", "publication_time"]
        exp = obs.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(back[exp.columns], exp)

    def test_reingest_is_idempotent(self, store: ParquetStore, small_cfg) -> None:
        obs = synthetic.storage_observations(small_cfg)
        store.append_normalised(obs, "observation")
        n_files = len(list((store.root / "normalised").rglob("*.parquet")))
        res = store.append_normalised(obs, "observation")
        assert res.n_written == 0 and res.n_duplicates_skipped == len(obs) and res.files == ()
        assert len(list((store.root / "normalised").rglob("*.parquet"))) == n_files
        assert len(store.read_normalised("observation")) == len(obs)

    def test_within_batch_exact_duplicates_collapse(self, store: ParquetStore) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)] * 3)
        res = store.append_normalised(df, "observation")
        assert res.n_written == 1 and res.n_duplicates_skipped == 2

    def test_new_vintage_appends(self, store: ParquetStore) -> None:
        store.append_normalised(obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)]),
                                "observation")
        store.append_normalised(obs_frame([("s.eu.x_twh", "2024-01-01", 1.5, "2024-01-05", 1)]),
                                "observation")
        back = store.read_normalised("observation")
        assert back["revision"].tolist() == [0, 1]
        # PIT still works on the stored layer
        early = resolve_as_of(back, pd.Timestamp("2024-01-03", tz="UTC"))
        late = resolve_as_of(back, pd.Timestamp("2024-01-06", tz="UTC"))
        assert early["value"].tolist() == [1.0] and late["value"].tolist() == [1.5]

    def test_conflicting_redelivery_raises(self, store: ParquetStore) -> None:
        store.append_normalised(obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)]),
                                "observation")
        changed = obs_frame([("s.eu.x_twh", "2024-01-01", 9.0, "2024-01-02", 0)])
        with pytest.raises(VintageConflictError, match="different values"):
            store.append_normalised(changed, "observation")
        res = store.append_normalised(changed, "observation", on_conflict="keep_existing")
        assert res.n_written == 0
        assert store.read_normalised("observation")["value"].tolist() == [1.0]

    def test_conflict_within_batch_raises(self, store: ParquetStore) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
                        ("s.eu.x_twh", "2024-01-01", 2.0, "2024-01-02", 0)])
        with pytest.raises(VintageConflictError):
            store.append_normalised(df, "observation")

    def test_nan_value_redelivery_is_duplicate(self, store: ParquetStore) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", float("nan"), "2024-01-02", 0)])
        store.append_normalised(df, "observation")
        assert store.append_normalised(df, "observation").n_written == 0

    def test_schema_violation_rejected(self, store: ParquetStore) -> None:
        df = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        df["event_time"] = df["event_time"].dt.tz_localize(None)
        with pytest.raises(SchemaError):
            store.append_normalised(df, "observation")

    def test_unknown_kind(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="Unknown normalised kind"):
            store.read_normalised("nope")

    def test_filters(self, store: ParquetStore) -> None:
        df = obs_frame([
            ("a.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0),
            ("a.eu.x_twh", "2024-01-02", 2.0, "2024-01-03", 0),
            ("a.eu.x_twh", "2024-01-02", 2.5, "2024-01-09", 1),
            ("b.eu.y_twh", "2024-01-01", 3.0, "2024-01-02", 0),
        ])
        store.append_normalised(df, "observation")
        assert set(store.read_normalised("observation", series_ids=["b.eu.y_twh"])["value"]) == {3.0}
        r = store.read_normalised("observation", start=pd.Timestamp("2024-01-02", tz="UTC"),
                                  end=pd.Timestamp("2024-01-02", tz="UTC"))
        assert sorted(r["value"]) == [2.0, 2.5]
        r = store.read_normalised("observation",
                                  publication_cutoff=pd.Timestamp("2024-01-05", tz="UTC"))
        assert 2.5 not in r["value"].tolist() and len(r) == 3
        assert store.read_normalised("observation", series_ids=["zzz"]).empty

    def test_naive_filter_refused(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="naive"):
            store.read_normalised("observation", publication_cutoff="2024-01-01")

    def test_empty_read_is_typed(self, store: ParquetStore) -> None:
        validate_frame(store.read_normalised("observation"), OBSERVATION_SCHEMA)

    def test_forecasts_and_curves(self, store: ParquetStore, small_cfg) -> None:
        fc = synthetic.weather_forecast_runs(small_cfg, max_lead_days=3)
        curve = synthetic.ttf_curve(small_cfg)
        assert store.append_normalised(fc, "forecast").n_written == len(fc)
        assert store.append_normalised(curve, "curve").n_written == len(curve)
        as_of = pd.Timestamp("2023-02-10 12:00", tz="UTC")
        pd.testing.assert_frame_equal(
            resolve_forecasts_as_of(store.read_normalised("forecast"), as_of)[fc.columns],
            resolve_forecasts_as_of(fc, as_of)[fc.columns],
        )
        one = curve["contract_id"].iloc[0]
        got = store.read_normalised("curve", series_ids=[one])
        assert (got["contract_id"] == one).all() and len(got) == (curve["contract_id"] == one).sum()
        assert store.append_normalised(curve, "curve").n_written == 0

    def test_contract_metadata_not_vintaged(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="not vintaged"):
            store.read_normalised("contract_metadata",
                                  publication_cutoff=pd.Timestamp("2024-01-01", tz="UTC"))


class TestRunArtifacts:
    def _features(self) -> pd.DataFrame:
        idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
                               name="origin_time")
        f = pd.DataFrame({"storage__dev": [0.1, 0.2, float("nan")]}, index=idx)
        f.attrs["pit_grade"] = {"storage__dev": "B"}
        return f

    def _predictions(self, synthetic_flag: bool = True) -> pd.DataFrame:
        t = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC").astype("datetime64[ns, UTC]")
        return pd.DataFrame({
            "origin_time": t, "target_id": "T3b", "horizon": pd.Series([5, 5], dtype="int64"),
            "model_id": "zero", "prediction": [0.0, 0.0], "train_end": t, "label_known_time": t,
            "fold_id": "f0", "feature_hash": "abc", "is_synthetic": [synthetic_flag] * 2,
        })

    def test_manifest_contents(self, store: ParquetStore) -> None:
        f = self._features()
        e = store.write_run_artifact("features", "run-001", "matrix", f, is_synthetic=True,
                                     config={"b": 2, "a": 1})
        m = store.read_manifest("features", "run-001")
        assert m.is_synthetic and m.config_hash and m.created_time
        assert m.artifacts["matrix"].n_rows == 3 and e.content_sha256 == frame_content_hash(f)
        back = store.read_run_artifact("features", "run-001", "matrix")
        pd.testing.assert_frame_equal(back, f, check_freq=False)
        assert back.attrs["pit_grade"] == {"storage__dev": "B"}

    def test_config_hash_order_independent(self, store: ParquetStore) -> None:
        f = self._features()
        store.write_run_artifact("features", "r1", "m", f, is_synthetic=True, config={"a": 1, "b": 2})
        store.write_run_artifact("features", "r2", "m", f, is_synthetic=True, config={"b": 2, "a": 1})
        assert (store.read_manifest("features", "r1").config_hash
                == store.read_manifest("features", "r2").config_hash)

    def test_rewrite_identical_ok_different_refused(self, store: ParquetStore) -> None:
        f = self._features()
        store.write_run_artifact("features", "r1", "m", f, is_synthetic=True)
        store.write_run_artifact("features", "r1", "m", f, is_synthetic=True)
        with pytest.raises(StorageError, match="immutable"):
            store.write_run_artifact("features", "r1", "m", f * 2, is_synthetic=True)

    def test_config_mismatch_refused(self, store: ParquetStore) -> None:
        f = self._features()
        store.write_run_artifact("features", "r1", "m", f, is_synthetic=True, config={"a": 1})
        with pytest.raises(StorageError, match="different config"):
            store.write_run_artifact("features", "r1", "m2", f, is_synthetic=True, config={"a": 2})

    def test_predictions_schema_enforced(self, store: ParquetStore) -> None:
        store.write_run_artifact("predictions", "r1", "p", self._predictions(), is_synthetic=True)
        with pytest.raises(SchemaError):
            store.write_run_artifact("predictions", "r1", "bad", pd.DataFrame({"x": [1.0]}),
                                     is_synthetic=True)
        validate_frame(store.read_run_artifact("predictions", "r1", "p"), PREDICTION_SCHEMA)

    def test_synthetic_cannot_be_published_as_real(self, store: ParquetStore) -> None:
        with pytest.raises(SyntheticLabelError):
            store.write_run_artifact("predictions", "r1", "p", self._predictions(True),
                                     is_synthetic=False)
        obs = obs_frame([("s.eu.x_twh", "2024-01-01", 1.0, "2024-01-02", 0)])
        with pytest.raises(SyntheticLabelError):
            store.write_run_artifact("features", "r2", "raw_copy", obs, is_synthetic=False)

    def test_synthetic_flag_fixed_per_run(self, store: ParquetStore) -> None:
        store.write_run_artifact("benchmarks", "r1", "b", self._predictions(False),
                                 is_synthetic=False)
        with pytest.raises(StorageError, match="is_synthetic"):
            store.write_run_artifact("benchmarks", "r1", "b2", self._predictions(True),
                                     is_synthetic=True)

    def test_tampered_file_detected(self, store: ParquetStore) -> None:
        store.write_run_artifact("features", "r1", "m", self._features(), is_synthetic=True)
        path = store.root / "features" / "r1" / "m.parquet"
        other = self._features() * 3
        other.to_parquet(path)
        with pytest.raises(StorageError, match="manifest hash"):
            store.read_run_artifact("features", "r1", "m")

    def test_missing_manifest_and_artifact(self, store: ParquetStore) -> None:
        with pytest.raises(StorageError, match="No manifest"):
            store.read_manifest("features", "nothing")
        store.write_run_artifact("features", "r1", "m", self._features(), is_synthetic=True)
        with pytest.raises(StorageError, match="No artefact"):
            store.read_run_artifact("features", "r1", "zzz")
