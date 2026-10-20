"""Layered local store: raw payloads, append-only normalised vintages, and run artefacts.

Layout (``docs/point_in_time_policy.md`` §7, ``docs/data_dictionary.md``)::

    <root>/raw/<source>/ingest_date=YYYY-MM-DD/<name>-<HHMMSSffffff>-<sha12>.<ext>   (+ .meta.json)
    <root>/normalised/<kind>/<partition_col>=<value>/part-<sha16>.parquet
    <root>/{features,predictions,benchmarks}/<run_id>/<artifact>.parquet + manifest.json

Rules enforced here:

* **Raw** payloads are saved as received (bytes, text, JSON or a frame) and never overwritten.
* **Normalised** is append-only. A new vintage is a new row; a row whose dedupe key (series key +
  ``revision`` + ``publication_time``) already exists is skipped, so re-ingesting the same batch is a
  no-op. The same key arriving with a *different* value is a conflict and raises by default.
* **Run artefacts** carry a manifest with content hashes, row counts, a config hash, creation time and
  ``is_synthetic``; a frame containing SYNTHETIC rows can never be written as non-synthetic.
* Every path is resolved and refused if it escapes the data root.

``Store`` is the interface the rest of the code depends on. ``ParquetStore`` is the v1 backend
(D-004); a DuckDB or database backend implements the same Protocol without touching callers.

Reading the normalised layer with ``publication_cutoff`` is a *pre-filter* only; feature code must
still pass the result through ``features.point_in_time.resolve_as_of`` to pick vintages.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    PREDICTION_SCHEMA,
    SYNTHETIC_SOURCE,
    DataError,
    Entitlement,
    QualityFlag,
    Schema,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.settings import Paths

# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class StorageError(DataError):
    """A storage operation failed (bad path, unknown kind, unreadable artefact)."""


class PathOutsideRootError(StorageError):
    """A computed path resolves outside the configured data root."""


class VintageConflictError(StorageError):
    """A row with an existing dedupe key arrived with different content.

    A vendor re-delivering the *same* vintage with a different value means either the revision or
    publication_time was not recorded correctly; silently keeping either copy would corrupt the
    as-of view, so the write is refused.
    """


class SyntheticLabelError(StorageError):
    """Synthetic rows were about to be stored or published as non-synthetic."""


# ---------------------------------------------------------------------------------------------
# Normalised kinds
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalisedKind:
    """How one canonical schema is stored in the normalised layer.

    Attributes:
        name: kind name, also the directory under ``normalised/``.
        schema: the canonical schema rows must satisfy.
        dedupe_key: columns identifying one stored vintage of one fact.
        value_columns: columns compared to detect a conflicting re-delivery.
        series_column: column matched by ``series_ids`` filters.
        event_column: column matched by event-time range filters (``None``: no event filter).
        partition_column: column whose value names the partition directory.
        vintaged: whether rows carry ``publication_time`` (cutoff filter allowed).
    """

    name: str
    schema: Schema
    dedupe_key: tuple[str, ...]
    value_columns: tuple[str, ...]
    series_column: str
    event_column: str | None
    partition_column: str
    vintaged: bool = True


NORMALISED_KINDS: dict[str, NormalisedKind] = {
    "observation": NormalisedKind(
        "observation",
        OBSERVATION_SCHEMA,
        ("series_id", "event_time", "revision", "publication_time"),
        ("value", "unit"),
        "series_id",
        "event_time",
        "series_id",
    ),
    "forecast": NormalisedKind(
        "forecast",
        FORECAST_SCHEMA,
        ("series_id", "issue_time", "valid_time", "model", "member", "revision",
         "publication_time"),
        ("value", "unit", "lead_hours"),
        "series_id",
        "valid_time",
        "series_id",
    ),
    "curve": NormalisedKind(
        "curve",
        CURVE_SCHEMA,
        ("hub", "contract_id", "field", "trade_date", "revision", "publication_time"),
        ("value", "unit", "currency"),
        "contract_id",
        "trade_date",
        "hub",
    ),
    "contract_metadata": NormalisedKind(
        "contract_metadata",
        CONTRACT_METADATA_SCHEMA,
        ("contract_id", "last_trade_date_source"),
        ("delivery_start", "delivery_end", "last_trade_date", "currency", "unit"),
        "contract_id",
        None,
        "hub",
        vintaged=False,
    ),
}

RUN_LAYERS: tuple[str, ...] = ("features", "predictions", "benchmarks")
DEFAULT_RUN_SCHEMAS: dict[str, Schema | None] = {
    "features": None,
    "predictions": PREDICTION_SCHEMA,
    "benchmarks": PREDICTION_SCHEMA,
}

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=\-]{0,199}$")
MANIFEST_NAME = "manifest.json"


def get_kind(kind: str) -> NormalisedKind:
    """Return the ``NormalisedKind`` for ``kind`` or raise listing the valid names."""
    try:
        return NORMALISED_KINDS[kind]
    except KeyError as exc:
        raise StorageError(
            f"Unknown normalised kind {kind!r}; expected one of {sorted(NORMALISED_KINDS)}"
        ) from exc


# ---------------------------------------------------------------------------------------------
# Results and manifests
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    """Where a raw payload landed and how to verify it."""

    path: Path
    sha256: str
    n_bytes: int
    written: bool  # False when an identical payload was already stored


@dataclass(frozen=True)
class AppendResult:
    """Outcome of an append to the normalised layer."""

    kind: str
    n_input: int
    n_written: int
    n_duplicates_skipped: int
    files: tuple[Path, ...]


@dataclass
class ArtifactEntry:
    """One artefact inside a run manifest."""

    name: str
    file: str
    n_rows: int
    n_columns: int
    columns: list[str]
    content_sha256: str
    file_sha256: str
    schema: str | None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Manifest of a run directory (features / predictions / benchmarks)."""

    layer: str
    run_id: str
    created_time: str
    config_hash: str | None
    is_synthetic: bool
    artifacts: dict[str, ArtifactEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        raw = json.loads(text)
        arts = {k: ArtifactEntry(**v) for k, v in raw.pop("artifacts", {}).items()}
        return cls(artifacts=arts, **raw)


# ---------------------------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def config_hash(config: Mapping[str, Any] | None) -> str | None:
    """Stable sha256 of a JSON-serialisable config mapping (keys sorted); ``None`` for no config."""
    if config is None:
        return None
    return sha256_bytes(json.dumps(config, sort_keys=True, default=str).encode("utf-8"))


def frame_content_hash(df: pd.DataFrame) -> str:
    """sha256 of a frame's content (values, index, column names and dtypes), independent of the
    file format it is written in. Row order matters: callers sort before hashing if it should not.
    """
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(json.dumps([str(t) for t in df.dtypes]).encode())
    h.update(json.dumps([str(n) for n in df.index.names]).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def is_synthetic_frame(df: pd.DataFrame) -> bool:
    """True if any row is labelled synthetic (source, entitlement, quality_flag or is_synthetic)."""
    checks = (
        ("source", SYNTHETIC_SOURCE),
        ("entitlement", Entitlement.SYNTHETIC.value),
        ("quality_flag", QualityFlag.SYNTHETIC.value),
    )
    for col, label in checks:
        if col in df.columns and (df[col] == label).any():
            return True
    return bool("is_synthetic" in df.columns and df["is_synthetic"].astype(bool).any())


# ---------------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class Store(Protocol):
    """Storage interface. Backends (Parquet now; DuckDB / database later) implement this."""

    def write_raw(
        self,
        source: str,
        name: str,
        payload: bytes | str | Mapping[str, Any] | Sequence[Any] | pd.DataFrame,
        *,
        ingestion_time: pd.Timestamp,
        metadata: Mapping[str, Any] | None = None,
    ) -> RawRecord:
        """Save a vendor payload as received; never overwrite."""
        ...

    def append_normalised(
        self, df: pd.DataFrame, kind: str, *, on_conflict: str = "raise"
    ) -> AppendResult:
        """Append canonical rows; skip exact re-deliveries; refuse conflicting ones."""
        ...

    def read_normalised(
        self,
        kind: str,
        *,
        series_ids: Sequence[str] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        publication_cutoff: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Read canonical rows, optionally filtered. Not an as-of resolution."""
        ...

    def write_run_artifact(
        self,
        layer: str,
        run_id: str,
        name: str,
        df: pd.DataFrame,
        *,
        is_synthetic: bool,
        config: Mapping[str, Any] | None = None,
        schema: Schema | None = None,
    ) -> ArtifactEntry:
        """Write a feature / prediction / benchmark frame under ``run_id`` and update its manifest."""
        ...

    def read_run_artifact(self, layer: str, run_id: str, name: str) -> pd.DataFrame:
        """Read an artefact back, verifying its content hash against the manifest."""
        ...

    def read_manifest(self, layer: str, run_id: str) -> RunManifest:
        """Return the manifest of a run directory."""
        ...


# ---------------------------------------------------------------------------------------------
# Parquet backend
# ---------------------------------------------------------------------------------------------


def _require_utc(ts: pd.Timestamp | str, what: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise StorageError(f"{what} {ts!r} is timezone-naive; pass a tz-aware timestamp")
    return t.tz_convert("UTC")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _frame_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    return buf.getvalue()


def _restore_utc(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Parquet round-trips tz-aware columns, but normalise unit/precision to ns UTC defensively."""
    for col in schema.columns:
        if col.kind == "ts_utc" and col.name in df.columns:
            s = df[col.name]
            if isinstance(s.dtype, pd.DatetimeTZDtype):
                df[col.name] = s.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    return df


class ParquetStore:
    """Parquet implementation of ``Store`` rooted at ``settings.Paths``.

    Not safe for concurrent writers to the same partition (single-process research use). Each write
    is atomic at file level (temp file + rename).
    """

    def __init__(self, paths: Paths | str | Path) -> None:
        self.paths = paths if isinstance(paths, Paths) else Paths(Path(paths))
        self.root = self.paths.root.resolve()
        self.paths.ensure()

    # -- path safety ------------------------------------------------------------------------

    def _safe(self, *parts: str) -> Path:
        """Join ``parts`` under the root after checking each segment; refuse escapes."""
        for p in parts:
            if not _SAFE_SEGMENT.match(p) or ".." in p:
                raise PathOutsideRootError(
                    f"Unsafe path segment {p!r}: use letters, digits, '.', '_', '-' only"
                )
        path = self.root.joinpath(*parts).resolve()
        if not path.is_relative_to(self.root):
            raise PathOutsideRootError(f"{path} is outside the data root {self.root}")
        return path

    # -- raw ---------------------------------------------------------------------------------

    def write_raw(
        self,
        source: str,
        name: str,
        payload: bytes | str | Mapping[str, Any] | Sequence[Any] | pd.DataFrame,
        *,
        ingestion_time: pd.Timestamp,
        metadata: Mapping[str, Any] | None = None,
    ) -> RawRecord:
        """Save ``payload`` under ``raw/<source>/ingest_date=<UTC date>/``.

        bytes -> ``.bin``; str -> ``.txt``; mapping/sequence -> ``.json`` (sorted keys);
        DataFrame -> ``.parquet``. A ``.meta.json`` sidecar records source, name, ingestion time,
        sha256, size and caller metadata. An identical payload already stored for the same source,
        name and ingestion date is not written twice.
        """
        t = _require_utc(ingestion_time, "ingestion_time")
        if isinstance(payload, pd.DataFrame):
            data, ext = _frame_to_parquet_bytes(payload), "parquet"
        elif isinstance(payload, bytes | bytearray):
            data, ext = bytes(payload), "bin"
        elif isinstance(payload, str):
            data, ext = payload.encode("utf-8"), "txt"
        elif isinstance(payload, Mapping | list | tuple):
            data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ext = "json"
        else:
            raise StorageError(f"Unsupported raw payload type {type(payload).__name__}")
        digest = sha256_bytes(data)
        part_dir = self._safe("raw", source, f"ingest_date={t:%Y-%m-%d}")
        self._safe(name)  # validate the name segment
        existing = sorted(part_dir.glob(f"{name}-*-{digest[:12]}.{ext}")) if part_dir.exists() else []
        if existing:
            return RawRecord(existing[0], digest, len(data), written=False)
        path = part_dir / f"{name}-{t:%H%M%S%f}-{digest[:12]}.{ext}"
        _atomic_write_bytes(path, data)
        meta = {
            "source": source,
            "name": name,
            "ingestion_time": t.isoformat(),
            "sha256": digest,
            "n_bytes": len(data),
            "format": ext,
            "metadata": dict(metadata or {}),
        }
        _atomic_write_bytes(
            path.with_suffix(path.suffix + ".meta.json"),
            json.dumps(meta, indent=2, sort_keys=True, default=str).encode("utf-8"),
        )
        return RawRecord(path, digest, len(data), written=True)

    # -- normalised --------------------------------------------------------------------------

    def _kind_dir(self, kind: NormalisedKind) -> Path:
        return self._safe("normalised", kind.name)

    def _partition_dir(self, kind: NormalisedKind, value: str) -> Path:
        return self._safe("normalised", kind.name, f"{kind.partition_column}={value}")

    def _read_partition(self, kind: NormalisedKind, part_dir: Path) -> pd.DataFrame:
        files = sorted(part_dir.glob("part-*.parquet"))
        if not files:
            return empty_frame(kind.schema)
        frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
        return _restore_utc(pd.concat(frames, ignore_index=True), kind.schema)

    def append_normalised(
        self, df: pd.DataFrame, kind: str, *, on_conflict: str = "raise"
    ) -> AppendResult:
        """Validate ``df`` against the kind's schema and append only rows not already stored.

        Args:
            df: canonical rows.
            kind: one of ``NORMALISED_KINDS``.
            on_conflict: ``"raise"`` (default) raises ``VintageConflictError`` when a stored dedupe
                key arrives with different ``value_columns``; ``"keep_existing"`` skips such rows.

        Raises:
            SchemaError, VintageConflictError, StorageError.
        """
        k = get_kind(kind)
        if on_conflict not in ("raise", "keep_existing"):
            raise StorageError(f"on_conflict must be 'raise' or 'keep_existing', got {on_conflict!r}")
        validate_frame(df, k.schema)
        key = list(k.dedupe_key)
        # Within-batch duplicates: identical rows collapse; differing content is a conflict.
        batch = df.drop_duplicates(subset=key + list(k.value_columns))
        dup_in_batch = batch.duplicated(subset=key, keep=False)
        if dup_in_batch.any():
            sample = batch.loc[dup_in_batch, key].head(3).to_dict("records")
            raise VintageConflictError(
                f"{int(dup_in_batch.sum())} rows in the batch share a dedupe key {key} with "
                f"different {list(k.value_columns)}; e.g. {sample}"
            )
        n_skipped = len(df) - len(batch)
        written: list[Path] = []
        n_written = 0
        for part_value, part in batch.groupby(k.partition_column, sort=True):
            part_dir = self._partition_dir(k, str(part_value))
            stored = self._read_partition(k, part_dir)
            new = part
            if len(stored):
                merged = part.merge(
                    stored[key + list(k.value_columns)], on=key, how="left",
                    suffixes=("", "__stored"), indicator=True,
                )
                seen = (merged["_merge"] == "both").to_numpy()
                differs = np.zeros(len(merged), dtype=bool)
                for col in k.value_columns:
                    a, b = merged[col], merged[f"{col}__stored"]
                    same = (a == b) | (a.isna() & b.isna())
                    differs |= seen & ~same.to_numpy()
                if differs.any():
                    if on_conflict == "raise":
                        sample = merged.loc[differs, key].head(3).to_dict("records")
                        raise VintageConflictError(
                            f"{int(differs.sum())} rows of kind {k.name!r} partition "
                            f"{part_value!r} re-deliver a stored vintage with different values; "
                            f"e.g. {sample}. Record a new revision/publication_time instead."
                        )
                new = part.loc[~seen]
                n_skipped += int(seen.sum())
            if new.empty:
                continue
            new = new.sort_values(key).reset_index(drop=True)
            digest = frame_content_hash(new)
            path = part_dir / f"part-{digest[:16]}.parquet"
            if not path.exists():
                _atomic_write_bytes(path, _frame_to_parquet_bytes(new))
            written.append(path)
            n_written += len(new)
        return AppendResult(k.name, len(df), n_written, n_skipped, tuple(written))

    def read_normalised(
        self,
        kind: str,
        *,
        series_ids: Sequence[str] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        publication_cutoff: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Read stored rows of ``kind``.

        Args:
            series_ids: keep rows whose ``series_column`` is in this list.
            start, end: inclusive bounds on ``event_column`` (tz-aware).
            publication_cutoff: keep rows with ``publication_time <= cutoff`` (tz-aware).

        Returns:
            All matching vintages, sorted by the dedupe key, validated against the schema. Use
            ``resolve_as_of`` / ``resolve_forecasts_as_of`` to pick one vintage per key.
        """
        k = get_kind(kind)
        kdir = self._kind_dir(k)
        part_dirs = sorted(p for p in kdir.glob(f"{k.partition_column}=*") if p.is_dir()) \
            if kdir.exists() else []
        if series_ids is not None and k.series_column == k.partition_column:
            wanted = {f"{k.partition_column}={s}" for s in series_ids}
            part_dirs = [p for p in part_dirs if p.name in wanted]
        frames = [self._read_partition(k, p) for p in part_dirs]
        frames = [f for f in frames if len(f)]
        out = pd.concat(frames, ignore_index=True) if frames else empty_frame(k.schema)
        mask = pd.Series(True, index=out.index)
        if series_ids is not None:
            mask &= out[k.series_column].isin(list(series_ids))
        if start is not None or end is not None:
            if k.event_column is None:
                raise StorageError(f"kind {k.name!r} has no event column; start/end not allowed")
            if start is not None:
                mask &= out[k.event_column] >= _require_utc(start, "start")
            if end is not None:
                mask &= out[k.event_column] <= _require_utc(end, "end")
        if publication_cutoff is not None:
            if not k.vintaged:
                raise StorageError(f"kind {k.name!r} is not vintaged; publication_cutoff not allowed")
            mask &= out["publication_time"] <= _require_utc(publication_cutoff, "publication_cutoff")
        out = out.loc[mask].sort_values(list(k.dedupe_key)).reset_index(drop=True)
        return validate_frame(out, k.schema)

    # -- run artefacts -----------------------------------------------------------------------

    def _run_dir(self, layer: str, run_id: str) -> Path:
        if layer not in RUN_LAYERS:
            raise StorageError(f"Run layer must be one of {RUN_LAYERS}, got {layer!r}")
        return self._safe(layer, run_id)

    def read_manifest(self, layer: str, run_id: str) -> RunManifest:
        """Return the manifest of ``<layer>/<run_id>``; raise if absent."""
        path = self._run_dir(layer, run_id) / MANIFEST_NAME
        if not path.exists():
            raise StorageError(f"No manifest at {path}")
        return RunManifest.from_json(path.read_text(encoding="utf-8"))

    def write_run_artifact(
        self,
        layer: str,
        run_id: str,
        name: str,
        df: pd.DataFrame,
        *,
        is_synthetic: bool,
        config: Mapping[str, Any] | None = None,
        schema: Schema | None = None,
    ) -> ArtifactEntry:
        """Write ``df`` to ``<layer>/<run_id>/<name>.parquet`` and record it in the manifest.

        ``schema`` defaults to ``PREDICTION_SCHEMA`` for predictions/benchmarks and none for
        features. The run's ``config_hash`` and ``is_synthetic`` are fixed by its first artefact;
        later artefacts must agree. Re-writing an artefact with identical content is a no-op;
        different content under the same name raises (runs are immutable).

        Raises:
            SyntheticLabelError: ``df`` contains synthetic rows but ``is_synthetic`` is False.
            StorageError: layer/run/name invalid, config mismatch, or artefact already differs.
        """
        run_dir = self._run_dir(layer, run_id)
        self._safe(name)
        use_schema = schema if schema is not None else DEFAULT_RUN_SCHEMAS[layer]
        if use_schema is not None:
            validate_frame(df, use_schema)
        if is_synthetic_frame(df) and not is_synthetic:
            raise SyntheticLabelError(
                f"Artefact {layer}/{run_id}/{name} contains SYNTHETIC rows; pass is_synthetic=True"
            )
        c_hash = config_hash(config)
        manifest_path = run_dir / MANIFEST_NAME
        if manifest_path.exists():
            manifest = RunManifest.from_json(manifest_path.read_text(encoding="utf-8"))
            if manifest.is_synthetic != is_synthetic:
                raise StorageError(
                    f"Run {layer}/{run_id} is_synthetic={manifest.is_synthetic}; got {is_synthetic}"
                )
            if c_hash is not None and manifest.config_hash not in (None, c_hash):
                raise StorageError(f"Run {layer}/{run_id} was created with a different config")
            if manifest.config_hash is None:
                manifest.config_hash = c_hash
        else:
            manifest = RunManifest(
                layer=layer,
                run_id=run_id,
                created_time=pd.Timestamp.now(tz="UTC").isoformat(),
                config_hash=c_hash,
                is_synthetic=is_synthetic,
            )
        content = frame_content_hash(df)
        if name in manifest.artifacts:
            prior = manifest.artifacts[name]
            if prior.content_sha256 == content:
                return prior
            raise StorageError(
                f"Artefact {layer}/{run_id}/{name} already exists with different content; "
                "runs are immutable — use a new run_id"
            )
        data = _frame_to_parquet_bytes(df)
        file_name = f"{name}.parquet"
        _atomic_write_bytes(run_dir / file_name, data)
        attrs = json.loads(json.dumps(df.attrs, default=str)) if df.attrs else {}
        entry = ArtifactEntry(
            name=name,
            file=file_name,
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=[str(c) for c in df.columns],
            content_sha256=content,
            file_sha256=sha256_bytes(data),
            schema=use_schema.name if use_schema is not None else None,
            attrs=attrs,
        )
        manifest.artifacts[name] = entry
        _atomic_write_bytes(manifest_path, manifest.to_json().encode("utf-8"))
        return entry

    def read_run_artifact(self, layer: str, run_id: str, name: str) -> pd.DataFrame:
        """Read an artefact, verify its file hash, and restore ``attrs`` from the manifest."""
        manifest = self.read_manifest(layer, run_id)
        if name not in manifest.artifacts:
            raise StorageError(
                f"No artefact {name!r} in {layer}/{run_id}; have {sorted(manifest.artifacts)}"
            )
        entry = manifest.artifacts[name]
        path = self._run_dir(layer, run_id) / entry.file
        data = path.read_bytes()
        if sha256_bytes(data) != entry.file_sha256:
            raise StorageError(f"{path} does not match its manifest hash (edited or corrupted)")
        df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
        if entry.attrs:
            df.attrs.update(entry.attrs)
        return df
