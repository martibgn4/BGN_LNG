"""Ingest pipeline: adapter call -> raw layer -> validation -> normalised layer.

``ingest`` is backend-agnostic: it takes any zero-argument-bound adapter call (e.g.
``functools.partial(adapter.get_series, ["storage.eu.inventory_twh"], start, end)`` or a lambda) and
any ``Store``. Order of operations:

1. call the adapter (errors such as ``BackendUnavailable`` propagate unchanged);
2. write what the adapter returned to the **raw** layer as received, before any check, so a
   rejected batch can still be inspected;
3. stamp ``ingestion_time`` (our receipt time, UTC; the adapter's value is overwritten unless
   ``keep_adapter_ingestion_time``);
4. schema validation (``validate_frame``) then quality checks (``validation.assess_quality``);
   hard failures raise ``DataQualityError`` unless ``fail_on_quality=False``;
5. append to the **normalised** layer (idempotent; conflicting re-deliveries raise);
6. return an ``IngestSummary``.

Note: the adapter Protocols in ``data/base.py`` return canonical frames, so the raw layer holds the
canonical frame the adapter produced. Vendor bytes exactly as sent by the vendor can only be
captured inside the adapter (``store.write_raw``); see ``docs/data_dictionary.md`` §3.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ttf_forecaster.data.base import DataError, validate_frame
from ttf_forecaster.data.storage import AppendResult, RawRecord, Store, get_kind, is_synthetic_frame
from ttf_forecaster.data.validation import QualityParams, QualityResult, assess_quality
from ttf_forecaster.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class IngestSummary:
    """What one ``ingest`` call did."""

    kind: str
    source: str
    call_name: str
    ingestion_time: pd.Timestamp
    n_received: int
    n_written: int
    n_duplicates_skipped: int
    raw_path: Path | None
    normalised_files: tuple[Path, ...]
    is_synthetic: bool
    quality: QualityResult | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-friendly view (for logs and run manifests)."""
        return {
            "kind": self.kind,
            "source": self.source,
            "call_name": self.call_name,
            "ingestion_time": self.ingestion_time.isoformat(),
            "n_received": self.n_received,
            "n_written": self.n_written,
            "n_duplicates_skipped": self.n_duplicates_skipped,
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "n_normalised_files": len(self.normalised_files),
            "is_synthetic": self.is_synthetic,
            "hard_failures": list(self.quality.hard_failures) if self.quality else [],
            "warnings": list(self.warnings),
        }


def already_stored(store: Any, source: str, raw_name: str) -> bool:
    """True when a raw file named ``raw_name`` already exists under ``raw/<source>/`` in a local ParquetStore.

    Lets a re-run skip a batch it has already written (idempotent daily jobs). Stores without a local root
    report False, so nothing is skipped by mistake.
    """
    root = getattr(store, "root", None)
    return root is not None and any((root / "raw" / source).glob(f"ingest_date=*/{raw_name}-*.parquet"))


def _call_name(adapter_call: Callable[..., Any]) -> str:
    func = getattr(adapter_call, "func", adapter_call)  # functools.partial
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    return str(name or type(adapter_call).__name__)


def ingest(
    adapter_call: Callable[..., pd.DataFrame],
    store: Store,
    *,
    kind: str,
    source: str,
    call_kwargs: Mapping[str, Any] | None = None,
    raw_name: str | None = None,
    ingestion_time: pd.Timestamp | None = None,
    keep_adapter_ingestion_time: bool = False,
    quality_params: QualityParams | None = None,
    quality_as_of: pd.Timestamp | None = None,
    expected_frequency: str | Mapping[str, str] | None = None,
    staleness_grace: Mapping[str, pd.Timedelta] | pd.Timedelta | None = None,
    calendar: pd.DatetimeIndex | None = None,
    expected_units: Mapping[str, str] | None = None,
    fail_on_quality: bool = True,
    on_conflict: str = "raise",
) -> IngestSummary:
    """Call ``adapter_call(**call_kwargs)`` and persist the result through ``store``.

    Args:
        adapter_call: any callable returning a canonical frame of ``kind``.
        store: a ``Store`` backend.
        kind: normalised kind (``observation``, ``forecast``, ``curve``, ``contract_metadata``).
        source: raw-layer partition label (e.g. ``bloomberg``, ``spark``, ``synthetic``).
        call_kwargs: keyword arguments for the call; also recorded in the raw sidecar.
        raw_name: raw payload file stem (default: the call's function name).
        ingestion_time: receipt time to stamp (tz-aware; default now UTC). Injected in tests.
        keep_adapter_ingestion_time: keep the adapter's ``ingestion_time`` column (e.g. replaying
            a historical snapshot whose receipt time is known) instead of stamping.
        quality_params, quality_as_of, expected_frequency, staleness_grace, calendar,
            expected_units: passed to ``validation.assess_quality``. ``quality_as_of`` defaults to
            ``ingestion_time``.
        fail_on_quality: raise ``DataQualityError`` on hard quality failures (default) or record
            them in the summary and still write.
        on_conflict: passed to ``Store.append_normalised``.

    Returns:
        ``IngestSummary``.

    Raises:
        DataError: adapter returned a non-DataFrame.
        SchemaError, DataQualityError, VintageConflictError: validation failures (raw is already
        written when these are raised).
    """
    k = get_kind(kind)
    kwargs = dict(call_kwargs or {})
    t = pd.Timestamp.now(tz="UTC") if ingestion_time is None else pd.Timestamp(ingestion_time)
    if t.tzinfo is None:
        raise DataError("ingestion_time must be tz-aware")
    t = t.tz_convert("UTC")
    name = _call_name(adapter_call)

    received = adapter_call(**kwargs)
    if not isinstance(received, pd.DataFrame):
        raise DataError(f"{name} returned {type(received).__name__}, expected a pandas DataFrame")

    raw: RawRecord = store.write_raw(
        source,
        raw_name or name.replace("<", "").replace(">", "").replace(".", "_"),
        received,
        ingestion_time=t,
        metadata={"call": name, "kind": kind, "call_kwargs": kwargs, "n_rows": len(received)},
    )

    warnings: list[str] = []
    if received.empty:
        warnings.append(f"{name} returned no rows")
        log.info("ingest_empty", extra={"fields": {"call": name, "kind": kind}})
        return IngestSummary(kind, source, name, t, 0, 0, 0, raw.path, (), False, None, warnings)

    df = received.copy()
    if k.vintaged and not keep_adapter_ingestion_time:
        df["ingestion_time"] = pd.Series(t, index=df.index).astype("datetime64[ns, UTC]")
    validate_frame(df, k.schema)

    if k.vintaged:
        early = int((df["ingestion_time"] < df["publication_time"]).sum())
        if early:
            est = int(df.loc[df["ingestion_time"] < df["publication_time"],
                             "publication_time_estimated"].sum())
            warnings.append(
                f"{early} rows received before their publication_time ({est} estimated): "
                "a lag rule may be too conservative, or the source timestamps are wrong")

    quality = assess_quality(
        df, kind,
        as_of=quality_as_of if quality_as_of is not None else t,
        expected_frequency=expected_frequency,
        staleness_grace=staleness_grace,
        calendar=calendar,
        expected_units=expected_units,
        params=quality_params,
    )
    warnings.extend(quality.warnings)
    if fail_on_quality:
        quality.raise_if_failed()
    elif quality.hard_failures:
        warnings.extend(f"HARD (not blocking): {f}" for f in quality.hard_failures)

    appended: AppendResult = store.append_normalised(df, kind, on_conflict=on_conflict)
    summary = IngestSummary(
        kind=kind,
        source=source,
        call_name=name,
        ingestion_time=t,
        n_received=len(received),
        n_written=appended.n_written,
        n_duplicates_skipped=appended.n_duplicates_skipped,
        raw_path=raw.path,
        normalised_files=appended.files,
        is_synthetic=is_synthetic_frame(df),
        quality=quality,
        warnings=warnings,
    )
    log.info("ingest_done", extra={"fields": summary.as_dict()})
    return summary
