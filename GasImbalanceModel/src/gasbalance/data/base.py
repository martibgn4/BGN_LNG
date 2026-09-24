"""Fetcher contract. SPEC 6.

Every fetcher:
  - caches its raw payload to Parquet before any transformation, so a bad
    conversion can be re-run without re-hitting the API;
  - converts to internal units inside `normalise`, because SPEC 5 forbids a
    raw-unit value from propagating past the fetcher;
  - raises `CredentialsMissing` rather than returning anything when it has no
    credentials. SPEC 6 and SPEC 11: never synthesise data to make a run work.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache"


class CredentialsMissing(RuntimeError):
    """No credential for a source that requires one.

    Deliberately fatal. SPEC 6: "the fetcher must raise, not return synthetic
    values."
    """


class SourceUnavailable(RuntimeError):
    """The source answered, but not with usable data (endpoint moved, outage).

    Also fatal. SPEC 6: "If any public endpoint has changed or a key is
    required, tell me - do not work around it by inventing data."
    """


@dataclass(frozen=True)
class FetchResult:
    """Normalised observations plus the provenance needed to write them."""

    frame: pd.DataFrame
    source: str
    vintage: datetime
    rows_raw: int
    rows_dropped: int
    drop_reasons: dict[str, int]


class Fetcher(ABC):
    """Base class for one data source."""

    source: str = "unset"
    requires_credentials: bool = False
    credential_env_var: str | None = None

    def __init__(self, cache_dir: Path | str = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir) / self.source
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- credentials --------------------------------------------------------

    def credential(self) -> str:
        """Read the API key from the environment. Never from a file in the repo."""
        if not self.requires_credentials:
            return ""
        if not self.credential_env_var:
            raise CredentialsMissing(
                f"{self.source} requires credentials but declares no env var"
            )
        value = os.environ.get(self.credential_env_var, "").strip()
        if not value:
            raise CredentialsMissing(
                f"{self.source} needs an API key in ${self.credential_env_var}. "
                "Set it and re-run. This fetcher will not return placeholder data."
            )
        return value

    def is_available(self) -> bool:
        """Whether this fetcher could run right now, without running it."""
        try:
            self.credential()
        except CredentialsMissing:
            return False
        return True

    # -- caching ------------------------------------------------------------

    def cache_path(self, tag: str, vintage: datetime) -> Path:
        stamp = vintage.strftime("%Y%m%dT%H%M%S")
        return self.cache_dir / f"{tag}__{stamp}.parquet"

    def cache_raw(self, frame: pd.DataFrame, tag: str, vintage: datetime) -> Path:
        path = self.cache_path(tag, vintage)
        frame.to_parquet(path, index=False)
        log.info("cached %d raw rows to %s", len(frame), path.name)
        return path

    # -- contract -----------------------------------------------------------

    @abstractmethod
    def fetch(self, start: date, end: date, **kwargs: object) -> FetchResult:
        """Fetch, cache raw, convert to internal units, return observations."""

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)
