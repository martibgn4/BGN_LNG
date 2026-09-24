"""DuckDB store. SPEC 3 (long-format schema) and SPEC 8 (vintaging).

The vintaging requirement drives the whole design. ENTSOG and AGSI restate
same-day figures for days after publication, so an observation is identified by
(obs_date, country, point_key, series_id, source, vintage) - NOT by
(obs_date, series_id) alone. Re-fetching the same gas day later inserts a new
row at a new vintage instead of overwriting the old one, and `as_of` queries
see only what was actually published at the time. Backtests must go through
`read_as_of`; querying `observations` directly returns restated data and
produces the flattering lies SPEC 8 warns about.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from gasbalance.config import config_hash

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "output" / "gasbalance.duckdb"

OBSERVATION_COLUMNS = [
    "obs_date",
    "country",
    "point_key",
    "operator_key",
    "direction_key",
    "series_id",
    "value",
    "unit",
    "source",
    "origin",
    "vintage",
    "source_updated_at",
    "is_estimated",
    "run_id",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    obs_date     DATE       NOT NULL,
    country      VARCHAR    NOT NULL,
    point_key    VARCHAR,             -- NULL for country-level series
    operator_key VARCHAR,             -- reporting TSO; the two sides of an IP
    direction_key VARCHAR,            -- differ only by these two columns
    series_id    VARCHAR    NOT NULL,
    value        DOUBLE     NOT NULL,
    unit         VARCHAR    NOT NULL, -- internal unit only: mcm/d or TWh
    source       VARCHAR    NOT NULL, -- publisher: entsog, agsi, alsi, bloomberg, kpler
    origin       VARCHAR,             -- physical gas origin used for the CV, if any
    vintage      TIMESTAMP  NOT NULL, -- when WE fetched it; SPEC 8
    source_updated_at TIMESTAMP,       -- publisher restatement stamp, if given
    is_estimated BOOLEAN    NOT NULL DEFAULT FALSE,  -- SPEC 3
    run_id       VARCHAR    NOT NULL
);

CREATE TABLE IF NOT EXISTS points (
    point_key        VARCHAR NOT NULL,
    point_label      VARCHAR,
    operator_key     VARCHAR,
    operator_label   VARCHAR,
    direction_key    VARCHAR,
    point_type       VARCHAR,
    cross_border_type VARCHAR,
    tso_country      VARCHAR,
    adjacent_country VARCHAR,
    ring_role        VARCHAR NOT NULL,  -- crossing | internal | outside | ignored
    has_data         BOOLEAN,
    is_double_reporting BOOLEAN,
    is_pipe_in_pipe  BOOLEAN,
    gcv_min          DOUBLE,
    gcv_max          DOUBLE,
    gcv_unit         VARCHAR,
    vintage          TIMESTAMP NOT NULL,
    run_id           VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR   NOT NULL,
    started_at  TIMESTAMP NOT NULL,
    command     VARCHAR   NOT NULL,
    config_hash VARCHAR   NOT NULL
);
"""


@dataclass(frozen=True)
class RunContext:
    """Identifies one ingestion or model run. SPEC 9 provenance."""

    run_id: str
    started_at: datetime
    command: str
    config_hash: str

    @classmethod
    def new(cls, command: str) -> RunContext:
        return cls(
            run_id=uuid.uuid4().hex[:12],
            started_at=datetime.now(UTC).replace(tzinfo=None),
            command=command,
            config_hash=config_hash(),
        )


class Store:
    """Thin wrapper over a single local DuckDB file."""

    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -------------------------------------------------------------

    def register_run(self, ctx: RunContext) -> None:
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            [ctx.run_id, ctx.started_at, ctx.command, ctx.config_hash],
        )

    def write_observations(self, df: pd.DataFrame, ctx: RunContext) -> int:
        """Append observations at a new vintage. Never updates in place.

        The frame must already be in internal units - conversion happens in the
        fetcher (SPEC 5), not here.
        """
        if df.empty:
            return 0
        missing = [c for c in OBSERVATION_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"observation frame missing columns: {missing}")
        frame = df[OBSERVATION_COLUMNS].copy()
        self.con.register("incoming", frame)
        self.con.execute("INSERT INTO observations SELECT * FROM incoming")
        self.con.unregister("incoming")
        return len(frame)

    def write_points(self, df: pd.DataFrame, ctx: RunContext) -> int:
        if df.empty:
            return 0
        self.con.register("incoming_points", df)
        self.con.execute("INSERT INTO points SELECT * FROM incoming_points")
        self.con.unregister("incoming_points")
        return len(df)

    # -- reads --------------------------------------------------------------

    def read_as_of(
        self,
        as_of: datetime,
        series_id: str | None = None,
        country: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return the latest observation per key that existed at `as_of`.

        This is the only read a backtest may use. SPEC 8.
        """
        clauses = ["vintage <= ?"]
        params: list[object] = [as_of]
        if series_id is not None:
            clauses.append("series_id = ?")
            params.append(series_id)
        if country is not None:
            clauses.append("country = ?")
            params.append(country)
        if start is not None:
            clauses.append("obs_date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("obs_date <= ?")
            params.append(end)
        where = " AND ".join(clauses)
        sql = f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY obs_date, country, point_key, operator_key,
                                 direction_key, series_id, source
                    ORDER BY vintage DESC
                ) AS rn
                FROM observations
                WHERE {where}
            ) WHERE rn = 1
        """
        return self.con.execute(sql, params).df()

    def read_latest(self, **kwargs: object) -> pd.DataFrame:
        """Latest vintage of everything. Fine for live runs, NOT for backtests."""
        return self.read_as_of(datetime.now(UTC).replace(tzinfo=None), **kwargs)  # type: ignore[arg-type]

    def restatement_history(self, series_id: str, country: str) -> pd.DataFrame:
        """Every vintage of one series, so restatement size can be measured."""
        return self.con.execute(
            """
            SELECT obs_date, point_key, vintage, value
            FROM observations
            WHERE series_id = ? AND country = ?
            ORDER BY obs_date, point_key, vintage
            """,
            [series_id, country],
        ).df()

    def latest_points(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY point_key, operator_key, direction_key
                    ORDER BY vintage DESC
                ) AS rn FROM points
            ) WHERE rn = 1
            """
        ).df()
