"""Command line: fetch, build, validate.

Every command writes through `store.py`, so every fetch is vintaged and
nothing overwrites. Commands that need a Bloomberg Terminal say so when it is
absent rather than failing obscurely.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd
import typer

from wxreturns import store
from wxreturns.config import CONFIG_DIR, config_hash, get
from wxreturns.data import openmeteo
from wxreturns.data.base import CredentialsMissing, SourceUnavailable

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def info() -> None:
    """Show config provenance and what is cached."""
    typer.echo(f"config dir   : {CONFIG_DIR}")
    typer.echo(f"config hash  : {config_hash()}")
    typer.echo(f"sample start : {get('model', 'sample', 'start')}")
    typer.echo(f"max lead     : "
               f"{get('sources', 'open_meteo', 'previous_runs', 'max_lead_days')} days "
               f"(archive from "
               f"{get('sources', 'open_meteo', 'previous_runs', 'archive_verified_from')})")
    typer.echo("")
    typer.echo("cached datasets:")
    names = store.datasets()
    if not names:
        typer.echo("  (none yet - run fetch-weather)")
    for name in names:
        try:
            frame = store.read(name)
            typer.echo(f"  {name:36s} {len(frame):>9,} rows")
        except FileNotFoundError:
            typer.echo(f"  {name:36s} (empty)")


@app.command("audit-weights")
def audit_weights() -> None:
    """List regions whose demand weights are still estimates.

    An estimate is fine for building the pipeline and is not fine for trading.
    This exists so those never quietly become permanent.
    """
    regions = get("regions", "regions")
    estimates = [(name, cfg["label"]) for name, cfg in regions.items()
                 if cfg.get("provenance") == "estimate"]
    for name, label in estimates:
        typer.echo(f"  ESTIMATE  {name:20s} {label}")
    verified = len(regions) - len(estimates)
    typer.echo("")
    typer.echo(f"{len(estimates)} of {len(regions)} regions are estimates, "
               f"{verified} verified.")
    if estimates:
        typer.echo("Replace the EU gas weights from "
                   "20260825-EuropeLocalDistributionZoneGasDemandMonitor.xlsx "
                   "before trading the TTF leg.")


@app.command("fetch-weather")
def fetch_weather(
    region: str = typer.Option(..., help="Region key from regions.yaml"),
    start: str = typer.Option(..., help="First target date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="Last target date, YYYY-MM-DD"),
    leads: str = typer.Option("", help="Comma-separated leads, default 1..7"),
) -> None:
    """Archived forecasts at leads 1..7 - the revision panel's raw material."""
    lead_list = ([int(x) for x in leads.split(",")] if leads else None)
    fetcher = openmeteo.HistoricalForecastFetcher()
    try:
        result = fetcher.fetch(region=region,
                               start=date.fromisoformat(start),
                               end=date.fromisoformat(end),
                               leads=lead_list)
    except (SourceUnavailable, CredentialsMissing) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    path = store.write(f"forecast_{region}", result.frame,
                       tag=f"{start}_{end}")
    typer.echo(f"{result.rows:,} rows -> {path}")


@app.command("fetch-actuals")
def fetch_actuals(
    region: str = typer.Option(..., help="Region key from regions.yaml"),
    start: str = typer.Option(..., help="First date, YYYY-MM-DD"),
    end: str = typer.Option(..., help="Last date, YYYY-MM-DD"),
) -> None:
    """ERA5 reanalysis: the actuals, and the basis for normals."""
    fetcher = openmeteo.ArchiveFetcher()
    try:
        result = fetcher.fetch(region=region,
                               start=date.fromisoformat(start),
                               end=date.fromisoformat(end))
    except (SourceUnavailable, CredentialsMissing) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    path = store.write(f"actuals_{region}", result.frame, tag=f"{start}_{end}")
    typer.echo(f"{result.rows:,} rows -> {path}")


@app.command("fetch-live")
def fetch_live(
    region: str = typer.Option(..., help="Region key from regions.yaml"),
    model: str = typer.Option("ecmwf_ifs025", help="Deterministic model"),
    ensemble: bool = typer.Option(False, help="Also pull the ensemble spread"),
) -> None:
    """Today's run. Run daily to accumulate the far leads history cannot give."""
    result = openmeteo.ForecastFetcher().fetch(region=region, model=model)
    path = store.write(f"live_{region}", result.frame, tag=model)
    typer.echo(f"deterministic: {result.rows:,} rows -> {path}")

    if ensemble:
        ens = openmeteo.EnsembleFetcher().fetch(region=region)
        spread = openmeteo.ensemble_spread(ens.frame)
        path = store.write(f"spread_{region}", spread)
        typer.echo(f"spread       : {len(spread):,} rows -> {path}")


@app.command("fetch-prices")
def fetch_prices(
    commodity: str = typer.Option(..., help="Key from targets.yaml contracts"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Bloomberg settles. Needs a logged-in Terminal."""
    from wxreturns.data.bloomberg import SettleFetcher
    try:
        result = SettleFetcher().fetch(commodity=commodity,
                                       start=date.fromisoformat(start),
                                       end=date.fromisoformat(end))
    except SourceUnavailable as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    path = store.write(f"settles_{commodity.strip()}", result.frame,
                       tag=f"{start}_{end}")
    typer.echo(f"{result.rows:,} rows -> {path}")


@app.command("check-revisions")
def check_revisions(
    region: str = typer.Option(..., help="Region key from regions.yaml"),
    cold_start: str = typer.Option("", help="Known cold snap start, YYYY-MM-DD"),
    cold_end: str = typer.Option("", help="Known cold snap end, YYYY-MM-DD"),
) -> None:
    """Gate 1: build the revision panel and check its sign on a known event."""
    from wxreturns.weather import degree_days, revisions as rev

    panel = store.read_latest(f"forecast_{region}")
    daily = degree_days.region_degree_days(panel, region)
    revision_frame = rev.daily_revisions(daily, value_column="hdd")

    typer.echo(f"revision rows : {len(revision_frame):,}")
    typer.echo(f"buckets       : "
               f"{sorted(revision_frame['bucket'].dropna().unique())}")
    typer.echo(f"mean |revision|: "
               f"{revision_frame['revision'].abs().mean():.3f} HDD")

    if cold_start and cold_end:
        checked = revision_frame.rename(columns={"issue_date": "obs_date"})
        outcome = rev.sanity_check_sign(checked, (cold_start, cold_end))
        for key, value in outcome.items():
            typer.echo(f"  {key:18s} {value}")
        if not outcome["sign_as_expected"]:
            typer.secho(
                "SIGN CHECK FAILED. A forecast turning colder must raise HDD, "
                "so a known cold snap must show a POSITIVE total revision. A "
                "negative one means the issue axis is inverted and every "
                "coefficient downstream would carry the wrong sign.",
                fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho("sign check passed", fg=typer.colors.GREEN)


@app.command("backfill")
def backfill(
    region: str = typer.Option(..., help="Region key from regions.yaml"),
    start: str = typer.Option("", help="Defaults to the archive floor"),
    end: str = typer.Option("", help="Defaults to yesterday"),
    chunk_days: int = typer.Option(90, help="Days per request"),
) -> None:
    """Walk the archive in chunks. Each chunk is a separate cache vintage."""
    floor = get("sources", "open_meteo", "previous_runs", "archive_verified_from")
    first = date.fromisoformat(start or floor)
    last = date.fromisoformat(end) if end else (datetime.now().date()
                                               - timedelta(days=1))
    fetcher = openmeteo.HistoricalForecastFetcher()

    total, cursor = 0, first
    while cursor <= last:
        stop = min(cursor + timedelta(days=chunk_days - 1), last)
        try:
            result = fetcher.fetch(region=region, start=cursor, end=stop)
        except SourceUnavailable as exc:
            typer.secho(f"{cursor}..{stop}: {exc}", fg=typer.colors.YELLOW)
            cursor = stop + timedelta(days=1)
            continue
        store.write(f"forecast_{region}", result.frame,
                    tag=f"{cursor}_{stop}")
        total += result.rows
        typer.echo(f"  {cursor}..{stop}  {result.rows:>8,} rows")
        cursor = stop + timedelta(days=1)

    typer.echo(f"backfilled {total:,} rows for {region}")


@app.command("roll-calendar")
def roll_calendar_command(
    commodity: str = typer.Option(..., help="Key from targets.yaml"),
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
) -> None:
    """Build the constructed roll calendar and show the delivery windows."""
    from wxreturns.market import contracts

    dates = pd.bdate_range(start, end)
    calendar = contracts.roll_calendar(commodity, dates)
    front = calendar[calendar["slot"] == "front"]
    typer.echo(f"{len(calendar):,} rows, "
               f"{front['code'].nunique()} distinct front contracts")
    typer.echo(front.groupby("code")
               .agg(first=("obs_date", "min"), last=("obs_date", "max"),
                    window_start=("window_start", "first"),
                    window_end=("window_end", "first"))
               .to_string())


if __name__ == "__main__":
    app()
