"""CLI. SPEC 3 asks for fetch, build, run, backtest, report.

Phase 1 implements `resolve-points`, `fetch` and `status`. The later commands
exist so the surface is stable, and raise rather than returning empty results,
because a command that silently does nothing is worse than one that refuses.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from gasbalance.config import CONFIG_DIR, config_hash
from gasbalance.data.agsi import AgsiFetcher
from gasbalance.data.base import CredentialsMissing, SourceUnavailable
from gasbalance.data.bloomberg import BloombergFetcher
from gasbalance.data.bnef_ldz import BnefLdzFetcher
from gasbalance.data.entsog import EntsogFetcher
from gasbalance.db import RunContext, Store
from gasbalance.perimeter import BALANCE_ROLES, ROLE_INTERNAL, Perimeter, summarise_roles
from gasbalance.units import Units
from gasbalance.validation.reconciliation import reconcile
from gasbalance.validation.storage_recon import reconcile_storage

app = typer.Typer(add_completion=False, help="European gas balance model")

GENERATED_POINTS = CONFIG_DIR / "generated" / "perimeter_points.yaml"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


@app.command("resolve-points")
def resolve_points(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Resolve the perimeter against the live ENTSOG point registry.

    Writes config/generated/perimeter_points.yaml with a provenance header, and
    records the classified registry in DuckDB. Never hand-edit the output.
    """
    _setup_logging(verbose)
    perimeter = Perimeter()
    fetcher = EntsogFetcher()
    ctx = RunContext.new("resolve-points")

    registry = fetcher.fetch_point_directions()
    classified = perimeter.classify(registry)
    roles = summarise_roles(classified)

    points = perimeter.to_points_table(classified)
    points["vintage"] = ctx.started_at
    points["run_id"] = ctx.run_id

    with Store() as store:
        store.register_run(ctx)
        store.write_points(points, ctx)

    balance_points = points[points["ring_role"].isin(BALANCE_ROLES)]
    GENERATED_POINTS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_provenance": {
            "source": "ENTSOG Transparency Platform /operatorpointdirections",
            "fetched_at": ctx.started_at.isoformat(),
            "run_id": ctx.run_id,
            "config_hash": ctx.config_hash,
            "perimeter": perimeter.name,
            "note": "GENERATED FILE - do not hand-edit. Re-run gasbalance resolve-points.",
        },
        "role_counts": roles,
        "points": [
            {
                "point_key": r.point_key,
                "point_label": r.point_label,
                "operator_key": r.operator_key,
                "direction": r.direction_key,
                "role": r.ring_role,
                "tso_country": r.tso_country,
                "adjacent_country": r.adjacent_country,
                "has_data": bool(r.has_data) if r.has_data is not None else None,
            }
            for r in balance_points.itertuples()
        ],
    }
    with GENERATED_POINTS.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)

    typer.echo(f"perimeter: {perimeter.name}")
    typer.echo(f"registry rows: {len(registry)}")
    for role, count in sorted(roles.items(), key=lambda kv: -kv[1]):
        marker = " <- in balance" if role in BALANCE_ROLES else ""
        typer.echo(f"  {count:5d}  {role}{marker}")
    typer.echo(f"wrote {GENERATED_POINTS.relative_to(CONFIG_DIR.parent)}")


@app.command()
def fetch(
    source: Annotated[str, typer.Argument(help="entsog | bloomberg | bnef_ldz | agsi")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch a date range from one source into DuckDB at a new vintage."""
    _setup_logging(verbose)
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    perimeter = Perimeter()
    ctx = RunContext.new(f"fetch {source} {start} {end}")

    with Store() as store:
        if source == "entsog":
            points = store.latest_points()
            if points.empty:
                raise typer.BadParameter(
                    "point registry is empty; run `gasbalance resolve-points` first"
                )
            # Internal points are pulled too: they never enter the balance,
            # but netting them to zero is the Phase 1 double-counting check.
            wanted = set(BALANCE_ROLES) | {ROLE_INTERNAL}
            keys = sorted(points.loc[points["ring_role"].isin(wanted), "point_key"].unique())
            result = EntsogFetcher().fetch(start_d, end_d, point_keys=keys)
        elif source == "bnef_ldz":
            result = BnefLdzFetcher().fetch(start_d, end_d)
        elif source == "bloomberg":
            result = BloombergFetcher().fetch(
                start_d, end_d, countries=sorted(perimeter.inside)
            )
        elif source == "agsi":
            result = AgsiFetcher().fetch(start_d, end_d, countries=sorted(perimeter.inside))
        else:
            raise typer.BadParameter(f"unknown source {source!r}")

        frame = result.frame.copy()
        frame["run_id"] = ctx.run_id
        store.register_run(ctx)
        written = store.write_observations(frame, ctx)

    typer.echo(f"source        {result.source}")
    typer.echo(f"vintage       {result.vintage.isoformat()}")
    typer.echo(f"rows raw      {result.rows_raw}")
    typer.echo(f"rows written  {written}")
    if result.drop_reasons:
        typer.echo("dropped (not filled, not interpolated):")
        for reason, count in result.drop_reasons.items():
            typer.echo(f"  {count:6d}  {reason}")


@app.command("reconcile")
def reconcile_cmd(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    as_of: Annotated[str, typer.Option(help="vintage cutoff, YYYY-MM-DD")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Phase 1 gate: net internal flows to zero and report coverage gaps."""
    _setup_logging(verbose)
    perimeter = Perimeter()
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    cutoff = (
        datetime.fromisoformat(as_of) if as_of else datetime.now(tz=None).replace(tzinfo=None)
    )

    with Store() as store:
        points = store.latest_points()
        obs = store.read_as_of(cutoff, start=start_d, end=end_d)

    if obs.empty:
        typer.echo("no observations in range; run `gasbalance fetch entsog` first")
        raise typer.Exit(1)

    flows = obs[obs["series_id"].str.startswith("physical_flow_")].copy()

    report = reconcile(flows, points, perimeter)
    typer.echo(f"as-of vintage             {cutoff.isoformat()}")
    for line in report.summary_lines():
        typer.echo(line)
    for note in report.notes:
        typer.echo(f"  ! {note}")
    raise typer.Exit(0 if report.passed else 1)


@app.command("reconcile-storage")
def reconcile_storage_cmd(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Phase 1 storage gate: does reported inventory close against reported flows?"""
    _setup_logging(verbose)
    perimeter = Perimeter()
    units = Units()
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)

    with Store() as store:
        obs = store.read_latest(start=start_d, end=end_d)

    storage = obs[obs["source"] == "bloomberg"]
    if storage.empty:
        typer.echo("no storage observations; run `gasbalance fetch bloomberg` first")
        raise typer.Exit(1)

    report = reconcile_storage(
        storage, units, perimeter.tolerances.alert_systematic_gap_mcm_per_d
    )
    for line in report.summary_lines():
        typer.echo(line)
    typer.echo("")
    typer.echo("per country (mcm/d):")
    typer.echo(report.by_country.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    out = Path("output") / f"storage_recon_{start}_{end}.csv"
    report.daily.to_csv(out, index=False)
    typer.echo("")
    typer.echo(f"wrote {out}")


@app.command()
def status() -> None:
    """Report what this install can actually reach right now."""
    _setup_logging(False)
    units = Units()
    perimeter = Perimeter()

    typer.echo(f"config hash        {config_hash()}")
    typer.echo(f"perimeter          {perimeter.name} ({len(perimeter.inside)} countries)")
    typer.echo(f"canonical GCV      {units.canonical_gcv} kWh/m3 (GCV)")
    typer.echo(f"1 bcm              {units.canonical_gcv} TWh")
    typer.echo("")
    typer.echo("sources:")
    for name, fetcher, role in (
        ("entsog", EntsogFetcher(), "flows (public, keyless)"),
        ("bloomberg", BloombergFetcher(), "storage via CGIE* (GIE redistribution)"),
        ("agsi", AgsiFetcher(), "optional - not needed, Bloomberg covers storage"),
    ):
        ok = fetcher.is_available()
        typer.echo(f"  {'OK  ' if ok else 'MISS'}  {name:10s} {role}")

    db = Store()
    points = db.latest_points()
    obs = db.con.execute("SELECT COUNT(*) FROM observations").fetchone()
    typer.echo("")
    typer.echo(f"points in db       {len(points)}")
    typer.echo(f"observations in db {obs[0] if obs else 0}")
    db.close()


def _not_yet(phase: int, what: str) -> None:
    raise typer.Exit(
        typer.echo(
            f"`{what}` is a Phase {phase} deliverable and is not built yet. "
            "SPEC 0: one phase at a time, with sign-off at each gate."
        )
        or 1
    )


@app.command()
def build() -> None:
    """Assemble the balance (Phase 4)."""
    _not_yet(4, "build")


@app.command()
def run() -> None:
    """Run the deterministic model (Phase 4/5)."""
    _not_yet(4, "run")


@app.command()
def backtest() -> None:
    """Backtest with as-of vintages (Phase 2 onward)."""
    _not_yet(2, "backtest")


@app.command()
def report() -> None:
    """One-page HTML summary (Phase 5/6)."""
    _not_yet(5, "report")


if __name__ == "__main__":
    app()
