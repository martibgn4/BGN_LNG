"""Command line: fetch, panel, fit, backtest, pricer-params, status.

Every command prints what it used and what it could not get. A run that
silently dropped half its features is the failure mode worth engineering
against, so the feature manifest is printed on every command that builds a
panel, not hidden behind a verbose flag.
"""

from __future__ import annotations

import logging

import pandas as pd
import typer

from nwebasis import features, store
from nwebasis.config import config_hash, get
from nwebasis.data import spark_api
from nwebasis.model import ecm, envelope, seasonal_ou

app = typer.Typer(add_completion=False, help=__doc__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

_DATASETS = {
    "basis": "spark_basis_nwe",
    "swe": "spark_basis_swe",
    "freight_atlantic": "spark_freight_atlantic",
    "freight_pacific": "spark_freight_pacific",
    "netbacks": "spark_netback_sabine",
    "slots": "spark_slots",
}


@app.command()
def status() -> None:
    """What is cached, how far back, and which features that enables."""
    typer.echo(f"config hash: {config_hash()}")
    for label, dataset in _DATASETS.items():
        try:
            frame = store.read_latest(dataset)
            date_column = "release_date" if "release_date" in frame else "ReleaseDate"
            dates = pd.to_datetime(frame[date_column])
            typer.echo(f"  {label:<18} {len(frame):>7} rows  "
                       f"{dates.min().date()} .. {dates.max().date()}")
        except FileNotFoundError:
            typer.echo(f"  {label:<18} {'not fetched':>7}")


@app.command()
def fetch(what: str = typer.Argument(..., help="basis|swe|freight|netbacks|slots|all"),
          limit: int = typer.Option(0, help="release count; 0 uses the config default")) -> None:
    """Pull from Spark and append to the vintaged cache."""
    token = spark_api.access_token()
    limit_arg = limit or None
    jobs = {
        "basis": lambda: (_DATASETS["basis"],
                          spark_api.BasisCurveFetcher().fetch("nwe_basis_monthly", limit_arg, token)),
        "swe": lambda: (_DATASETS["swe"],
                        spark_api.BasisCurveFetcher().fetch("swe_basis_monthly", limit_arg, token)),
        "netbacks": lambda: (_DATASETS["netbacks"],
                             spark_api.NetbackFetcher().fetch("sabine_pass", None, token)),
        "slots": lambda: (_DATASETS["slots"], spark_api.TerminalSlotFetcher().fetch(token)),
    }
    if what in ("freight", "all"):
        for route, dataset in (("atlantic_spot", "freight_atlantic"),
                               ("pacific_spot", "freight_pacific")):
            result = spark_api.FreightFetcher().fetch(route, limit_arg, token)
            path = store.write(_DATASETS[dataset], result.frame, result.vintage)
            typer.echo(f"{dataset}: {result.rows} rows -> {path.name}")
    for name, job in jobs.items():
        if what not in (name, "all"):
            continue
        dataset, result = job()
        path = store.write(dataset, result.frame, result.vintage)
        typer.echo(f"{name}: {result.rows} rows -> {path.name}")


@app.command()
def panel() -> None:
    """Build the design matrix and print the feature manifest."""
    frame, manifest = features.build_panel()
    typer.echo(manifest.report())
    typer.echo("")
    typer.echo(frame.tail().to_string())


@app.command("pricer-params")
def pricer_params(tenor: str = typer.Option("M+1"),
                  sample_start: str = typer.Option("")) -> None:
    """Fit the seasonal OU and print the block NWE_option_pricer.py wants."""
    frame, manifest = features.build_panel()
    table, offsets = seasonal_ou.fit_by_tenor(frame, sample_start or None)
    typer.echo(table.to_string(index=False))
    typer.echo("\ndelivery-month offsets (USD/MMBtu, centred):")
    typer.echo(offsets.round(3).to_string())

    row = table[table["tenor"] == tenor]
    if row.empty or "error" in row.columns and row["error"].notna().all():
        typer.echo(f"\n{tenor}: no usable fit; see the table above.")
        raise typer.Exit(code=1)
    parameters = seasonal_ou.OUParameters(**{
        key: row.iloc[0][key] for key in seasonal_ou.OUParameters.__dataclass_fields__
    })
    typer.echo("\n--- paste into adhoc_scripts/NWE_option_pricer.py ---")
    typer.echo(parameters.as_pricer_block())


@app.command()
def fit(tenor: str = typer.Option("M+1"),
        sample_start: str = typer.Option("", help="default: model.sample.stable_start")) -> None:
    """Fit the error-correction model on one tenor and print the coefficients."""
    frame, manifest = features.build_panel()
    typer.echo(manifest.report())
    frame = _apply_sample(frame, sample_start)
    anchor, short_run = _blocks(frame)
    subset = frame[frame["tenor"] == tenor].sort_values("release_date")
    model = ecm.fit(subset, anchor, short_run)
    typer.echo(f"\nn={model.n_observations}  R2(differences)={model.r_squared:.3f}  "
               f"residual sd={model.residual_sd:.4f}  "
               f"implied half-life={model.half_life_days:.1f}d")
    typer.echo(model.summary().to_string(index=False))


@app.command()
def backtest(tenor: str = typer.Option("M+1"), horizon: int = typer.Option(21),
             sample_start: str = typer.Option("", help="default: model.sample.stable_start")) -> None:
    """Walk forward and score against the market benchmarks."""
    from nwebasis import backtest as bt

    frame, manifest = features.build_panel()
    typer.echo(manifest.report())
    frame = _apply_sample(frame, sample_start)
    anchor, short_run = _blocks(frame)
    predictions, table = bt.run(frame, anchor, short_run, horizon, tenor)
    typer.echo("")
    typer.echo(table.to_string(index=False))
    if "directional" in table.attrs:
        typer.echo("\ndirectional test vs the market:")
        for key, value in table.attrs["directional"].items():
            typer.echo(f"  {key:<26} {value}")


@app.command("cost-anchor")
def cost_anchor() -> None:
    """Report the regas cost floor, or exactly which entries are unset."""
    try:
        stack = envelope.cost_stack()
    except envelope.RegasCostUnset as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)
    typer.echo(f"capacity-weighted tariff: {stack.capacity_weighted_tariff:.3f}")
    for key, value in stack.adders.items():
        typer.echo(f"  + {key:<28} {value:.3f}")
    typer.echo(f"implied basis floor: {stack.implied_basis_floor:.3f} USD/MMBtu")


def _apply_sample(frame: pd.DataFrame, sample_start: str) -> pd.DataFrame:
    """Cut the panel to the estimation window, and say so out loud.

    The default is ``model.sample.stable_start``, not the full history. Fitted
    across the 2022 dislocation the same estimator returns a mean of -0.687 and
    a standard deviation of 1.294 against -0.388 and 0.192 on the stable
    sample - an average of a capacity crisis and its absence, which describes
    no market that has traded since. Pass --sample-start 2022-11-29 to fit the
    whole thing deliberately.
    """
    start = sample_start or get("model", "sample", "stable_start")
    cut = frame[frame["release_date"] >= pd.Timestamp(start)]
    dropped = len(frame) - len(cut)
    typer.echo("")
    typer.echo(f"sample: from {start} ({len(cut)} rows, {dropped} earlier rows excluded)")
    if cut.empty:
        raise typer.BadParameter(f"no rows on or after {start}")
    return cut


def _blocks(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Which columns actually made it into the panel, split long-run vs short-run."""
    long_run = [c for c in ("log_free_slots", "ttf_time_spread", "month_sin", "month_cos")
                if c in frame.columns and frame[c].notna().any()]
    short_run = [c for c in ("arb_nea_minus_nwe", "d_freight_atlantic",
                             "d_freight_basin", "swe_nwe_spread")
                 if c in frame.columns and frame[c].notna().any()]
    if not long_run:
        raise typer.BadParameter("no long-run features available; run `nwebasis fetch all`")
    return long_run, short_run


if __name__ == "__main__":
    app()
