"""Command-line workflow.

    python -m ttf_forecaster.cli synthetic-e2e [--worlds null planted] [--data-root DIR]
    python -m ttf_forecaster.cli identifiers           # vendor identifier status summary
    python -m ttf_forecaster.cli check-config          # load and validate every config file

No command makes a live vendor call.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from ttf_forecaster.settings import CONFIG_DIR, PROJECT_ROOT


def _cmd_synthetic(args: argparse.Namespace) -> int:
    from ttf_forecaster.pipelines.synthetic_e2e import run_world
    from ttf_forecaster.reporting.research_report import write_report

    results = [run_world(w, data_root=args.data_root, run_tag=args.run_tag) for w in args.worlds]
    out = Path(args.report or PROJECT_ROOT / "outputs" / "synthetic_e2e_report.md")
    write_report(results, out)
    for r in results:
        c1 = {m: c["criterion_1"]["passed"] for m, c in r.criteria.items()}
        print(f"[SYNTHETIC] world={r.world} run_id={r.run_id} criterion_1={c1}")
    print(f"[SYNTHETIC] report: {out}")
    return 0


def _cmd_real_h1(args: argparse.Namespace) -> int:
    from ttf_forecaster.pipelines.real_h1 import run_h1
    from ttf_forecaster.reporting.research_report import write_report
    from ttf_forecaster.settings import load_config

    results = run_h1(run_tag=args.run_tag)
    out = Path(args.report or PROJECT_ROOT / load_config("pipeline")["real"]["report"])
    write_report(results, out, synthetic=False,
                 title="H1 — EU storage vs the TTF forward curve (real data)",
                 intro="Data: TZT monthly settlements 2015+ (vendor last-trade dates), GIE AGSI EU "
                       "storage via CGIEEU* 2011+, EURUSD; pulled 2026-10-19 under explicit user "
                       "authorisation. Same evaluation code as the synthetic run "
                       "(`pipelines.synthetic_e2e.evaluate_bundle`).")
    for r in results:
        print(f"[REAL] {r.world} run_id={r.run_id} headline={r.headline_pass}")
    print(f"[REAL] report: {out}")
    return 0


def _iter_series(node: object, path: str = ""):  # type: ignore[no-untyped-def]
    if isinstance(node, dict):
        if "status" in node and ("ticker" in node or "unit" in node or "vendor" in node):
            yield path, node
        for k, v in node.items():
            yield from _iter_series(v, f"{path}.{k}" if path else str(k))


def _cmd_identifiers(_: argparse.Namespace) -> int:
    totals: Counter[str] = Counter()
    for f in sorted((CONFIG_DIR / "sources").glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        counts = Counter(str(n.get("status")) for _, n in _iter_series(data))
        totals.update(counts)
        print(f"{f.name:32s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("TOTAL".ljust(33) + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    return 0


def _cmd_check_config(_: argparse.Namespace) -> int:
    bad = 0
    for f in sorted(CONFIG_DIR.rglob("*.yaml")):
        if f.name.startswith("local"):
            continue
        try:
            yaml.safe_load(f.read_text(encoding="utf-8"))
            print(f"ok   {f.relative_to(CONFIG_DIR)}")
        except yaml.YAMLError as exc:
            bad += 1
            print(f"FAIL {f.relative_to(CONFIG_DIR)}: {exc}")
    from ttf_forecaster.data.bloomberg import make_market_adapter
    from ttf_forecaster.data.fundamentals import load_fundamentals_registry

    make_market_adapter("mock")
    load_fundamentals_registry()
    print("ok   adapter registries load (mock)")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point (``ttf-forecaster`` console script)."""
    parser = argparse.ArgumentParser(prog="ttf-forecaster")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic-e2e", help="run the SYNTHETIC end-to-end pipeline")
    s.add_argument("--worlds", nargs="+", default=["null", "planted"], choices=["null", "planted"])
    s.add_argument("--data-root", default=None, help="store root (default ./data/synthetic_runs)")
    s.add_argument("--run-tag", default=None)
    s.add_argument("--report", default=None)
    s.set_defaults(func=_cmd_synthetic)
    rh = sub.add_parser("real-h1", help="H1 on the stored live pull (no vendor calls)")
    rh.add_argument("--run-tag", default=None)
    rh.add_argument("--report", default=None)
    rh.set_defaults(func=_cmd_real_h1)
    sub.add_parser("identifiers", help="vendor identifier status").set_defaults(
        func=_cmd_identifiers)
    sub.add_parser("check-config", help="validate configs").set_defaults(func=_cmd_check_config)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
