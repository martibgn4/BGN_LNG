"""No numeric constant may live in a .py file.

Same rule and same enforcement as GasImbalanceModel and NWEBasisModel. A base
temperature, a stress threshold or a cost assumption buried in source is a
number nobody reviews and nobody can find later. Everything tunable goes in
config/.

The rule is on FLOAT literals. Integers are structure - a slice bound, a lag of
one, twelve months in a year - and naming them in YAML would obscure rather
than document. A float is almost always a calibrated quantity.

0.0, 1.0 and -1.0 stay allowed: they are identity and sign, not assumptions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "wxreturns"
ALLOWED = {0.0, 1.0, -1.0}


def python_files() -> list[Path]:
    return sorted(SOURCE_DIR.rglob("*.py"))


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_float_literals(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, float):
            continue
        if node.value in ALLOWED:
            continue
        offenders.append((node.lineno, node.value))
    assert not offenders, (
        f"{path.name} carries float literals {offenders}. Move them into "
        "config/ - a calibrated number in source is an unreviewed assumption."
    )
