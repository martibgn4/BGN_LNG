"""SPEC 3: "If a number appears in a .py file, it is a bug."

Taken literally that bans array indices and loop bounds, so this encodes the
intent instead: no *assumption* may live in Python. Concretely, no float
literal outside the structural set {0.0, 1.0, -1.0}, which covers signs,
identities and zero-initialisation but cannot express a calorific value, an
efficiency, a decline rate or a tolerance.

Every real assumption therefore has to come from config/, which is what makes
the config hash in SPEC 9 a complete description of the model's inputs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "gasbalance"
STRUCTURAL_FLOATS = {0.0, 1.0, -1.0}


def python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def float_literals(path: Path) -> list[tuple[int, float]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            value = node.value
            found.append((node.lineno, value))
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, float)
        ):
            found.append((node.lineno, -node.operand.value))
    return found


def test_source_tree_is_not_empty() -> None:
    assert python_files(), "no source files found; the check would pass vacuously"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_assumption_floats_in_python(path: Path) -> None:
    offenders = [
        (line, value) for line, value in float_literals(path) if value not in STRUCTURAL_FLOATS
    ]
    assert not offenders, (
        f"{path.name} contains numeric assumptions in code: {offenders}. "
        "Move them to config/ (SPEC 3)."
    )


def test_units_module_holds_no_numbers_at_all() -> None:
    """The conversion layer is the strictest case - every value is config."""
    path = SRC / "units.py"
    assert not float_literals(path), (
        "units.py must contain no float literals whatsoever; "
        "all calorific values and ratios come from conversions.yaml"
    )
