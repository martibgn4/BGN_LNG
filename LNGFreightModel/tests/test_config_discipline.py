"""Config discipline, carried over from the sibling models.

A model whose numbers live in its code cannot be reproduced from a config
diff, and every one of these projects has the same rule: the .py files hold
structure, the .yaml files hold quantities.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lngfreight"

# Values that are structural rather than tunable: identity, halving, a
# percentage, the two-sided factor in a p-value, degree-1 polynomial fits.
ALLOWED = {0, 1, 2, 5, 100, 365.25, 0.0, 1.0, 2.0, 0.5, 1e-9, 3, 4,
           60,      # seconds per minute, in the cache-age conversion
           70}      # width of a printed separator rule


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_tunable_literals_in_code(path: Path):
    """Numeric literals in .py must be structural, not model choices."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            continue
        if node.value in ALLOWED:
            continue
        offenders.append((node.lineno, node.value))
    assert not offenders, (
        f"{path.name} carries tunable numeric literals {offenders}. Move them "
        "into config/*.yaml so the result is reproducible from the config."
    )


def test_every_source_declares_a_publication_lag():
    """A source with no declared lag would silently be treated as same-day."""
    from lngfreight.config import load
    sources = load("sources")

    def walk(node, trail):
        if isinstance(node, dict):
            if "ticker" in node or "id" in node:
                yield trail, node
            for key, value in node.items():
                yield from walk(value, trail + [key])

    missing = []
    for trail, node in walk(sources, []):
        # A lag may be declared on the spec itself or on any ancestor block -
        # the on_water and routing families set one lag for the whole block.
        declared = "publication_lag_days" in node
        ancestor = sources
        for key in trail:
            if not isinstance(ancestor, dict):
                break
            if "publication_lag_days" in ancestor:
                declared = True
            ancestor = ancestor.get(key, {})
        if not declared:
            missing.append(".".join(trail))
    # The unentitled block is bare strings, not source specs, so it is exempt.
    missing = [m for m in missing if "unentitled" not in m]
    assert not missing, f"no publication lag declared for: {missing}"


def test_horizons_match_the_brief():
    """1 day, 1 week, 2 weeks, 3 weeks in business days."""
    from lngfreight.config import get
    assert get("target", "target", "horizons_bd") == [1, 5, 10, 15]
