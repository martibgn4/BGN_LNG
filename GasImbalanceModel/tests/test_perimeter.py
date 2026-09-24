"""Perimeter and netting tests. SPEC 7 Phase 1.

The fixtures here are hand-built logic fixtures with the ENTSOG column layout.
They are not market data and are never written to the database - they exist to
exercise the classification and netting rules.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from gasbalance.perimeter import (
    BALANCE_ROLES,
    ROLE_CROSSING,
    ROLE_INTERNAL,
    ROLE_LNG,
    ROLE_STORAGE,
    Perimeter,
    PerimeterError,
)

CROSS_TYPE = "Cross-Border Transmission IP within EU"


def registry_row(
    point_key: str,
    operator_key: str,
    direction: str,
    tso_country: str,
    adjacent_country: str | None,
    point_type: str = CROSS_TYPE,
) -> dict[str, object]:
    return {
        "pointKey": point_key,
        "pointLabel": point_key,
        "operatorKey": operator_key,
        "operatorLabel": operator_key,
        "directionKey": direction,
        "pointType": point_type,
        "crossBorderPointType": "Cross-Border EU|EU",
        "tSOCountry": tso_country,
        "adjacentCountry": adjacent_country,
        "hasData": True,
        "isDoubleReporting": False,
        "isPipeInPipe": False,
        "tpTsoGCVMin": "36.0",
        "tpTsoGCVMax": "44.0",
        "tpTsoGCVUnit": "MJ/nm3",
    }


@pytest.fixture(scope="module")
def perimeter() -> Perimeter:
    return Perimeter()


@pytest.fixture
def registry() -> pd.DataFrame:
    """The Easington/IUK case from SPEC 7, plus an LNG and a storage point."""
    return pd.DataFrame(
        [
            # Norway -> UK at Easington. NO is outside the ring: one import.
            registry_row("ITP-EASINGTON", "UK-TSO-0001", "entry", "UK", "NO"),
            # UK -> BE through the IUK. Both inside: one internal transfer.
            registry_row("ITP-IUK", "UK-TSO-0001", "exit", "UK", "BE"),
            registry_row("ITP-IUK", "BE-TSO-0001", "entry", "BE", "UK"),
            # UK -> NL through the BBL. Also internal.
            registry_row("ITP-BBL", "UK-TSO-0001", "exit", "UK", "NL"),
            registry_row("ITP-BBL", "NL-TSO-0001", "entry", "NL", "UK"),
            # DE -> CH. CH is outside this first cut: a perimeter exit.
            registry_row("ITP-WALLBACH", "DE-TSO-0001", "exit", "DE", "CH"),
            # An LNG terminal and a storage site inside the ring.
            registry_row("LNG-ZEEBRUGGE", "BE-TSO-0001", "entry", "BE", None, "LNG Entry point"),
            registry_row("STO-BERGERMEER", "NL-TSO-0001", "entry", "NL", None, "Storage point"),
            # A point wholly outside the ring.
            registry_row("ITP-BAUMGARTEN-SK", "SK-TSO-0001", "exit", "SK", "AT"),
        ]
    )


# -- classification ---------------------------------------------------------


def test_easington_is_an_import_and_iuk_is_internal(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    """The exact double-counting case SPEC 7 names as the top failure mode."""
    roles = perimeter.classify(registry).set_index(["pointKey", "operatorKey"])["ring_role"]
    assert roles[("ITP-EASINGTON", "UK-TSO-0001")] == ROLE_CROSSING
    assert roles[("ITP-IUK", "UK-TSO-0001")] == ROLE_INTERNAL
    assert roles[("ITP-IUK", "BE-TSO-0001")] == ROLE_INTERNAL
    assert roles[("ITP-BBL", "UK-TSO-0001")] == ROLE_INTERNAL


def test_exit_to_a_country_outside_the_ring_is_a_crossing(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    roles = perimeter.classify(registry).set_index("pointKey")["ring_role"]
    assert roles["ITP-WALLBACH"] == ROLE_CROSSING


def test_lng_and_storage_are_separated_from_pipeline_crossings(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    roles = perimeter.classify(registry).set_index("pointKey")["ring_role"]
    assert roles["LNG-ZEEBRUGGE"] == ROLE_LNG
    assert roles["STO-BERGERMEER"] == ROLE_STORAGE
    # Storage is a time-shifter, not supply (SPEC 8), so it is not a perimeter term.
    assert ROLE_STORAGE not in BALANCE_ROLES


def test_points_outside_the_ring_are_not_in_the_balance(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    classified = perimeter.classify(registry)
    outside = classified[classified["pointKey"] == "ITP-BAUMGARTEN-SK"]
    assert not outside["ring_role"].isin(BALANCE_ROLES).any()


def test_unknown_country_raises_rather_than_defaulting_to_outside(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    rogue = pd.concat(
        [registry, pd.DataFrame([registry_row("ITP-NEW", "ZZ-TSO-0001", "entry", "ZZ", "DE")])],
        ignore_index=True,
    )
    with pytest.raises(PerimeterError, match="not listed in perimeter.yaml"):
        perimeter.classify(rogue)


def test_unknown_point_type_raises(perimeter: Perimeter, registry: pd.DataFrame) -> None:
    odd = registry.copy()
    odd.loc[0, "pointType"] = "Some New ENTSOG Category"
    with pytest.raises(PerimeterError, match="perimeter_point_types"):
        perimeter.classify(odd)


# -- netting ----------------------------------------------------------------


def flow_row(point_key: str, operator_key: str, direction: str, value: float) -> dict[str, object]:
    return {
        "obs_date": date(2025, 1, 15),
        "point_key": point_key,
        "operator_key": operator_key,
        "direction_key": direction,
        "value": value,
    }


def test_internal_flows_net_to_zero(perimeter: Perimeter, registry: pd.DataFrame) -> None:
    """SPEC 7: "a test that asserts internal flows sum to zero net"."""
    points = perimeter.to_points_table(perimeter.classify(registry))
    flows = pd.DataFrame(
        [
            flow_row("ITP-EASINGTON", "UK-TSO-0001", "entry", 120.0),
            # Both TSOs report the same IUK flow, one as exit, one as entry.
            flow_row("ITP-IUK", "UK-TSO-0001", "exit", 30.0),
            flow_row("ITP-IUK", "BE-TSO-0001", "entry", 30.0),
            flow_row("ITP-BBL", "UK-TSO-0001", "exit", 20.0),
            flow_row("ITP-BBL", "NL-TSO-0001", "entry", 20.0),
        ]
    )
    signed = perimeter.signed_flows(flows, points)
    net = perimeter.internal_net(signed)
    assert net["internal_net_mcm_d"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert perimeter.internal_net_breaches(net).empty


def test_tso_disagreement_on_an_internal_point_is_surfaced(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    """A gap between the two sides must show up, not be absorbed."""
    points = perimeter.to_points_table(perimeter.classify(registry))
    gap = perimeter.tolerances.internal_net_abs_mcm * 10
    flows = pd.DataFrame(
        [
            flow_row("ITP-IUK", "UK-TSO-0001", "exit", 30.0),
            flow_row("ITP-IUK", "BE-TSO-0001", "entry", 30.0 + gap),
        ]
    )
    net = perimeter.internal_net(perimeter.signed_flows(flows, points))
    breaches = perimeter.internal_net_breaches(net)
    assert len(breaches) == 1
    assert breaches["internal_net_mcm_d"].iloc[0] == pytest.approx(gap)


def test_internal_transfer_does_not_reach_the_perimeter_balance(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    """The Easington import counts once; the onward IUK leg counts zero times."""
    points = perimeter.to_points_table(perimeter.classify(registry))
    flows = pd.DataFrame(
        [
            flow_row("ITP-EASINGTON", "UK-TSO-0001", "entry", 120.0),
            flow_row("ITP-IUK", "UK-TSO-0001", "exit", 30.0),
            flow_row("ITP-IUK", "BE-TSO-0001", "entry", 30.0),
        ]
    )
    perim = perimeter.perimeter_flows(perimeter.signed_flows(flows, points))
    crossing = perim.loc[perim["ring_role"] == ROLE_CROSSING, "net_mcm_d"].sum()
    assert crossing == pytest.approx(120.0)


def test_flow_with_no_registry_entry_raises(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    points = perimeter.to_points_table(perimeter.classify(registry))
    flows = pd.DataFrame([flow_row("ITP-GHOST", "UK-TSO-0001", "entry", 10.0)])
    with pytest.raises(PerimeterError, match="no entry in the point registry"):
        perimeter.signed_flows(flows, points)


# -- config sanity ----------------------------------------------------------


def test_ring_countries_are_disjoint(perimeter: Perimeter) -> None:
    assert not (perimeter.inside & perimeter.outside)


def test_norway_is_outside_the_ring(perimeter: Perimeter) -> None:
    """Norway is a supply source, never a demand country inside the balance."""
    assert "NO" in perimeter.outside
    assert "NO" not in perimeter.inside


def test_luxembourg_is_inside_with_belgium(perimeter: Perimeter) -> None:
    """BeLux is one balancing zone; splitting it invents a perimeter crossing."""
    assert {"BE", "LU"} <= perimeter.inside


def test_country_codes_survive_yaml_boolean_coercion(perimeter: Perimeter) -> None:
    """An unquoted NO in YAML parses as False and deletes Norway from the ring."""
    for code in perimeter.inside | perimeter.outside:
        assert isinstance(code, str), f"{code!r} was coerced by the YAML parser"
        assert len(code) == 2 and code.isalpha()


def test_bare_no_in_config_is_rejected() -> None:
    with pytest.raises(PerimeterError, match="boolean False"):
        Perimeter._validate_country_lists({"inside": ["DE"], "outside": [False]})


# -- matched pairs vs one-sided coverage ------------------------------------


def test_one_sided_internal_point_is_not_reported_as_a_netting_failure(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    """A missing counterparty side is a coverage gap, not a double-count.

    Without this split, any internal point whose second TSO failed to publish
    shows up as a large netting breach, and the model gets blamed for a hole in
    the data.
    """
    points = perimeter.to_points_table(perimeter.classify(registry))
    flows = pd.DataFrame([flow_row("ITP-IUK", "UK-TSO-0001", "exit", 30.0)])
    pairs = perimeter.internal_pairs(perimeter.signed_flows(flows, points))

    assert not pairs["matched"].any()
    assert perimeter.internal_net_matched(pairs).empty
    orphans = perimeter.internal_orphans(pairs)
    assert len(orphans) == 1
    assert orphans["gross_mcm_d"].iloc[0] == pytest.approx(30.0)


def test_matched_pair_is_counted_and_orphan_is_excluded(
    perimeter: Perimeter, registry: pd.DataFrame
) -> None:
    points = perimeter.to_points_table(perimeter.classify(registry))
    flows = pd.DataFrame(
        [
            flow_row("ITP-IUK", "UK-TSO-0001", "exit", 30.0),
            flow_row("ITP-IUK", "BE-TSO-0001", "entry", 30.0),
            flow_row("ITP-BBL", "UK-TSO-0001", "exit", 20.0),  # no counterparty
        ]
    )
    pairs = perimeter.internal_pairs(perimeter.signed_flows(flows, points))
    net = perimeter.internal_net_matched(pairs)

    assert net["internal_net_mcm_d"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert len(perimeter.internal_orphans(pairs)) == 1
