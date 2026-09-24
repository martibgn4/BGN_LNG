"""Fetcher contract tests. SPEC 6 and SPEC 11.

The behaviour under test is refusal: a fetcher without credentials must raise,
not return synthetic values. These tests do not hit any network.
"""

from __future__ import annotations

from datetime import date

import pytest

from gasbalance.data.agsi import AgsiFetcher
from gasbalance.data.base import CredentialsMissing
from gasbalance.data.bloomberg import BloombergFetcher, BloombergUnavailable
from gasbalance.data.entsog import EntsogFetcher
from gasbalance.data.kpler import KplerApiFetcher, KplerCsvFetcher


def test_agsi_raises_without_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("GIE_API_KEY", raising=False)
    fetcher = AgsiFetcher(cache_dir=tmp_path)
    assert not fetcher.is_available()
    with pytest.raises(CredentialsMissing, match="GIE_API_KEY"):
        fetcher.fetch(date(2025, 1, 1), date(2025, 1, 2), countries=["DE"])


def test_agsi_error_message_names_the_env_var(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GIE_API_KEY", raising=False)
    with pytest.raises(CredentialsMissing) as exc:
        AgsiFetcher(cache_dir=tmp_path).credential()
    assert "will not return placeholder data" in str(exc.value)


def test_agsi_blank_key_is_treated_as_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GIE_API_KEY", "   ")
    assert not AgsiFetcher(cache_dir=tmp_path).is_available()


def test_kpler_api_raises_without_a_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KPLER_API_KEY", raising=False)
    with pytest.raises(CredentialsMissing, match="KPLER_API_KEY"):
        KplerApiFetcher(cache_dir=tmp_path).fetch(date(2025, 1, 1), date(2025, 1, 2))


def test_kpler_csv_loader_refuses_without_a_real_export(tmp_path) -> None:
    fetcher = KplerCsvFetcher(cache_dir=tmp_path)
    assert not fetcher.is_available()
    with pytest.raises(CredentialsMissing, match="will not invent cargo data"):
        fetcher.fetch(date(2025, 1, 1), date(2025, 1, 2))


def test_bloomberg_refuses_without_xbbg(tmp_path) -> None:
    """Where xbbg is absent the fetcher must refuse, naming the real fix."""
    fetcher = BloombergFetcher(cache_dir=tmp_path)
    try:
        import xbbg  # noqa: F401
    except ImportError:
        assert not fetcher.is_available()
        with pytest.raises(BloombergUnavailable, match="blpapi"):
            fetcher.fetch(date(2025, 1, 1), date(2025, 1, 2), countries=["DE"])
    else:
        assert fetcher.is_available()


def test_bloomberg_requires_an_explicit_country_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit `countries` list"):
        BloombergFetcher(cache_dir=tmp_path).fetch(date(2025, 1, 1), date(2025, 1, 2))


def test_bloomberg_country_without_a_declared_code_raises(tmp_path) -> None:
    """A new ring country must be mapped deliberately, not silently skipped."""
    from gasbalance.data.base import SourceUnavailable

    fetcher = BloombergFetcher(cache_dir=tmp_path)
    with pytest.raises(SourceUnavailable, match="no Bloomberg code"):
        fetcher.storage_tickers(["ZZ"])


def test_storage_free_countries_are_declared_not_dropped(tmp_path) -> None:
    """IE and LU have no storage; that is a recorded fact, not a missing series."""
    fetcher = BloombergFetcher(cache_dir=tmp_path)
    tickers = fetcher.storage_tickers(["IE", "LU", "DE"])
    assert not any(t.startswith(("CGIEIE", "CGIELU")) for t in tickers)
    assert "CGIEDEST Index" in tickers
    assert set(fetcher.config()["countries_without_storage"]) >= {"IE", "LU"}


def test_uk_maps_to_the_bloomberg_gb_code(tmp_path) -> None:
    """ENTSOG says UK, Bloomberg says GB. The two vocabularies must not mix."""
    tickers = BloombergFetcher(cache_dir=tmp_path).storage_tickers(["UK"])
    assert "CGIEGBST Index" in tickers
    assert tickers["CGIEGBST Index"] == ("UK", "gas_in_storage")
    assert not any(t.startswith("CGIEUK") for t in tickers)


def test_entsog_needs_no_credentials(tmp_path) -> None:
    """Verified on the wire: ENTSOG answers without a token."""
    fetcher = EntsogFetcher(cache_dir=tmp_path)
    assert fetcher.requires_credentials is False
    assert fetcher.is_available()
