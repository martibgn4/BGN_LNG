"""Spark Commodities. The only source this model can reach without a Terminal.

Self-contained rather than importing ``Utils.utils_spark`` from the parent
BGN_LNG repo, for the same reason GasImbalanceModel is self-contained: the
package has its own venv and must not depend on the parent being importable.
The credential file path is shared, so there is one place to rotate keys.

Four fetchers, in descending order of importance to the model:

  BasisCurveFetcher    the target itself, and the SWE twin
  NetbackFetcher       the east-west arb, freight-adjusted by Spark, plus TTF
                       and JKM prices as a by-product
  TerminalSlotFetcher  free regas slots by delivery month - the scarcity variable
  FreightFetcher       Atlantic and Pacific spot day rates
"""

from __future__ import annotations

import json
import logging
import urllib.request
from base64 import b64encode
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin

import pandas as pd

from nwebasis.config import get
from nwebasis.data.base import CredentialsMissing, Fetcher, FetchResult, SourceUnavailable

log = logging.getLogger(__name__)

_TOKEN_URI = "/oauth/token/"
_JSON_ACCEPT = "application/json"
_GRANT_TYPE = "clientCredentials"
_SCOPES = "read:access,read:prices"
_CREDENTIAL_HEADERS = ("clientId,clientSecret", "client_id,client_secret")


def _credentials() -> tuple[str, str]:
    path = Path(get("sources", "spark", "credentials_path"))
    if not path.is_file():
        raise CredentialsMissing(
            f"Spark credentials not found at {path}. Expected a two-line CSV "
            "whose header names the client id and secret."
        )
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if lines[0].replace(" ", "") not in _CREDENTIAL_HEADERS:
        raise CredentialsMissing(f"{path} does not look like a Spark credentials file")
    client_id, client_secret = lines[1].split(",")
    return client_id, client_secret


def access_token() -> str:
    """OAuth2 client-credentials token. Valid ~3h; callers reuse one per run.

    Spark's scheme is not quite standard Basic auth: the Authorization header
    carries the bare base64 of ``clientId:clientSecret`` with no ``Basic``
    prefix, the grant and scopes go in the JSON body, and success is 201 rather
    than 200. Matched to ``Utils/utils_spark.get_access_token``, which is what
    the desk's production report already authenticates with.
    """
    base = get("sources", "spark", "api_base_url")
    client_id, client_secret = _credentials()
    headers = {
        "Authorization": b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        "Accept": _JSON_ACCEPT,
        "Content-Type": _JSON_ACCEPT,
    }
    body = json.dumps({"grantType": _GRANT_TYPE, "scopes": _SCOPES}).encode()
    request = urllib.request.Request(urljoin(base, _TOKEN_URI), data=body, headers=headers)
    try:
        response = urllib.request.urlopen(request)
    except HTTPError as exc:
        raise SourceUnavailable(f"Spark auth failed: HTTP {exc.code}") from exc
    return json.loads(response.read())["accessToken"]


def _get(uri: str, token: str, accept: str = _JSON_ACCEPT) -> object:
    """One authenticated GET. Raises on any non-200 - never returns a partial."""
    base = get("sources", "spark", "api_base_url")
    request = urllib.request.Request(
        urljoin(base, uri), headers={"Authorization": f"Bearer {token}", "accept": accept}
    )
    try:
        raw = urllib.request.urlopen(request).read()
    except HTTPError as exc:
        raise SourceUnavailable(f"Spark GET {uri} failed: HTTP {exc.code}") from exc
    return raw.decode("utf-8") if accept != _JSON_ACCEPT else json.loads(raw)


def _releases(token: str, contract: str, limit: int, vessel: str | None = None) -> list[dict]:
    query = f"?limit={limit}"
    if vessel is not None:
        query += f"&vessel-type={vessel}"
    payload = _get(f"/v1.0/contracts/{contract}/price-releases/{query}", token)
    data = payload["data"]
    if not data:
        raise SourceUnavailable(f"Spark contract {contract!r} returned no releases")
    return data


class BasisCurveFetcher(Fetcher):
    """The NWE basis curve - the forecast target - and the SWE curve beside it.

    Output is long: one row per (release date, delivery month). ``tenor`` is the
    M+n label Spark assigns, ``delivery_start`` the first day of the delivery
    month. Both are kept: the tenor is what rolls, the delivery month is what
    the seasonal attaches to, and conflating them is how a seasonal ends up
    fitted to the observation month by accident.
    """

    source = "spark_basis"

    def fetch(self, contract_key: str = "nwe_basis_monthly", limit: int | None = None,
              token: str | None = None) -> FetchResult:
        cfg = get("sources", "spark", "contracts", contract_key)
        token = token or access_token()
        limit = limit or _default_release_limit()
        rows = []
        for release in _releases(token, cfg["id"], limit):
            for block in release["data"]:
                for point in block["dataPoints"]:
                    derived = (point.get("derivedPrices") or {}).get(cfg["price_key"]) or {}
                    if derived.get("spark") is None:
                        continue
                    period = point["deliveryPeriod"]
                    rows.append({
                        "release_date": release["releaseDate"],
                        "contract": release["contractId"],
                        "tenor": period.get("name"),
                        "delivery_start": period["startAt"],
                        "value": float(derived["spark"]),
                        "value_min": _maybe_float(derived.get("sparkMin")),
                        "value_max": _maybe_float(derived.get("sparkMax")),
                    })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["release_date"] = pd.to_datetime(frame["release_date"]).dt.tz_localize(None)
            frame["delivery_start"] = pd.to_datetime(frame["delivery_start"])
        return self._result(frame, f"{self.source}:{contract_key}")


class FreightFetcher(Fetcher):
    """Atlantic (Spark30S) and Pacific (Spark25S) spot day rates."""

    source = "spark_freight"

    def fetch(self, route_key: str = "atlantic_spot", limit: int | None = None,
              token: str | None = None) -> FetchResult:
        cfg = get("sources", "spark", "freight", route_key)
        vessel = get("sources", "spark", "freight", "vessel_type")
        token = token or access_token()
        limit = limit or _default_release_limit()
        rows = []
        for release in _releases(token, cfg["id"], limit, vessel=vessel):
            for block in release["data"]:
                for point in block["dataPoints"]:
                    derived = (point.get("derivedPrices") or {}).get(cfg["price_key"]) or {}
                    if derived.get("spark") is None:
                        continue
                    rows.append({
                        "release_date": release["releaseDate"],
                        "route": route_key,
                        "delivery_start": point["deliveryPeriod"]["startAt"],
                        "usd_per_day": float(derived["spark"]),
                    })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["release_date"] = pd.to_datetime(frame["release_date"]).dt.tz_localize(None)
        return self._result(frame, f"{self.source}:{route_key}")


class NetbackFetcher(Fetcher):
    """East-west arbitrage from a US Gulf load port, already freight-adjusted.

    ``via-point`` is mandatory. Omit it and the endpoint answers HTTP 200 with
    an empty data list - no error, no warning. That silent-empty case is turned
    into a logged skip here so it can never reach the design matrix as a zero.

    One call per (port, release date), so a full history pull is slow. Results
    are cached and vintaged; pass ``release_dates`` to fetch only what is new.
    """

    source = "spark_netback"

    def fetch(self, port_key: str = "sabine_pass", release_dates: list[str] | None = None,
              token: str | None = None) -> FetchResult:
        cfg = get("sources", "spark", "netbacks", "fob_ports", port_key)
        endpoint = get("sources", "spark", "netbacks", "endpoint")
        token = token or access_token()
        if release_dates is None:
            release_dates = self.available_releases(token)
        rows = []
        for release_date in release_dates:
            uri = (f"{endpoint}?fob-port={cfg['uuid']}"
                   f"&via-point={cfg['via_point']}&release-date={release_date}")
            try:
                payload = _get(uri, token)
            except SourceUnavailable as exc:
                log.warning("netback %s %s unavailable: %s", port_key, release_date, exc)
                continue
            data = payload.get("data")
            if not data or not data.get("netbacks"):
                log.warning("netback %s %s returned an empty payload; skipped",
                            port_key, release_date)
                continue
            for point in data["netbacks"]:
                rows.append({
                    "release_date": release_date,
                    "fob_port": port_key,
                    "month_index": point["monthIdx"],
                    "load_month": point["load"]["month"],
                    "nwe_delivery_date": point.get("nweCargoDeliveryDate"),
                    "arb_nea_minus_nwe": _nested(point, "neaMinusNwe", "ttfBasis"),
                    "nwe_netback_basis": _nested(point, "nwe", "ttfBasis"),
                    "nea_netback_basis": _nested(point, "nea", "ttfBasis"),
                    "route_cost_nwe": _meta(point, "nweMeta", "routeCost"),
                    "route_cost_nea": _meta(point, "neaMeta", "routeCost"),
                    "ttf_price": _meta(point, "nweMeta", "ttfPrice"),
                    "jkm_price": _meta(point, "neaMeta", "desLngPrice"),
                })
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["release_date"] = pd.to_datetime(frame["release_date"])
            frame["nwe_delivery_date"] = pd.to_datetime(frame["nwe_delivery_date"])
        return self._result(frame, f"{self.source}:{port_key}")

    @staticmethod
    def available_releases(token: str) -> list[str]:
        uri = get("sources", "spark", "netbacks", "reference_endpoint")
        payload = _get(uri, token)
        releases = payload["data"]["staticData"].get("sparkReleases")
        if not releases:
            raise SourceUnavailable("netback reference data carried no release dates")
        return list(releases)


class TerminalSlotFetcher(Fetcher):
    """Unsold regas slots by terminal and delivery month.

    A COUNT OF FREE SLOTS: falling means tightening. The sign trips people up,
    so it is restated wherever the series is used.

    Coverage starts 2023-11-28 for the NWE terminals and later for several
    others, which is the binding constraint on how far back the full-feature
    model can be fitted. ``coverage`` reports it rather than leaving the model
    to discover it as a wall of NaN.
    """

    source = "spark_slots"

    def fetch(self, token: str | None = None,
              terminal_codes: list[str] | None = None) -> FetchResult:
        token = token or access_token()
        accept = get("sources", "spark", "terminal_slots", "accept_header")
        listing = self.terminal_list(token)
        if terminal_codes is not None:
            listing = listing[listing["TerminalCode"].isin(terminal_codes)]
            if listing.empty:
                raise SourceUnavailable(f"none of {terminal_codes} are known terminals")
        detail = get("sources", "spark", "terminal_slots", "detail_endpoint")
        frames = []
        for _, row in listing.iterrows():
            uri = detail.format(uuid=row["TerminalUUID"])
            try:
                text = _get(uri, token, accept=accept)
            except SourceUnavailable as exc:
                log.warning("terminal %s unavailable: %s", row["TerminalCode"], exc)
                continue
            frames.append(pd.read_csv(StringIO(text)))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not frame.empty:
            frame["ReleaseDate"] = pd.to_datetime(frame["ReleaseDate"])
        return self._result(frame, self.source)

    @staticmethod
    def terminal_list(token: str) -> pd.DataFrame:
        uri = get("sources", "spark", "terminal_slots", "list_endpoint")
        accept = get("sources", "spark", "terminal_slots", "accept_header")
        return pd.read_csv(StringIO(_get(uri, token, accept=accept)))

    @staticmethod
    def coverage(frame: pd.DataFrame) -> pd.DataFrame:
        """First and last release per terminal, so gaps are stated not discovered."""
        return (frame.groupby("TerminalCode")["ReleaseDate"]
                .agg(["min", "max", "count"])
                .sort_values("min"))


def _default_release_limit() -> int:
    """Enough releases to cover the configured training window comfortably."""
    train = int(get("model", "backtest", "initial_train_business_days"))
    tenors = len(get("target", "tenors", "fitted"))
    return train * tenors


def _maybe_float(value: object) -> float | None:
    return None if value is None else float(value)


def _nested(point: dict, outer: str, inner: str) -> float | None:
    node = (point.get(outer) or {}).get(inner) or {}
    return _maybe_float(node.get("usdPerMMBtu"))


def _meta(point: dict, meta_key: str, field: str) -> float | None:
    node = (point.get(meta_key) or {}).get(field) or {}
    return _maybe_float(node.get("usdPerMMBtu"))
