import json
import os
import sys
import time
from io import StringIO

import pandas as pd
import requests
from base64 import b64encode
from urllib.parse import urljoin
import datetime

try:
    from urllib import request, parse
    from urllib.error import HTTPError
except ImportError:
    raise RuntimeError("Python 3 required")


# defining query functions
API_BASE_URL = "https://api.sparkcommodities.com"


__all__ = [
    "retrieve_credentials",
    "do_api_post_query",
    "do_api_get_query",
    "get_access_token",
    "list_contracts",
    "fetch_latest_price_releases",
    "fetch_historical_price_releases",
    "fetch_freight_prices",
    "fetch_ffa_prices",
    "fetch_cargo_prices"
]

def retrieve_credentials(file_path=None):
    """
    Find credentials either by reading the client_credentials file or reading
    environment variables
    """
    if file_path is None:
        client_id = os.getenv("SPARK_CLIENT_ID")
        client_secret = os.getenv("SPARK_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "SPARK_CLIENT_ID and SPARK_CLIENT_SECRET environment vars required"
            )
    else:
        # Parse the file
        if not os.path.isfile(file_path):
            raise RuntimeError("The file {} doesn't exist".format(file_path))

        with open(file_path) as fp:
            lines = [l.replace("\n", "") for l in fp.readlines()]

        if lines[0] in ("clientId,clientSecret", "client_id,client_secret"):
            client_id, client_secret = lines[1].split(",")
        else:
            print("First line read: '{}'".format(lines[0]))
            raise RuntimeError(
                "The specified file {} doesn't look like to be a Spark API client "
                "credentials file".format(file_path)
            )

    print(">>>> Found credentials!")
    print(
        ">>>> Client_id={}****, client_secret={}****".format(
            client_id[:5], client_secret[:5]
        )
    )

    return client_id, client_secret


def do_api_post_query(uri, body, headers):
    """
    OAuth2 authentication requires a POST request with client credentials before accessing the API.
    This POST request will return an Access Token which will be used for the API GET request.
    """
    url = urljoin(API_BASE_URL, uri)

    data = json.dumps(body).encode("utf-8")

    # HTTP POST request
    req = request.Request(url, data=data, headers=headers)
    try:
        response = request.urlopen(req)
    except HTTPError as e:
        print("HTTP Error: ", e.code)
        print(e.read())
        sys.exit(1)

    resp_content = response.read()

    # The server must return HTTP 201. Raise an error if this is not the case
    assert response.status == 201, resp_content

    # The server returned a JSON response
    content = json.loads(resp_content)

    return content


def do_api_get_query(uri, access_token):
    """
    After receiving an Access Token, we can request information from the API.
    """
    url = urljoin(API_BASE_URL, uri)

    headers = {
        "Authorization": "Bearer {}".format(access_token),
        "accept": "application/json",
    }

    # print(f"Fetching {url}")

    # HTTP GET request
    req = request.Request(url, headers=headers)
    try:
        response = request.urlopen(req)
    except HTTPError as e:
        print("HTTP Error: ", e.code)
        print(e.read())
        sys.exit(1)

    resp_content = response.read()

    # The server must return HTTP 201. Raise an error if this is not the case
    assert response.status == 200, resp_content

    # The server returned a JSON response
    content = json.loads(resp_content)

    return content


def get_access_token(client_id, client_secret):
    """
    Get a new access_token. Access tokens are the thing that applications use to make
    API requests. Access tokens must be kept confidential in storage.

    # Procedure:

    Do a POST query with `grantType` and `scopes` in the body. A basic authorization
    HTTP header is required. The "Basic" HTTP authentication scheme is defined in
    RFC 7617, which transmits credentials as `clientId:clientSecret` pairs, encoded
    using base64.
    """

    # Note: for the sake of this example, we choose to use the Python urllib from the
    # standard lib. One should consider using https://requests.readthedocs.io/

    payload = "{}:{}".format(client_id, client_secret).encode()
    headers = {
        "Authorization": b64encode(payload).decode(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {
        "grantType": "clientCredentials",
        "scopes": "read:access,read:prices"
    }

    content = do_api_post_query(uri="/oauth/token/", body=body, headers=headers)

    print(
        ">>>> Successfully fetched an access token {}****, valid {} seconds.".format(
            content["accessToken"][:5], content["expiresIn"]
        )
    )

    return content["accessToken"]


def list_netbacks(access_token):
    """
    Fetch available routes. Return contract ticker symbols

    # Procedure:

    Do a GET query to /v1.0/routes/ with a Bearer token authorization HTTP header.
    """
    content = do_api_get_query(uri="/v1.0/netbacks/reference-data/", access_token=access_token)

    print(">>>> All the routes you can fetch")
    tickers = []
    fobPort_names = []

    availablevia = []

    for contract in content["data"]['staticData']['fobPorts']:
        tickers.append(contract["uuid"])
        fobPort_names.append(contract['name'])

        availablevia.append(contract['availableViaPoints'])

    reldates = content["data"]['staticData']['sparkReleases']

    dicto1 = content["data"]

    return tickers, fobPort_names, availablevia, reldates, dicto1


# Defining function for collecting the list of contracts
def list_contracts(access_token):
    """
    Fetch available contracts. Return contract ticker symbols

    # Procedure:

    Do a GET query to /v1.0/contracts/ with a Bearer token authorization HTTP header.
    """
    content = do_api_get_query(uri="/v1.0/contracts/", access_token=access_token)

    print(">>>> All the contracts you can fetch")
    tickers = []
    for contract in content["data"]:
        print(contract["fullName"])
        tickers.append(contract["id"])

    return tickers


def fetch_latest_price_releases(access_token, ticker):
    """
    For a contract, fetch then display the latest price release

    # Procedure:

    Do GET queries to /v1.0/contracts/{contract_ticker_symbol}/price-releases/latest/
    with a Bearer token authorization HTTP header.
    """
    content = do_api_get_query(
        uri="/v1.0/contracts/{}/price-releases/latest/".format(ticker),
        access_token=access_token,
    )

    release_date = content["data"]["releaseDate"]

    print(">>>> Get latest price release for {}".format(ticker))
    print("release date =", release_date)
    return content["data"]

def fetch_historical_price_releases(access_token, ticker, limit=4, offset=None, vessel=None):

    query_params = "?limit={}".format(limit)
    if offset is not None:
        query_params += "&offset={}".format(offset)

    # '174-2stroke' or '160-tfde'
    if vessel is not None:
        query_params += "&vessel-type={}".format(vessel)

    # print("/v1.0/contracts/{}/price-releases/{}".format(ticker, query_params))

    content = do_api_get_query(
        uri="/v1.0/contracts/{}/price-releases/{}".format(ticker, query_params),
        access_token=access_token,
    )

    my_dict = content['data']

    return my_dict


def get_terminal_list(access_token):
    uri = urljoin(API_BASE_URL,'beta/terminal-slots/terminals/')
    headers = {
            "Authorization": "Bearer {}".format(access_token),
            "accept": "text/csv"
        }
    response = requests.get(uri, headers=headers)
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
    else:
        print('Bad Request')
    return df


# Function to collect and store historical slots for one specific terminal
def get_individual_terminal(access_token, terminal_uuid):
    uri = urljoin(API_BASE_URL, f'/beta/terminal-slots/terminals/{terminal_uuid}/')
    headers = {
            "Authorization": "Bearer {}".format(access_token),
            "accept": "text/csv"
        }
    response = requests.get(uri, headers=headers)
    if response.status_code == 200:
        df = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(df))
        return df

    elif response.content == b'{"errors":[{"code":"object_not_found","detail":"Object not found"}]}':
        print('Bad Terminal Request')
        return None
    else:
        print('Bad Request')
        return None


# Function to collect and store each terminal's historical slots data
def get_all_terminal_data(access_token, terminal_list):
    terminals_all = pd.DataFrame()
    for i in range(len(terminal_list)):
        terminal_df = get_individual_terminal(access_token, terminal_list['TerminalUUID'].loc[i])
        # time.sleep(0.1)
        terminals_all = pd.concat([terminals_all,terminal_df])
    return terminals_all


def fetch_freight_prices(access_token, ticker, my_lim, my_vessel=None, latest_only=False, cal_month=None):
    if not latest_only:
        my_dict_hist = fetch_historical_price_releases(access_token, ticker, limit=my_lim, vessel=my_vessel)
    else:
        my_dict_hist = fetch_latest_price_releases(access_token, ticker)
        my_dict_hist = [my_dict_hist]

    release_dates = []
    period_start = []
    ticker = []
    usd_day = []

    day_min = []
    day_max = []

    for release in my_dict_hist:
            release_date = release["releaseDate"]

            data_points = release["data"][0]["dataPoints"]
            for data_point in data_points:
                period_start_at = data_point["deliveryPeriod"]["startAt"]
                calendar_month = datetime.datetime.strptime(period_start_at, '%Y-%m-%d').strftime('%b-%Y')
                if cal_month is None:
                    ticker.append(release['contractId'])
                    release_dates.append(release_date)

                    period_start_at = data_point["deliveryPeriod"]["startAt"]
                    period_start.append(period_start_at)

                    usd_day.append(data_point['derivedPrices']['usdPerDay']['spark'])
                    day_min.append(data_point['derivedPrices']['usdPerDay']['sparkMin'])
                    day_max.append(data_point['derivedPrices']['usdPerDay']['sparkMax'])
                else:
                    if cal_month == calendar_month:
                        ticker.append(release['contractId'])
                        release_dates.append(release_date)

                        period_start_at = data_point["deliveryPeriod"]["startAt"]
                        period_start.append(period_start_at)

                        usd_day.append(data_point['derivedPrices']['usdPerDay']['spark'])
                        day_min.append(data_point['derivedPrices']['usdPerDay']['sparkMin'])
                        day_max.append(data_point['derivedPrices']['usdPerDay']['sparkMax'])


    historical_df = pd.DataFrame({
        'Release Date': release_dates,
        'ticker': ticker,
        'Period Start': period_start,
        'USDperday': usd_day,
        'USDperdayMax': day_max,
        'USDperdayMin': day_min})

    historical_df['USDperday'] = pd.to_numeric(historical_df['USDperday'])
    historical_df['USDperdayMax'] = pd.to_numeric(historical_df['USDperdayMax'])
    historical_df['USDperdayMin'] = pd.to_numeric(historical_df['USDperdayMin'])

    historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'])

    return historical_df


# Defining the function
def fetch_ffa_prices(access_token, my_tick, my_lim, latest_only=False):
    print(my_tick)

    if not latest_only:
        my_dict_hist = fetch_historical_price_releases(access_token, my_tick, limit=my_lim)
    else:
        my_dict_hist = fetch_latest_price_releases(access_token, my_tick)
        my_dict_hist = [my_dict_hist]

    release_dates = []

    period_start = []
    period_end = []
    period_name = []
    cal_month = []

    ticker = []

    usd_day = []

    day_min = []
    day_max = []

    for release in my_dict_hist:
        release_date = release["releaseDate"]

        print("- release date =", release_date)

        data = release["data"]

        for d in data:
            data_points = d["dataPoints"]
            for data_point in data_points:
                period_start_at = data_point["deliveryPeriod"]["startAt"]
                period_start.append(period_start_at)
                period_end_at = data_point["deliveryPeriod"]["endAt"]
                period_end.append(period_end_at)
                period_name.append(data_point["deliveryPeriod"]["name"])

                release_dates.append(release_date)
                # release_dates.append(datetime.strptime(release_date, '%Y-%m-%d'))
                ticker.append(release["contractId"])
                cal_month.append(
                    datetime.datetime.strptime(period_start_at, "%Y-%m-%d").strftime("%b-%Y")
                )

                usd_day.append(int(data_point["derivedPrices"]["usdPerDay"]["spark"]))
                day_min.append(
                    int(data_point["derivedPrices"]["usdPerDay"]["sparkMin"])
                )
                day_max.append(
                    int(data_point["derivedPrices"]["usdPerDay"]["sparkMax"])
                )

    historical_df = pd.DataFrame(
        {
            "Release Date": release_dates,
            "ticker": ticker,
            "Period Name": period_name,
            "Period Start": period_start,
            "Period End": period_end,
            "Calendar Month": cal_month,
            "Spark": usd_day,
            "SparkMin": day_min,
            "SparkMax": day_max,
        }
    )

    historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'],format='%Y-%m-%d')

    return historical_df


def fetch_ffa_prices_for_month_only(access_token, my_tick, my_lim, month_tenor):
    print(my_tick)

    my_dict_hist = fetch_historical_price_releases(access_token, my_tick, limit=my_lim);

    release_dates = []
    period_name = []
    cal_month = []

    ticker = []

    usd_day = []

    day_min = []
    day_max = []

    for release in my_dict_hist:
        release_date = release["releaseDate"]

        # print("- release date =", release_date)

        data = release["data"]

        for d in data:
            data_points = d["dataPoints"]
            for data_point in data_points:
                period_start_at = data_point["deliveryPeriod"]["startAt"]
                cal_mont_str = datetime.datetime.strptime(period_start_at, "%Y-%m-%d").strftime("%b-%Y")
                if cal_mont_str == month_tenor:
                    period_start_at = data_point["deliveryPeriod"]["startAt"]
                    # period_start.append(period_start_at)
                    period_end_at = data_point["deliveryPeriod"]["endAt"]
                    # period_end.append(period_end_at)
                    period_name.append(data_point["deliveryPeriod"]["name"])

                    release_dates.append(release_date)
                    # release_dates.append(datetime.strptime(release_date, '%Y-%m-%d'))
                    ticker.append(release["contractId"])
                    cal_month.append(
                        cal_mont_str
                    )

                    usd_day.append(int(data_point["derivedPrices"]["usdPerDay"]["spark"]))
                    day_min.append(
                        int(data_point["derivedPrices"]["usdPerDay"]["sparkMin"])
                    )
                    day_max.append(
                        int(data_point["derivedPrices"]["usdPerDay"]["sparkMax"])
                    )

    historical_df = pd.DataFrame(
        {
            "Release Date": release_dates,
            "ticker": ticker,
            # "Period Name": period_name,
            # "Period Start": period_start,
            # "Period End": period_end,
            "Calendar Month": cal_month,
            "Spark": usd_day,
            "SparkMin": day_min,
            "SparkMax": day_max,
        }
    )

    historical_df['Release Date'] = pd.to_datetime(historical_df['Release Date'],format='%Y-%m-%d')

    return historical_df


def fetch_cargo_prices(access_token, ticker, limit, latest_only=False, cal_month=None):

    # imports front month or forward curve prices, depending on the "month" user input
    release_dates = []
    period_start = []
    tickers = []
    spark = []

    spark_min = []
    spark_max = []
    cal_months = []

    small_tickers = ["-b-f", "-b-fo"] if "sparknwe-fin-monthly" not in ticker else [""]

    for small_ticker in small_tickers:
        full_tick = ticker + small_ticker

        if not latest_only:
            hist_data = fetch_historical_price_releases(access_token, full_tick, limit=limit)
        else:
            hist_data = fetch_latest_price_releases(access_token, full_tick)
            hist_data = [hist_data]

        # iterating through historical data points to fetch relevant data
        for release in hist_data:
                release_date = release["releaseDate"]
                data_points = release['data'][0]['dataPoints']
                for data_point in data_points:
                    period_start_at = data_point["deliveryPeriod"]["startAt"]
                    calendar_month = datetime.datetime.strptime(period_start_at, '%Y-%m-%d').strftime('%b-%Y')
                    if cal_month is None:  # Add them all
                        period_start.append(period_start_at)
                        tickers.append(release['contractId'])
                        release_dates.append(release_date)

                        spark.append(data_point['derivedPrices']['usdPerMMBtu']['spark'])
                        spark_min.append(data_point['derivedPrices']['usdPerMMBtu']['sparkMin'])
                        spark_max.append(data_point['derivedPrices']['usdPerMMBtu']['sparkMax'])

                        cal_months.append(calendar_month)
                    else:  # Add only calendar month
                        if cal_month == calendar_month:
                            period_start.append(period_start_at)
                            tickers.append(release['contractId'])
                            release_dates.append(release_date)

                            spark.append(data_point['derivedPrices']['usdPerMMBtu']['spark'])
                            spark_min.append(data_point['derivedPrices']['usdPerMMBtu']['sparkMin'])
                            spark_max.append(data_point['derivedPrices']['usdPerMMBtu']['sparkMax'])

                            cal_months.append(calendar_month)

    # Converting into DataFrame
    hist_df = pd.DataFrame({
        'Release Date': release_dates,
        'ticker': tickers,
        'Period Start': period_start,
        'Price': spark,
        })


    hist_df['Price'] = pd.to_numeric(hist_df['Price'])
    hist_df['Release Date'] = pd.to_datetime(hist_df['Release Date'])

    hist_df['Release Date'] = hist_df['Release Date'].dt.tz_localize(None)

    return hist_df


if __name__ == "__main__":
    file_pathh = "C:\\Users\\marti.fernandezreal\\OneDrive - BAYEGAN DIS TIC. A.S\\Marti\\python_tests\\BGN_LNG\\adhoc_scripts\\client_credentials.csv"
    client_id, client_secret = retrieve_credentials(file_path=file_pathh)
    access_token = get_access_token(client_id, client_secret)

    # hist_df = fetch_ffa_prices_for_month_only(access_token, "spark30ffa-monthly", 30 * 30, month_tenor="Feb-2026")
    a = 1

    print(list_contracts(access_token))


