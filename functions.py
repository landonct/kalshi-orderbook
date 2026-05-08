from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
import pandas as pd
import time
import numpy as np
import requests
import base64
import datetime
from urllib.parse import urlparse
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding


BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"


def load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )


def sign_request(private_key, timestamp, method, path):
    # Strip query parameters from path before signing
    path_without_query = path.split("?")[0]

    # Create the message to sign
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")

    # Sign with RSA-PSS
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )

    # Return base64 encoded
    return base64.b64encode(signature).decode("utf-8")


def create_signature(private_key, timestamp, method, path):
    """Create the request signature."""
    # Strip query parameters before signing
    path_without_query = path.split("?")[0]
    message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def get(private_key, api_key_id, path, params=None, base_url=BASEURL):
    """Make an authenticated GET request to the Kalshi API."""
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    # Signing requires the full URL path from root (e.g. /trade-api/v2/portfolio/balance)
    sign_path = urlparse(base_url + path).path
    signature = create_signature(private_key, timestamp, "GET", sign_path)

    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

    return requests.get(base_url + path, headers=headers, params=params)


def get_all_orderbooks(
    tickers: list,
    PRIVATE_KEY: RSAPrivateKey,
    KALSHI_ACCESS_KEY: str,
    base_url="/markets/orderbooks",
):
    if len(tickers) > 100:
        raise AttributeError("Too many tickers, maximum 100")
    response = get(
        private_key=PRIVATE_KEY,
        api_key_id=KALSHI_ACCESS_KEY,
        path=base_url,
        params={"tickers": tickers},
    )

    return response.json()


def summary_stat(df: pd.DataFrame) -> pd.DataFrame:
    pass


def ofi(df: pd.DataFrame, period: str = None, abs: bool = False) -> pd.DataFrame:
    """This function takes the Kalshi market data and
    adds a column of signed (if abs=False) order flow imbalance
    for each word

    Args:
        df (pd.DataFrame): pandas.DataFrame containing a column 'word', for grouping
        and a column with 'count_fp' and a 'taker_side' column

        period (str): Valid period to resample the data to compute the OFI across that
        period of time

        abs (bool): Return signed or unsigned OFI

    Raises:
        ValueError: If 'word' column is not found, raise a value error

    Returns:
        pd.DataFrame: Returns a resampled pandas.DataFrame at frequency 'period' with
        signed, if abs=False
    """
    if "ticker" not in df.columns:
        raise ValueError(
            f"Columns 'ticker' and 'taker_side' not found in {df.columns}, cannot group the data"
        )

    grouped_df = df.set_index("created_time").groupby(["ticker", "taker_side"]).resample(period) if period is not None else df.groupby(["ticker", "taker_side"])
    unsigned_agg = grouped_df.agg(volume=("count_fp", "sum")).reset_index()

    unsigned_agg["signed_volume"] = np.where(
        unsigned_agg["taker_side"] == "yes",
        unsigned_agg["volume"],
        -unsigned_agg["volume"],
    )

    agg_df = unsigned_agg.groupby(["created_time", "ticker"]).agg(
        ofi=("signed_volume", "sum")
    )

    if abs:
        agg_df["ofi"] = agg_df["ofi"].abs()

    return agg_df


def get_market(series_ticker: str, LIM: int = 100) -> list[dict]:
    """Pulls all the market tickers from the Kalshi API

    Args:
        series_ticker (str): Queries the Kalshi API using ticker
        series_ticker
        LIM (int, optional): Number of markets to pull. Defaults to 100.

    Returns:
        list[dict]: List of the market names pulled from the API
    """
    url = f"{BASEURL}/markets"
    params = {"series_ticker": series_ticker, "limit": LIM}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["markets"]


def get_trades(ticker: str, BASEURL: str = BASEURL, depth: int = 0) -> pd.DataFrame:
    """Pull orderbook trades for each ticker, typically tickers are from get_markets

    Args:
        ticker (str): Market ticker from Kalshi API

    Returns:
        pd.DataFrame: Contains each trade, along with other data about the contracts
    """
    url = f"{BASEURL}/markets/trades"
    all_trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000, "depth": depth}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        trades = data.get("trades", [])
        if not trades:
            print("No trades")
            break

        all_trades.extend(trades)
        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["created_time"] = pd.to_datetime(df["created_time"])
    df = df.sort_values("created_time").reset_index(drop=True)
    mask = df.columns.str.contains(r"dollars|fp")
    df[df.columns[mask]] = df.loc[:, mask].apply(pd.to_numeric)
    return df


def get_orderbook(ticker: str, depth: int = 0) -> pd.DataFrame:
    """Pull orderbook for each ticker, typically tickers are from get_markets

    Args:
        ticker (str): Market ticker from Kalshi API

    Returns:
        pd.DataFrame: Contains each trade, along with other data about the contracts
    """
    url = f"{BASEURL}/markets/{ticker}/orderbook"
    all_trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000, "depth": depth}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        trades = data.get("trades", [])
        if not trades:
            print("No trades")
            break

        all_trades.extend(trades)
        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["created_time"] = pd.to_datetime(df["created_time"])
    df = df.sort_values("created_time").reset_index(drop=True)
    mask = df.columns.str.contains(r"dollars|fp")
    df[df.columns[mask]] = df.loc[:, mask].apply(pd.to_numeric)
    return df


def get_historical_market(series_ticker: str, LIM: int = 100) -> list[dict]:
    """Pulls all the market tickers from the Kalshi API

    Args:
        series_ticker (str): Queries the Kalshi API using ticker
        series_ticker
        LIM (int, optional): Number of markets to pull. Defaults to 100.

    Returns:
        list[dict]: List of the market names pulled from the API
    """
    url = f"{BASEURL}/historical/markets"
    params = {"series_ticker": series_ticker, "limit": LIM}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["markets"]


def get_historical_trades(
    ticker: str, max_ts: int, BASEURL: str = BASEURL, limit: int = 1000
):
    """Pull orderbook trades for each ticker, typically tickers are from get_markets

    Args:
        ticker (str): Market ticker from Kalshi API

    Returns:
        pd.DataFrame: Contains each trade, along with other data about the contracts
    """
    url = f"{BASEURL}/historical/trades"
    all_trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": limit, "max_ts": max_ts}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        trades = data.get("trades", [])
        if not trades:
            print("No trades")
            break

        all_trades.extend(trades)
        cursor = data.get("cursor")
        if not cursor:
            break

        time.sleep(0.1)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["created_time"] = pd.to_datetime(df["created_time"])
    df = df.sort_values("created_time").reset_index(drop=True)
    mask = df.columns.str.contains(r"dollars|fp")
    df[df.columns[mask]] = df.loc[:, mask].apply(pd.to_numeric)
    return df


def extract_word(df):
    # Extract the strike word from the ticker
    word = df["ticker"].str.extract(r"(?<=-)(\w+)$").squeeze()
    df.insert(0, "word", word)


def make_event_frame(df: pd.DataFrame, resample_pd: str = None):
    out = (
        df[["ticker", "created_time", "no_price_dollars", "yes_price_dollars"]]
        .set_index("created_time")
        .groupby(["ticker"])
        .resample(resample_pd)
        .agg(
            no_price=("no_price_dollars", "last"),
            yes_price=("yes_price_dollars", "last"),
        )
        .ffill()
        .swaplevel()
        .sort_index()
    )

    return out

def onebx(x):
    return (1 / x) if x != 0 else (np.inf)