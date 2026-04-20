import pandas as pd
import time
import numpy as np
import requests

BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"


def summary_stat(df: pd.DataFrame) -> pd.DataFrame:
    pass


def ofi(df: pd.DataFrame, period: str, abs: bool = False) -> pd.DataFrame:
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
    if "word" not in df.columns:
        raise ValueError(
            f"Column 'word' not found in {df.columns}, cannot group the data"
        )

    grouped_df = (
        df.set_index("created_time").groupby(["word", "taker_side"]).resample(period)
    )
    unsigned_agg = grouped_df.agg(volume=("count_fp", "sum")).reset_index()

    unsigned_agg["signed_volume"] = np.where(
        unsigned_agg["taker_side"] == "yes",
        unsigned_agg["volume"],
        -unsigned_agg["volume"],
    )

    agg_df = unsigned_agg.groupby(["created_time", "word"]).agg(
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


def get_trades(ticker: str, depth: int = 0) -> pd.DataFrame:
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
