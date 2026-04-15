# TODO: Chart the series and simulate the orderbook
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import statsmodels.formula.api as smf

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


def get_trades(ticker: str) -> pd.DataFrame:
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
        params = {"ticker": ticker, "limit": 1000}
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


# Pull all market tickers for TICKER base ticker
markets = get_market(TICKER, LIM=250)
print(f"Found {len(markets)} in {TICKER}")

# Pull all trades from markets into a dictionary of ticker: pandas.DataFrame
# for each market with all trades
all_market_trades = {}
for market in markets:
    ticker = market["ticker"]
    print(f"Getting trades for {ticker}")
    market_trade = get_trades(ticker)
    print(f"    Found {len(market_trade)} trades")
    all_market_trades[ticker] = market_trade

# Take the dict into a dataframe with the ticker
full_frame = pd.DataFrame()
for key, value in all_market_trades.items():
    full_frame = pd.concat([full_frame, value])

full_frame = full_frame.reset_index(drop=True)

# Extract the strike word from the ticker
word = full_frame["ticker"].str.extract(r"(?<=-)(\w+)$").squeeze()
full_frame.insert(0, "word", word)

# Extract the full array of words
array_of_words = full_frame["word"].sort_values().unique()

# Set up the dataframe filtered to the press conference
full_frame_event = full_frame[
    (full_frame["created_time"] >= pd.Timestamp("2026-03-18 18:29:00", tz="UTC"))
    & (full_frame["created_time"] <= pd.Timestamp("2026-03-18 19:29:00", tz="UTC"))
].reset_index(drop=True)

# Get unsigned OFI data
ofi_data = ofi(full_frame_event, "1s", abs=False)

event_frame = (
    full_frame_event[["word", "created_time", "no_price_dollars", "yes_price_dollars"]]
    .set_index("created_time")
    .groupby("word")
    .resample("1s")
    .agg(no_price=("no_price_dollars", "last"), yes_price=("yes_price_dollars", "last"))
    .ffill()
    .reset_index("created_time")
)

data_joined = event_frame.merge(
    ofi_data.reset_index("created_time"), on=["word", "created_time"]
)

data_joined["price_lead"] = data_joined.groupby("word")["yes_price"].shift(-1)
data_joined["price_diff"] = data_joined["price_lead"] - data_joined["yes_price"]
data_joined["ofi_thresh"] = data_joined.groupby("word").apply(
    lambda group: 3 * np.sqrt(group[group["ofi"] > 0][["ofi"]].var())
)
data_joined = data_joined.reset_index()

yes_thresh = 0.9
data_window = pd.DataFrame()
for word in data_joined["word"].unique():
    count = 1
    data_join_word = data_joined[data_joined["word"] == word]
    spike_times = data_join_word[
        (data_join_word["ofi"] > data_join_word["ofi_thresh"])
    ]["created_time"].reset_index(drop=True)
    for spike in spike_times:
        data_spike = data_join_word.set_index("created_time").loc[
            spike - pd.Timedelta("20s") : spike + pd.Timedelta("5s")
        ]
        data_spike = data_spike[
            (data_spike["yes_price"] < yes_thresh)
            & (data_spike["no_price"] < yes_thresh)
        ]
        data_spike["window_num"] = count
        count += 1
        data_window = pd.concat([data_window, data_spike])

data_model = data_window.reset_index()

nw_lags = int(np.floor(4 * (30 / 100) ** (2 / 9)))

results = data_model.groupby(["word", "window_num"]).apply(
    lambda group: smf.ols("price_diff ~ ofi", data=group).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
)

for word, result in results.items():
    print(f"======================= Results for {word} =======================\n\n")
    print(result.summary())

summary = pd.DataFrame(
    {
        word: {
            "beta": result.params["ofi"],
            "pvalue": result.pvalues["ofi"],
            "sig": result.params["ofi"]
            / np.where(result.bse["ofi"] != 0, result.bse["ofi"], 100)
            > 2.58,
            "r2": result.rsquared,
        }
        for word, result in results.items()
    }
).T

summary[summary["sig"]]
