# TODO: Chart the series and simulate the orderbook
import pandas as pd
import time
import matplotlib.pyplot as plt
# import numpy as np
import requests
import seaborn as sns

BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"


def summary_stat(df: pd.DataFrame) -> pd.DataFrame:
    pass

def ofi(df: pd.DataFrame) -> int:
    pass


def get_market(series_ticker: str) -> list[dict]:
    """Queries the Kalshi markets API to get all the current market data
    Returns a JSON"""
    url = f"{BASEURL}/markets"
    params = {"series_ticker": TICKER, "limit": 100}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["markets"]


def get_trades(ticker: str) -> pd.DataFrame:
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


markets = get_market(TICKER)
print(f"Found {len(markets)} in {TICKER}")

all_market_trades = {}
for market in markets:
    ticker = market["ticker"]
    print(f"Getting trades for {ticker}")
    market_trade = get_trades(ticker)
    print(f"    Found {len(market_trade)} trades")
    all_market_trades[ticker] = market_trade

full_frame = pd.DataFrame()
for key, value in all_market_trades.items():
    full_frame = pd.concat([full_frame, value])

word = full_frame["ticker"].str.extract(r"(?<=-)(\w+)$").squeeze()
full_frame.insert(0, "word", word)

full_frame["word"].sort_values().unique()

full_frame_filtered = full_frame[full_frame["word"] == "CRED"].copy()
mask = full_frame_filtered.columns.str.contains(r"dollars|fp")
full_frame_filtered[full_frame_filtered.columns[mask]] = full_frame_filtered.loc[
    :, mask
].apply(pd.to_numeric)
sns.lineplot(data=full_frame_filtered, x="created_time", y="yes_price_dollars")

df_volume = (
    full_frame[
        (full_frame["yes_price_dollars"] < 0.8)
        & (full_frame["yes_price_dollars"] > 0.2)
    ]
    .groupby("word")
    .agg(volume=("count_fp", "sum"))
    .sort_values(by="volume", ascending=False)
    .reset_index()
)
top_3_volume = df_volume["word"].head(3).to_list()
filtered_top_3_volume = full_frame[full_frame["word"].isin(top_3_volume)]

fig, ax = plt.subplots()
sns.lineplot(
    data=filtered_top_3_volume, x="created_time", y="yes_price_dollars", hue="word"
)
plt.show()

filtered_top_3_volume.set_index("created_time").resample("5min").last().ffill()

data_agg = (
    full_frame.groupby(["ticker", "word"])
    .agg(count=("trade_id", "count"))
    .sort_values("count", ascending=False)
)

data_plot = full_frame[
        (full_frame["ticker"].str.contains(rf"{TICKER}-26MAR"))
        & (full_frame["created_time"] >= pd.Timestamp("2026-03-18 18:29:00", tz="UTC"))
    ][["created_time", "word", "yes_price_dollars"]].set_index("created_time").groupby("word").resample("1s").mean().reset_index().ffill()

data_plot_filtered = data_plot[data_plot["word"].isin(["PAND", "TRAD", "PROJ"])]
data_plot["diff"] = data_plot.groupby("word")["yes_price_dollars"].diff().dropna()
data_plot["max_diff"] = data_plot.groupby("word")["diff"].transform("max")
data_plot_filtered = data_plot[data_plot["max_diff"].abs() > .2]

fig, ax = plt.subplots()
sns.lineplot(
    data=data_plot_filtered.groupby("word").filter(lambda group: (group["yes_price_dollars"] < .99).all()), # data_plot_filtered[data_plot_filtered["yes_price_dollars"] < .99],
    x="created_time",
    y="yes_price_dollars",
    hue="word",
    legend=True,
    ax=ax
)
plt.show()

full_frame.to_csv("full_frame.csv")