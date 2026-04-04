# TODO: Chart the series and simulate the orderbook
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf
import math

BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"


def summary_stat(df: pd.DataFrame) -> pd.DataFrame:
    pass


def ofi(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if "word" not in df.columns:
        raise ValueError(f"Column 'word' not found in {df.columns}")

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

    return agg_df


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

data_plot = (
    full_frame[
        (full_frame["ticker"].str.contains(rf"{TICKER}-26MAR"))
        & (full_frame["created_time"] >= pd.Timestamp("2026-03-18 18:29:00", tz="UTC"))
    ][["created_time", "word", "yes_price_dollars"]]
    .set_index("created_time")
    .groupby("word")
    .resample("1s")
    .mean()
    .reset_index()
    .ffill()
)

data_plot_filtered = data_plot[data_plot["word"].isin(["PAND", "TRAD", "PROJ"])]
data_plot["diff"] = data_plot.groupby("word")["yes_price_dollars"].diff().dropna()
data_plot["max_diff"] = data_plot.groupby("word")["diff"].transform("max")
data_plot_filtered = data_plot[data_plot["max_diff"].abs() > 0.2]

fig, ax = plt.subplots()
sns.lineplot(
    data=data_plot_filtered.groupby("word").filter(
        lambda group: (group["yes_price_dollars"] < 0.99).all()
    ),  # data_plot_filtered[data_plot_filtered["yes_price_dollars"] < .99],
    x="created_time",
    y="yes_price_dollars",
    hue="word",
    legend=True,
    ax=ax,
)
plt.show()

full_frame.to_csv("full_frame.csv")

full_frame_event = full_frame[
    (full_frame["created_time"] >= pd.Timestamp("2026-03-18 18:29:00", tz="UTC"))
    & (full_frame["created_time"] <= pd.Timestamp("2026-03-18 19:29:00", tz="UTC"))
]
ofi_data = ofi(full_frame_event, "1s")

ofi_plot = ofi_data.reset_index().set_index("created_time").resample("1s")

fig, ax = plt.subplots()
sns.lineplot(data=ofi_data, x="created_time", y="ofi", hue="word", legend=False, ax=ax)
ax.set_ylim(bottom=-2000, top=2000)
plt.show()

event_frame = (
    full_frame_event[["word", "created_time", "no_price_dollars", "yes_price_dollars"]]
    .set_index("created_time")
    .groupby("word")
    .resample("10s")
    .agg(no_price=("no_price_dollars", "last"), yes_price=("yes_price_dollars", "last"))
    .ffill()
    .reset_index("created_time")
)

data_joined = event_frame.merge(
    ofi_data.reset_index("created_time"), on=["word", "created_time"]
)

data_joined["price_lead"] = data_joined.groupby("word")["yes_price"].shift(-1)
data_joined["price_diff"] = data_joined["price_lead"] - data_joined["yes_price"]

nw_lags = math.floor(4 * (3450 / 100) ** (2 / 9))

results = data_joined.groupby("word").apply(
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

summary
