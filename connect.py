# TODO: Chart the series and simulate the orderbook
import pandas as pd

# import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import statsmodels.formula.api as smf
import functions

BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"

# Pull all market tickers for TICKER base ticker
markets = functions.get_market(TICKER, LIM=250)
print(f"Found {len(markets)} in {TICKER}")

# Pull all trades from markets into a dictionary of ticker: pandas.DataFrame
# for each market with all trades
all_market_trades = {}
for market in markets:
    ticker = market["ticker"]
    print(f"Getting trades for {ticker}")
    market_trade = functions.get_trades(ticker)
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
ofi_data = functions.ofi(full_frame_event, "1s", abs=False)

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
