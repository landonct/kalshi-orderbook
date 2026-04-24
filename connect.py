# TODO: Chart the series and simulate the orderbook
import pandas as pd
import datetime
import re

# import matplotlib.pyplot as plt
import numpy as np
from zoneinfo import ZoneInfo

# import seaborn as sns
import statsmodels.formula.api as smf
import functions
import os
from dotenv import load_dotenv

load_dotenv(".env")
KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
PRIVATE_KEY = functions.load_private_key(PRIVATE_KEY_PATH)
BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
TICKER = "KXFEDMENTION"

FOMC_CAL_STR = "<ul><li>/monetarypolicy/fomcpresconf20260318.htm</li><li>/monetarypolicy/fomcpressconf20260128.htm</li><li>/monetarypolicy/fomcpresconf20251210.htm</li><li>/monetarypolicy/fomcpresconf20251029.htm</li><li>/monetarypolicy/fomcpresconf20250917.htm</li><li>/monetarypolicy/fomcpresconf20250730.htm</li><li>/monetarypolicy/fomcpresconf20250618.htm</li><li>/monetarypolicy/fomcpresconf20250507.htm</li><li>/monetarypolicy/fomcpresconf20250319.htm</li>"
FOMC_CAL_STR = re.sub(r"<.*?>", "", FOMC_CAL_STR)
FOMC_CAL_STR = re.sub(r"/monetarypolicy/fomcpres{1,2}conf", " ", FOMC_CAL_STR)
FOMC_CAL = re.sub(r"\.htm", "", FOMC_CAL_STR).strip().split(" ")
PRESS_CONF_START = datetime.time(14, 30, 00)
FOMC_START = [
    datetime.datetime.combine(pd.to_datetime(date, format="%Y%m%d"), PRESS_CONF_START)
    for date in FOMC_CAL
]
FOMC_END = [
    (date + pd.Timedelta(minutes=90)).replace(tzinfo=ZoneInfo("America/New_York"))
    for date in FOMC_START
]

# Pull all market tickers for TICKER base ticker
markets = functions.get_market(TICKER, LIM=250)
print(f"Found {len(markets)} in {TICKER}")

# Pull all trades from markets into a dictionary of ticker: pandas.DataFrame
# for each market with all trades
all_tickers = pd.Series()
all_market_trades = {}
for market in markets:
    ticker = market["ticker"]
    all_tickers = pd.concat([all_tickers, pd.Series(ticker)])
    print(f"Getting trades for {ticker}")
    market_trade = functions.get_trades(ticker)
    print(f"    Found {len(market_trade)} trades")
    all_market_trades[ticker] = market_trade

markets_hist = functions.get_historical_market(TICKER, LIM=1000)
print(f"Found {len(markets_hist)} in {TICKER}")

all_tickers_hist = pd.Series()
all_market_trades_hist = {}
for market in markets_hist:
    ticker = market["ticker"]
    all_tickers = pd.concat([all_tickers_hist, pd.Series(ticker)])
    print(f"Getting trades for {ticker}")
    market_trade = functions.get_historical_trades(
        ticker, int(datetime.datetime.now().timestamp()), BASEURL
    )
    print(f"    Found {len(market_trade)} trades")
    all_market_trades_hist[ticker] = market_trade

# Take the dict into a dataframe with the ticker
full_frame = pd.DataFrame()
for key, value in all_market_trades.items():
    full_frame = pd.concat([full_frame, value])

full_frame_hist = pd.DataFrame()
for key, value in all_market_trades_hist.items():
    full_frame_hist = pd.concat([full_frame_hist, value])

full_frame = pd.concat([full_frame, full_frame_hist])

full_frame = full_frame.reset_index(drop=True)
# full_frame["created_time"] = full_frame["created_time"].dt.tz_convert(
#     "America/New_York"
# )

mask = pd.Series(False, index=full_frame.index)
for start, end in zip(FOMC_START, FOMC_END):
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)
    mask |= (full_frame["created_time"] >= start) & (full_frame["created_time"] <= end)

press_conf_df = full_frame[mask].copy()
press_conf_df["created_time"] = pd.to_datetime(press_conf_df["created_time"])

functions.extract_word(press_conf_df)

# Extract the full array of words
array_of_words = press_conf_df["word"].sort_values().unique()

press_conf_df = press_conf_df.reset_index()
resolved_yes = press_conf_df.groupby("ticker").filter(
    lambda x: (x.sort_values("created_time")["yes_price_dollars"].iloc[0] < 0.9)
    & (x["yes_price_dollars"] >= 0.99).any()
)

# Get unsigned OFI data
ofi_data = functions.ofi(press_conf_df, "1s", abs=False)

event_frame = functions.make_event_frame(press_conf_df)

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

all_tickers_filter = all_tickers[all_tickers.str.contains(r"-26APR-")]
functions.get_all_orderbooks(all_tickers_filter, PRIVATE_KEY, KALSHI_ACCESS_KEY)

press_conf_df.groupby("word").apply(
    lambda group: [
        group.loc[0]["yes_price_dollars"],
        group.loc[-1]["yes_price_dollars"],
    ]
)

press_conf_df.loc[0]
