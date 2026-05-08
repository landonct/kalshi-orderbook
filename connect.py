# TODO: Chart the series and simulate the orderbook
from requests import HTTPError
from pandas.io.formats import printing
import pandas as pd
import datetime
import time
import re
from dateutil import tz
import matplotlib.pyplot as plt
import numpy as np
from zoneinfo import ZoneInfo
import seaborn as sns
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
TZ = tz.gettz("America/New_York")
PRESS_CONF_START = datetime.time(14, 30, 00, tzinfo=TZ)
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

    try:
        market_trade = functions.get_trades(ticker)
    except HTTPError:
        print("an error occured")

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

    try:
        market_trade = functions.get_historical_trades(
            ticker, int(datetime.datetime.now().timestamp()), BASEURL
        )
    except HTTPError:
        print("an error occured")

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

press_conf_df = press_conf_df.reset_index(drop=True)
resolved_yes = press_conf_df.groupby("ticker").filter(
    lambda x: (x.sort_values("created_time")["yes_price_dollars"].iloc[0] < 0.9)
    & (x["yes_price_dollars"] >= 0.99).any()
)
ticker_uniq = resolved_yes["ticker"].unique()
print(f"There are {len(ticker_uniq)} tickers that resolved to yes")

sns.lineplot(
    data=resolved_yes[
        resolved_yes["created_time"] < pd.to_datetime("2025-05", utc=True)
    ],
    x="created_time",
    y="yes_price_dollars",
    hue="word",
    legend=True,
)

# Get unsigned OFI data
ofi_data = functions.ofi(resolved_yes, "1s", abs=True)

event_frame = functions.make_event_frame(resolved_yes)

data_joined = event_frame.merge(ofi_data, on=["created_time", "ticker"]).reset_index()

data_joined["price_lag"] = data_joined.groupby("ticker")["yes_price"].shift(1)
data_joined["price_diff"] = data_joined["yes_price"] - data_joined["price_lag"]
# data_joined["ofi_thresh"] = data_joined.groupby("ticker").apply(
#     lambda group: 3 * np.sqrt(group[group["ofi"] > 0][["ofi"]].var())
# )

data_model = data_joined

nw_lags = int(np.floor(4 * (100000 / 100) ** (2 / 9)))

results = {
    ticker: smf.ols("price_diff ~ ofi", data=group).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lags}
    )
    for ticker, group in data_model.groupby("ticker")
    if len(group) > nw_lags + 1
}

for word, result in results.items():
    print(f"\n======================= Results for {word} =======================\n")
    print(result.summary())
    time.sleep(1)

summary = pd.DataFrame(
    {
        word: {
            "beta": result.params["ofi"],
            "pvalue": result.pvalues["ofi"],
            "sig": (
                abs(result.params["ofi"] / result.bse["ofi"]) > 2.58
                if result.bse["ofi"] != 0
                else np.NaN
            ),
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
