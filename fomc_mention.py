import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_parquet("full_frame.parquet")

data["meeting"] = data["ticker"].str.split("-").str[1]
data[["year", "month"]] = data["meeting"].str.extract(r"(\d{2})(\w{3})")

df = data[(data["year"] >= "25") & (data["month"] != "SEP")].reset_index(drop=True)
df["price_paid"] = np.where(
    df["taker_outcome_side"] == "yes", df["yes_price_dollars"], df["no_price_dollars"]
)
df = df[
    [
        "meeting",
        "created_time",
        "ticker",
        "count_fp",
        "price_paid",
        "taker_outcome_side",
    ]
]
df_agg = (
    df.assign(transaction=df["count_fp"] * df["price_paid"])
    .groupby(["meeting", "created_time", "ticker"])
    .agg(
        volume=("count_fp", "sum"),
        transaction=("transaction", "sum"),
        avg_price=("price_paid", "mean"),
    )
    .sort_values("transaction")
)
df2 = df_agg.reset_index()

whale = 1000
df2[df2["transaction"] >= whale]
df3 = (
    df2.groupby("meeting")
    .agg(
        volume=("volume", lambda x: sum(x) / 1000000),
        transaction=("transaction", lambda x: sum(x) / 1000000),
    )
    .reset_index()
)

df3["meeting_dt"] = pd.to_datetime(df3["meeting"], format="%y%b")

df4 = df3.set_index("meeting_dt").sort_index()

sns.barplot(data=df4, x=df4.index.strftime("%B %Y"), y=df4["volume"])

df2["whale"] = df2["transaction"] > whale
df5 = df2.groupby(["meeting"]).agg(whale_count=("whale", "sum")).reset_index()
df5["meeting"] = pd.to_datetime(df5["meeting"], format="%y%b")
df6 = df5.sort_values("meeting")

sns.barplot(data=df6, x=df6["meeting"].dt.strftime("%B %Y"), y=df6["whale_count"]).set(
    title="# of whales"
)
