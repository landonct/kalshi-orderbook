import functions
import pandas as pd
import numpy as np
import requests
import time
import sqlite3
from dotenv import load_dotenv
import os

load_dotenv(".env")
KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
PRIVATE_KEY = functions.load_private_key(PRIVATE_KEY_PATH)
BASEURL = "https://api.elections.kalshi.com/trade-api/v2"
DB = "kalshi_sports.db"

con = sqlite3.connect(DB)
con.executescript(
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id      TEXT PRIMARY KEY,
        market_ticker TEXT,
        series_ticker TEXT,
        yes_price     REAL,
        no_price      REAL,
        count         REAL,
        taker_side    TEXT,
        created_time  TEXT
    );
    CREATE TABLE IF NOT EXISTS progress (
        series_ticker TEXT PRIMARY KEY,
        status        TEXT,
        markets_found INTEGER,
        trades_found  INTEGER,
        finished_at   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_ticker);
    CREATE INDEX IF NOT EXISTS idx_trades_series ON trades(series_ticker);
"""
)
con.commit()


def get_all_market_tickers(series_ticker):
    tickers = []
    cursor = None
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASEURL}/markets", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        batch = data.get("markets", [])
        tickers.extend(m["ticker"] for m in batch)
        cursor = data.get("cursor")
        if not cursor or len(batch) < 1000:
            break
        time.sleep(0.1)
    return tickers


def get_sport_series():
    resp = requests.get(f"{BASEURL}/series", params={"category": "Sports"})
    resp.raise_for_status()
    return resp.json()["series"]


def stream_trades(con, market_ticker, series_ticker):
    cursor = None
    total = 0
    while True:
        params = {"ticker": market_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{BASEURL}/markets/trades", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        trades = data.get("trades", [])
        if trades:
            con.executemany(
                "INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        t.get("trade_id"),
                        market_ticker,
                        series_ticker,
                        (
                            float(t["yes_price_dollars"])
                            if t.get("yes_price_dollars")
                            else None
                        ),
                        (
                            float(t["no_price_dollars"])
                            if t.get("no_price_dollars")
                            else None
                        ),
                        float(t["count_fp"]) if t.get("count_fp") else None,
                        t.get("taker_side"),
                        t.get("created_time"),
                    )
                    for t in trades
                ],
            )
            con.commit()
            total += len(trades)
        cursor = data.get("cursor")
        if not cursor or len(trades) < 1000:
            break
        time.sleep(0.1)
    return total


# ── Main loop ─────────────────────────────────────────────────────────────────

sports_series = get_sport_series()
sports_tickers = [s["ticker"] for s in sports_series]

# Skip series already completed — safe to restart if it crashes
done = {
    row[0]
    for row in con.execute("SELECT series_ticker FROM progress WHERE status='done'")
}
remaining = [t for t in sports_tickers if t not in done]
print(
    f"Found {len(sports_tickers)} series | {len(done)} already done | {len(remaining)} remaining"
)

for series_ticker in remaining:
    try:
        print(
            f"\n[{sports_tickers.index(series_ticker)+1}/{len(sports_tickers)}] {series_ticker}"
        )
        market_tickers = get_all_market_tickers(series_ticker)
        print(f"  {len(market_tickers)} markets")

        series_trades = 0
        for i, market_ticker in enumerate(market_tickers):
            try:
                n = stream_trades(con, market_ticker, series_ticker)
                series_trades += n
                if n:
                    print(
                        f"  [{i+1}/{len(market_tickers)}] {market_ticker}: {n} trades"
                    )
                time.sleep(0.1)
            except Exception as e:
                print(f"  SKIPPED {market_ticker}: {e}")
                time.sleep(1)
                continue

        # Mark series as done so a restart skips it
        con.execute(
            "INSERT OR REPLACE INTO progress VALUES (?,?,?,?,datetime('now'))",
            (series_ticker, "done", len(market_tickers), series_trades),
        )
        con.commit()

    except Exception as e:
        print(f"  FAILED {series_ticker}: {e}")
        time.sleep(2)
        continue

con.close()
print("\nAll done.")
