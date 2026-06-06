import sqlite3
import pandas as pd
import duckdb

con_read = sqlite3.connect("kalshi_sports.db", timeout=30)
pd.read_sql("SELECT COUNT(*) as total_trades FROM trades", con_read)

con = duckdb.connect()
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute("ATTACH 'kalshi_sports.db' AS k (TYPE sqlite)")

con.execute(
    """
SELECT COUNT(*) as total_trades
FROM k.trades
WHERE created_time::TIMESTAMP >= '2026-05-01'
"""
).df()

con.execute(
    """
    COPY (SELECT * FROM k.trades)
    TO 'trades.parquet'
    (FORMAT PARQUET, COMPRESSION ZSTD)
"""
)

counts = con.execute(
    """
    SELECT
        CASE
            WHEN series_ticker LIKE 'KXNBA%' THEN 'NBA'
            WHEN series_ticker LIKE 'KXMLB%' THEN 'MLB'
            WHEN series_ticker LIKE 'KXNFL%' THEN 'NFL'
            WHEN series_ticker LIKE 'KXUFC%' THEN 'UFC'
        END AS sport,
        COUNT(*) as n_trades
    FROM read_parquet('trades.parquet')
    WHERE created_time::TIMESTAMP >= (CURRENT_TIMESTAMP - INTERVAL '12 months')
    AND (
    series_ticker LIKE 'KXNBA%'
       OR series_ticker LIKE 'KXMLB%'
       OR series_ticker LIKE 'KXNFL%'
       OR series_ticker LIKE 'KXUFC%'
       )
    GROUP BY sport
    ORDER BY n_trades DESC
"""
).df()

data = con.execute(
    """
    SELECT *
    FROM read_parquet('trades.parquet')
    WHERE created_time::TIMESTAMP >= (CURRENT_TIMESTAMP - INTERVAL '1 weeks')
    AND series_ticker LIKE 'KXMLB%'
    """
).df()

data

data.head(100).drop(["trade_id", "series_ticker", "no_price"], axis=1).to_json(
    "../coding-practice/coding-practice/kalshi_mlb.json", orient="records", lines=True
)
