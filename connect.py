import pandas as pd
import numpy as np
import requests

# https://kalshi.com/markets/kxfedmention/fed-mention/kxfedmention-26mar
markets_url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXFEDMENTION&status=open"  # noqa: F541
markets_response = requests.get(markets_url)
markets_data = markets_response.json()

markets_json = pd.read_json(markets_url).markets

row = []
word = []
data = pd.DataFrame()
for row in markets_json:
    word.append(row.get("custom_strike", {}).get("Word"))
    print(word)

data['word'] = word

with open("out_test.txt", "w") as f:
    for row in range(len(markets_json)):
        for keys, values in markets_json.iloc[row].items():
            print(f"{keys}: {values}", file=f)
        print("\n ================================= \n", file=f)