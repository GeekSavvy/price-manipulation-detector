# src/cleaning.py
import pandas as pd
from pathlib import Path

DATA = Path("data")
ph = pd.read_csv(DATA/"simulated_price_history.csv", parse_dates=["date"])
products = pd.read_csv(DATA/"raw_products.csv")

# merge metadata
df = ph.merge(products[["product_id","product_name","category","rating","num_reviews"]],
              on="product_id", how="left")

df = df.sort_values(["product_id","date"])
# price change features
df["price_change"] = df.groupby("product_id")["price"].diff().fillna(0)
df["price_change_pct"] = df.groupby("product_id")["price"].pct_change().fillna(0) * 100

# rolling features (7-day)
df["roll_mean_7"] = df.groupby("product_id")["price"].rolling(7, min_periods=1).mean().reset_index(0,drop=True)
df["roll_std_7"] = df.groupby("product_id")["price"].rolling(7, min_periods=1).std().reset_index(0,drop=True).fillna(0)

# suspicious flag: large percent change (>30%) or event == spike
df["suspicious"] = ((df["price_change_pct"].abs() > 30) | (df.get("event") == "spike"))
df.to_csv(DATA/"price_history_clean.csv", index=False)
print("Saved", DATA/"price_history_clean.csv")
