# src/simulate_prices.py (debug version)
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys, traceback, os

DATA = Path("data")
DATA.mkdir(exist_ok=True)

RAW_FILE = DATA / "raw_products.csv"
OUT_FILE = DATA / "simulated_price_history.csv"

def simulate(product_row, start_date, days=90,
             vol=0.02, sale_prob=0.03, spike_prob=0.005):
    prices = []
    try:
        base = float(product_row.price)
    except Exception:
        # try alternative column names
        if "price" in product_row.index:
            base = float(product_row["price"])
        else:
            base = 0.0
    price = base
    for i in range(days):
        date = start_date + timedelta(days=i)
        daily_return = np.random.normal(loc=0, scale=vol)
        price = price * (1 + daily_return)
        if np.random.rand() < sale_prob:
            discount = np.random.uniform(0.05, 0.5)
            price = price * (1 - discount)
            event = "sale"
        elif np.random.rand() < spike_prob:
            factor = np.random.choice([1.5, 0.5, 2.0])
            price = price * factor
            event = "spike"
        else:
            event = None
        price = max(0.01, round(price, 2))
        prices.append({
            "product_id": product_row.get("product_id", product_row.get("id", None)),
            "date": date.strftime("%Y-%m-%d"),
            "price": price,
            "event": event
        })
    return prices

def main(days=180):
    try:
        print("Working dir:", Path.cwd())
        print("Looking for:", RAW_FILE.resolve())
        if not RAW_FILE.exists():
            print("ERROR: raw_products.csv not found at", RAW_FILE)
            print("Please run data_collection.py first (or place raw_products.csv in data/).")
            sys.exit(1)
        products = pd.read_csv(RAW_FILE)
        print("raw_products.csv loaded. Shape:", products.shape)
        if products.shape[0] == 0:
            print("ERROR: raw_products.csv is empty (0 rows).")
            sys.exit(1)
        print("Preview of first rows:")
        print(products.head().to_string(index=False))
        rows = []
        start = datetime.today() - timedelta(days=days)
        for idx, r in products.iterrows():
            rows.extend(simulate(r, start_date=start, days=days))
        df = pd.DataFrame(rows)
        print("Generated rows:", len(df))
        # ensure we actually have rows
        if df.shape[0] == 0:
            print("ERROR: No simulated rows generated. Exiting.")
            sys.exit(1)
        out_path = OUT_FILE.resolve()
        df.to_csv(OUT_FILE, index=False)
        print("Saved", out_path)
        print("First 5 rows written:")
        print(df.head().to_string(index=False))
    except Exception as e:
        print("EXCEPTION in simulate_prices.py:", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main(days=180)
