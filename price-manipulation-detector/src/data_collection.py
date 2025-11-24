# src/data_collection.py
import requests
import pandas as pd
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def fetch_products():
    resp = requests.get("https://fakestoreapi.com/products")
    resp.raise_for_status()
    products = resp.json()
    df = pd.json_normalize(products)
    # keep useful fields
    df = df[["id","title","price","category","rating.rate","rating.count"]]
    df.columns = ["product_id","product_name","price","category","rating","num_reviews"]
    df.to_csv(OUT/"raw_products.csv", index=False)
    print("Saved", OUT/"raw_products.csv")
    return df

if __name__ == "__main__":
    fetch_products()
