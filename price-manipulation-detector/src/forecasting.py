# src/forecasting.py
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

DATA_FILE = Path("data") / "price_history_clean.csv"
OUT_FILE = Path("data") / "price_forecast.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

results = []

for pid in df["product_id"].unique():
    sub = df[df["product_id"] == pid].sort_values("date").copy()
    prices = sub["price"].values

    # Skip very short series (ARIMA won't like them)
    if len(prices) < 10:
        print(f"Skipping product {pid} because series is too short ({len(prices)} points).")
        continue

    try:
        # Fit a simple ARIMA model
        model = ARIMA(prices, order=(2, 1, 2))
        model_fit = model.fit()

        # Get in-sample prediction and confidence interval
        pred = model_fit.get_forecast(steps=len(prices))
        forecast = pred.predicted_mean                             # 1D array-like

        ci = pred.conf_int(alpha=0.20)                             # 80% CI
        # ci can be a DataFrame or ndarray depending on statsmodels version -> make robust
        ci_arr = np.asarray(ci)
        if ci_arr.ndim == 2 and ci_arr.shape[1] >= 2:
            lower = ci_arr[:, 0]
            upper = ci_arr[:, 1]
        else:
            # Fallback: no CI available -> just use NaNs
            lower = np.full_like(forecast, np.nan, dtype=float)
            upper = np.full_like(forecast, np.nan, dtype=float)

        sub["forecast_price"] = forecast
        sub["ci_lower"] = lower
        sub["ci_upper"] = upper

        results.append(sub)
        print(f"Forecast computed for product {pid} (n={len(prices)})")

    except Exception as e:
        print(f"Skipping product {pid} due to model failure: {e}")

# Save combined result
if not results:
    raise RuntimeError("No forecasts were generated for any product. Check logs above.")
final = pd.concat(results, ignore_index=True)
final.to_csv(OUT_FILE, index=False)
print(f"Saved forecasting output to: {OUT_FILE.resolve()}")
