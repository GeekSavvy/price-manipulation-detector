# Price Manipulation Detector

An end-to-end analytics project to **detect suspicious price behavior** for retail products.

The system:
- Collects product and price data from a public API
- Simulates realistic price histories (sales, spikes, volatility)
- Engineers features for price dynamics and volatility
- Detects anomalies using machine learning (Isolation Forest)
- Visualizes everything in an **interactive Streamlit dashboard**

> Repository: https://github.com/GeekSavvy/price-manipulation-detector

---

## ✨ Key Features

- 📥 Data collection
  - Fetches product catalog and metadata from `https://fakestoreapi.com/products`
- 🧪 Price simulation
  - Generates synthetic daily price histories (sales, spikes, noise)
- 🧹 Data cleaning & feature engineering
  - Rolling statistics, price change %, volatility, suspicious flags
- 🕵️ Anomaly detection
  - Isolation Forest scores for each product’s price history
- 📈 Forecasting
  - ARIMA-based forecasts for price trends
- 📊 Interactive dashboard
  - Product-level drilldown
  - Suspicion score per product
  - Time-series plots with anomalies highlighted

---

## 🧱 Project Structure

```text
price-manipulation-detector/
├─ data/
│  ├─ raw_products.csv               # collected product data
│  ├─ simulated_price_history.csv    # simulated daily prices
│  ├─ price_history_clean.csv        # cleaned & feature-engineered data
│  ├─ price_history_anomalies.csv    # with anomaly flags & scores
│  └─ price_forecast.csv             # ARIMA forecasts
│
├─ notebooks/
│  ├─ 01_data_collection.ipynb
│  └─ 02_cleaning_and_eda.ipynb
│
├─ src/
│  ├─ data_collection.py             # fetch products from API
│  ├─ simulate_prices.py             # simulate price histories
│  ├─ cleaning.py                    # merge + feature engineering
│  ├─ anomaly_detection.py           # Isolation Forest detection
│  ├─ forecasting.py                 # ARIMA-based price forecasting
│  ├─ dashboard_streamlit.py         # main Streamlit dashboard
│  ├─ dashboard_streamlit_anomaly.py # anomaly-focused view
│  ├─ dashboard_streamlit_debug.py   # debug variant of dashboard
│  └─ test_streamlit.py              # basic smoke tests / utilities
│
├─ .streamlit/
│  └─ config.toml                    # Streamlit page config (if used)
│
├─ requirements.txt
└─ README.md
