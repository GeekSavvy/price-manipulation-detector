# src/dashboard_streamlit_anomaly.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide", page_title="Price Manipulation Detector - Intermediate")

DATA = Path("data")
ANOM_FILE = DATA / "price_history_anomalies.csv"

if not ANOM_FILE.exists():
    st.error("price_history_anomalies.csv not found. Run anomaly_detection.py first.")
    st.stop()

df = pd.read_csv(ANOM_FILE, parse_dates=["date"])

st.title("Price Manipulation Detector — Intermediate (With Anomaly Detection)")

# ---- Sidebar filters ----
product_list = df["product_name"].unique()
prod_choice = st.sidebar.selectbox("Product", options=product_list)

date_min = df["date"].min()
date_max = df["date"].max()
date_range = st.sidebar.date_input("Date range", [date_min, date_max])

# filter by product and date
sub = df[df["product_name"] == prod_choice].copy()
sub = sub[(sub["date"] >= pd.to_datetime(date_range[0])) &
          (sub["date"] <= pd.to_datetime(date_range[1]))]

# derive some metrics
total_days = len(sub)
num_iforest_anom = int((sub["iforest_anomaly"] == -1).sum())
num_rule_suspicious = int(sub["suspicious"].sum()) if "suspicious" in sub.columns else 0

anom_pct = (num_iforest_anom / total_days * 100) if total_days > 0 else 0.0

st.subheader(prod_choice)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Days in range", total_days)
with col2:
    st.metric("Model anomalies (IsolationForest)", f"{num_iforest_anom} ({anom_pct:.1f}%)")
with col3:
    st.metric("Rule-based suspicious days", num_rule_suspicious)

# ---- Price chart with anomalies highlighted ----
fig = px.line(sub, x="date", y="price", title="Price over time")

# anomalies from model
iforest_anom = sub[sub["iforest_anomaly"] == -1]
if not iforest_anom.empty:
    fig.add_scatter(
        x=iforest_anom["date"],
        y=iforest_anom["price"],
        mode="markers",
        name="IsolationForest anomaly",
        marker=dict(size=10, symbol="x")
    )

# rule-based suspicious flag
if "suspicious" in sub.columns:
    rule_anom = sub[sub["suspicious"] == True]
    if not rule_anom.empty:
        fig.add_scatter(
            x=rule_anom["date"],
            y=rule_anom["price"],
            mode="markers",
            name="Rule-based suspicious",
            marker=dict(size=9, symbol="circle-open")
        )

st.plotly_chart(fig, use_container_width=True)

# ---- Extra views ----
st.markdown("### Anomaly details")

tab1, tab2 = st.tabs(["Model anomalies", "All data"])

with tab1:
    anom_rows = sub[sub["iforest_anomaly"] == -1].copy()
    if anom_rows.empty:
        st.info("No anomalies detected for this product in the selected date range.")
    else:
        show_cols = ["date", "price", "price_change", "price_change_pct",
                     "roll_mean_7", "roll_std_7", "iforest_score"]
        show_cols = [c for c in show_cols if c in anom_rows.columns]
        st.dataframe(anom_rows[show_cols])

with tab2:
    show_cols = ["date", "price", "price_change", "price_change_pct",
                 "roll_mean_7", "roll_std_7", "iforest_anomaly", "suspicious"]
    show_cols = [c for c in show_cols if c in sub.columns]
    st.dataframe(sub[show_cols])
