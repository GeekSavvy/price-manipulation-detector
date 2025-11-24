import streamlit as st, pandas as pd, plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide")
st.title("DEBUG Dashboard — should show a plot below")

DATA = Path("data") / "price_history_clean.csv"
st.write("data file path:", str(DATA.resolve()))

try:
    df = pd.read_csv(DATA, parse_dates=["date"])
    st.write("Loaded rows:", len(df))
except Exception as e:
    st.error("Failed to load CSV: " + str(e))
    st.stop()

st.write("Products sample:", df.product_id.unique()[:10].tolist())

prod = df.product_id.unique()[0]
sub = df[df.product_id == prod].sort_values("date")

st.write("Rows for product", prod, ":", len(sub))

if len(sub) > 0:
    fig = px.line(sub, x="date", y="price", title=f"Product {prod}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No rows for selected product")
