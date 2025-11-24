# src/dashboard_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Price Manipulation Detector",
    layout="wide",
)

# ---- LOAD DATA ----
DATA = Path("data") / "price_history_clean.csv"
df = pd.read_csv(DATA, parse_dates=["date"])

if df.empty:
    st.error("No data found in price_history_clean.csv. Run the pipeline first.")
    st.stop()

# Ensure expected columns exist
for col in ["product_id", "price", "price_change_pct", "roll_std_7", "suspicious"]:
    if col not in df.columns:
        st.error(f"Expected column `{col}` missing from data.")
        st.stop()

# ---- PRODUCT METADATA ----
meta_cols = [c for c in ["product_name", "category", "rating", "num_reviews"] if c in df.columns]
products_info = (
    df[["product_id"] + meta_cols]
    .drop_duplicates()
    .sort_values(meta_cols[0] if meta_cols else "product_id")
)

# ---- COMPUTE SUSPICION SCORE PER PRODUCT ----
def compute_scores(data: pd.DataFrame) -> pd.DataFrame:
    grp = data.groupby("product_id")
    agg = grp.agg(
        days=("date", "count"),
        suspicious_days=("suspicious", "sum"),
        avg_abs_change_pct=("price_change_pct", lambda x: np.mean(np.abs(x))),
        avg_volatility=("roll_std_7", "mean"),
    ).reset_index()

    agg["suspicious_rate"] = agg["suspicious_days"] / agg["days"].replace(0, np.nan)

    def norm(col):
        mn, mx = col.min(), col.max()
        if pd.isna(mn) or mn == mx:
            return pd.Series(0.0, index=col.index)
        return (col - mn) / (mx - mn)

    agg["nr_susp_rate"] = norm(agg["suspicious_rate"])
    agg["nr_abs_change"] = norm(agg["avg_abs_change_pct"])
    agg["nr_vol"] = norm(agg["avg_volatility"])

    agg["suspicion_score"] = 100 * (
        0.5 * agg["nr_susp_rate"] +
        0.25 * agg["nr_abs_change"] +
        0.25 * agg["nr_vol"]
    )

    agg = agg.merge(products_info, on="product_id", how="left")

    cols_order = [
        "product_id", "product_name", "category",
        "suspicion_score",
        "suspicious_days", "days", "suspicious_rate",
        "avg_abs_change_pct", "avg_volatility",
        "rating", "num_reviews",
    ]
    existing_cols = [c for c in cols_order if c in agg.columns]
    return agg[existing_cols].sort_values("suspicion_score", ascending=False)

scores_df = compute_scores(df)
scores_df["price_integrity_score"] = 100 - scores_df["suspicion_score"]

# ---- SIDEBAR NAVIGATION ----
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("View", ["Overview", "Product detail"])

if "product_name" in products_info.columns:
    name_col = "product_name"
else:
    name_col = "product_id"

# ----------------------------------------------------------------------
#  OVERVIEW PAGE
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("📊 Price Manipulation Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tracked products", len(scores_df))
    with col2:
        st.metric("Avg suspicion score", f"{scores_df['suspicion_score'].mean():.1f}")
    with col3:
        st.metric("Avg price integrity", f"{scores_df['price_integrity_score'].mean():.1f}")

    # Explanation panel
    with st.expander("ℹ️ How is the suspicion score calculated?"):
        st.markdown(
            """
The **suspicion score** is a composite index in the range **0–100** that tries to
capture unusual price behaviour for each product.

We start from three raw signals computed per product:

1. **Suspicious rate**  
   - Fraction of days flagged as `suspicious`  
   - `suspicious_rate = suspicious_days / total_days`

2. **Average absolute daily % change**  
   - Mean of `abs(price_change_pct)`  
   - Captures how *jumpy* the price is

3. **Average volatility (7-day rolling std)**  
   - Mean of `roll_std_7`  
   - Measures short-term price variability

Each metric is min–max normalised to [0,1]:

- `nr_susp_rate`, `nr_abs_change`, `nr_vol`

Then we combine them:

> **suspicion_score = 100 × ( 0.5 · nr_susp_rate + 0.25 · nr_abs_change + 0.25 · nr_vol )**

Higher score = more suspicious *relative to other products in this dataset*.
            """
        )

    # Bar chart of suspicion scores
    bar_df = scores_df.copy()
    label_col = "product_name" if "product_name" in bar_df.columns else "product_id"
    fig_bar = px.bar(
        bar_df,
        x=label_col,
        y="suspicion_score",
        title="Suspicion score per product",
        labels={label_col: "Product", "suspicion_score": "Suspicion score (0–100)"},
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("### 🔍 Product suspicion details")
    st.dataframe(
        bar_df.set_index("product_id"),
        use_container_width=True,
    )

# ----------------------------------------------------------------------
#  PRODUCT DETAIL PAGE (WITH STEP 5: FORECAST DEVIATIONS)
# ----------------------------------------------------------------------
else:
    st.title("📈 Product Detail")

    # Product selector
    if "product_name" in products_info.columns:
        options = products_info["product_name"].tolist()
        default_name = scores_df.iloc[0].get("product_name", None)
        index_default = options.index(default_name) if default_name in options else 0
        product_name_choice = st.sidebar.selectbox(
            "Product", options=options, index=index_default
        )
        product_row = products_info[products_info["product_name"] == product_name_choice].iloc[0]
    else:
        options = products_info["product_id"].tolist()
        product_name_choice = st.sidebar.selectbox("Product ID", options=options)
        product_row = products_info[products_info["product_id"] == product_name_choice].iloc[0]

    product_id_choice = product_row["product_id"]

    # Date range
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Filter data
    sub = df[df["product_id"] == product_id_choice].copy()
    sub = sub[
        (sub["date"] >= pd.to_datetime(date_range[0])) &
        (sub["date"] <= pd.to_datetime(date_range[1]))
    ].sort_values("date")

    if sub.empty:
        st.warning("No data for this product in the selected date range.")
        st.stop()

    # Suspicion score for this product
    score_row = scores_df[scores_df["product_id"] == product_id_choice].iloc[0]
    susp_score = score_row["suspicion_score"]
    integrity_score = score_row["price_integrity_score"]

    # Header & meta
    title_label = product_row.get("product_name", f"Product {product_id_choice}")
    st.markdown(f"## {title_label}")

    meta_bits = []
    if "category" in product_row:
        meta_bits.append(f"Category: **{product_row['category']}**")
    if "rating" in product_row:
        meta_bits.append(f"Rating: **{product_row['rating']}⭐**")
    if "num_reviews" in product_row:
        meta_bits.append(f"({int(product_row['num_reviews'])} reviews)")
    st.caption(" · ".join(meta_bits))

    # We'll fill this inside the chart block
    n_big_dev = 0

    # KPI cards (now 5, including forecast deviations)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Suspicion score", f"{susp_score:.1f}")
    with col2:
        st.metric("Price integrity", f"{integrity_score:.1f}")
    with col3:
        st.metric("Average price", f"{sub['price'].mean():.2f}")
    with col4:
        st.metric("Min price", f"{sub['price'].min():.2f}")
    with col5:
        # temporary placeholder, will be updated visually by user understanding
        st.metric("Big forecast deviations", "see chart")

    # -------- MAIN CHART WITH FORECAST + STEP 5 --------
    left, right = st.columns([2, 1])

    with left:
        # Actual price line
        fig = px.line(
            sub,
            x="date",
            y="price",
            title="Price over time (actual vs forecast)",
            labels={"date": "Date", "price": "Price"},
        )

        # Rule-based suspicious points
        susp = sub[sub["suspicious"]]
        if not susp.empty:
            fig.add_scatter(
                x=susp["date"],
                y=susp["price"],
                mode="markers",
                name="Suspicious (rule-based)",
            )

        try:
            # Load forecast data
            fdf = pd.read_csv("data/price_forecast.csv", parse_dates=["date"])
            fsub = fdf[fdf["product_id"] == product_id_choice].copy()

            # Align forecast with current date range
            fsub = fsub[
                (fsub["date"] >= sub["date"].min()) &
                (fsub["date"] <= sub["date"].max())
            ].sort_values("date")

            if not fsub.empty and "forecast_price" in fsub.columns:
                # Forecast line
                fig.add_scatter(
                    x=fsub["date"],
                    y=fsub["forecast_price"],
                    mode="lines",
                    name="Forecast",
                )

                # Confidence band if available
                if "ci_lower" in fsub.columns and "ci_upper" in fsub.columns:
                    fig.add_scatter(
                        x=fsub["date"],
                        y=fsub["ci_upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                    fig.add_scatter(
                        x=fsub["date"],
                        y=fsub["ci_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(200,200,200,0.2)",
                        showlegend=False,
                    )

                # ---------- STEP 5: Forecast deviations ----------
                # Merge actual + forecast on date
                merged = pd.merge(
                    sub[["date", "price"]],
                    fsub[["date", "forecast_price"]],
                    on="date",
                    how="inner",
                )

                # Deviation & percentage deviation
                merged["deviation"] = merged["price"] - merged["forecast_price"]
                merged["abs_dev_pct"] = (
                    merged["deviation"].abs() / merged["price"]
                ) * 100

                THRESHOLD_PCT = 15.0  # tweak as you like
                merged["far_from_expected"] = merged["abs_dev_pct"] > THRESHOLD_PCT
                n_big_dev = int(merged["far_from_expected"].sum())

                big_dev = merged[merged["far_from_expected"]]

                if not big_dev.empty:
                    fig.add_scatter(
                        x=big_dev["date"],
                        y=big_dev["price"],
                        mode="markers",
                        marker=dict(color="red", size=8),
                        name=f"Deviation > {THRESHOLD_PCT:.0f}%",
                    )

            else:
                st.info(
                    "Forecast data exists but no matching dates for this selection "
                    "or missing 'forecast_price' column."
                )

        except FileNotFoundError:
            st.info("No forecast file found yet. Run `python src/forecasting.py` to generate one.")
        except Exception as e:
            st.warning(f"Could not load/use forecast data: {e}")

        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Volatility (7-day rolling)")
        st.line_chart(sub.set_index("date")["roll_std_7"], height=220)
        st.markdown("#### Daily % change")
        st.line_chart(sub.set_index("date")["price_change_pct"], height=220)

    # Detailed table
    st.markdown("### 🔍 Detailed price movements")
    with st.expander("Show raw data"):
        st.dataframe(
            sub[["date", "price", "price_change", "price_change_pct", "roll_std_7", "suspicious"]],
            use_container_width=True,
        )
