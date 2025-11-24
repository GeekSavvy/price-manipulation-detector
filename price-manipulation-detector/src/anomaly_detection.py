# src/anomaly_detection.py
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

DATA = Path("data")
IN_FILE = DATA / "price_history_clean.csv"
OUT_FILE = DATA / "price_history_anomalies.csv"

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"{IN_FILE} not found. Run cleaning.py first.")

    df = pd.read_csv(IN_FILE, parse_dates=["date"])

    # features we’ll use for anomaly detection
    feature_cols = ["price", "price_change_pct", "roll_std_7"]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

    df[feature_cols] = df[feature_cols].fillna(0)

    all_groups = []
    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("date").reset_index(drop=True)

        # tiny groups are not good for IsolationForest
        if len(group) < 20:
            group["iforest_score"] = 0.0
            group["iforest_anomaly"] = 0  # 0 = not evaluated
            all_groups.append(group)
            continue

        X = group[feature_cols]

        # Isolation Forest: unsupervised anomaly detection
        model = IsolationForest(
            n_estimators=100,
            contamination=0.03,   # ~3% of points are anomalies
            random_state=42
        )
        model.fit(X)

        group["iforest_score"] = model.decision_function(X)  # higher = less anomalous
        pred = model.predict(X)  # 1 = normal, -1 = anomaly
        group["iforest_anomaly"] = pred  # store as -1/1

        all_groups.append(group)

    out_df = pd.concat(all_groups, ignore_index=True)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved anomalies file: {OUT_FILE.resolve()}")
    print("Sample rows with anomalies (-1):")
    print(out_df[out_df["iforest_anomaly"] == -1].head().to_string(index=False))

if __name__ == "__main__":
    main()
