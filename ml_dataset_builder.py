# ml_dataset_builder.py  — UPDATED for banking dataset
# Now includes executor_cores as a config feature
# Save as: C:\Aravindh\ml_dataset_builder.py  (REPLACE old version)

import pandas as pd
import os
from feature_builder import build_feature_vector

HISTORY_PATH  = "C:/Aravindh/data/job_history.csv"
OUTPUT_PATH   = "C:/Aravindh/data/ml_dataset.csv"
DATA_DIR      = "C:/Aravindh/data/"

# Map dataset names to full paths
DATASET_PATHS = {
    "txn_500k.csv" : DATA_DIR + "txn_500k.csv",
    "txn_1m.csv"   : DATA_DIR + "txn_1m.csv",
    "txn_5m.csv"   : DATA_DIR + "txn_5m.csv",
    "txn_10m.csv"  : DATA_DIR + "txn_10m.csv",
    "txn_25m.csv"  : DATA_DIR + "transactions_25m.csv",
}

# SQL templates per job type
SQL_TEMPLATES = {
    "join":
        "SELECT category, account_type, country, COUNT(*), SUM(amount), "
        "AVG(amount), SUM(is_fraud), AVG(credit_score) "
        "FROM transactions JOIN accounts ON account_id "
        "JOIN merchants ON merchant_id GROUP BY category, account_type, country",

    "aggregation":
        "SELECT merchant_id, transaction_type, channel, COUNT(*), SUM(amount), "
        "AVG(amount), STDDEV(amount), MIN(amount), MAX(amount), "
        "SUM(is_fraud), AVG(is_fraud), COUNTDISTINCT(account_id) "
        "FROM transactions GROUP BY merchant_id, transaction_type, channel",

    "filter":
        "SELECT location, channel, day_of_week, COUNT(*), SUM(amount), "
        "SUM(is_fraud), AVG(amount) FROM transactions "
        "WHERE amount > 1000 AND status != failed "
        "AND transaction_type IN purchase, transfer, withdrawal "
        "AND hour_of_day BETWEEN 0 AND 5 OR is_fraud = 1 "
        "GROUP BY location, channel, day_of_week",

    "window":
        "SELECT account_id, amount, RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "DENSE_RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "PERCENT_RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "SUM(amount) OVER(PARTITION BY merchant_id ORDER BY timestamp) "
        "FROM transactions",
}

print("=" * 55)
print("  ML DATASET BUILDER — Banking Version")
print("=" * 55)

history = pd.read_csv(HISTORY_PATH)
print(f"\n  Loaded {len(history)} experiment records")

# Remove failed/timeout rows
history = history[history["execution_time_sec"] > 0].copy()
print(f"  Valid rows after filtering: {len(history)}")

# Subtract JVM baseline (10th percentile)
jvm_baseline = history["execution_time_sec"].quantile(0.10)
print(f"  JVM baseline (p10): {jvm_baseline:.2f}s")
history["execution_time_sec"] = (
    history["execution_time_sec"] - jvm_baseline
).clip(lower=0.1)

ml_rows = []

for _, row in history.iterrows():
    job_name    = row["job_name"]
    dataset     = row["dataset"]
    shuffle     = row["shuffle_partitions"]
    memory      = row["executor_memory"]
    cores       = row.get("executor_cores", 2)
    exec_time   = row["execution_time_sec"]

    dataset_path = DATASET_PATHS.get(dataset)
    if not dataset_path:
        continue

    sql  = SQL_TEMPLATES.get(job_name, "")
    feat = build_feature_vector(sql, dataset_path)

    feat["shuffle_partitions"] = shuffle
    feat["executor_memory_gb"] = float(str(memory).replace("g", ""))
    feat["executor_cores"]     = int(cores)    # NEW feature
    feat["job_type"]           = job_name
    feat["execution_time_sec"] = exec_time

    ml_rows.append(feat)

df = pd.DataFrame(ml_rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n  ML dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"  Features: {[c for c in df.columns if c != 'execution_time_sec']}")
print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nNext step: python train_optimizer_model.py")