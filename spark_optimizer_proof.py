# spark_optimizer_proof.py — UPDATED for banking dataset
# Proves ML recommendation beats default Spark config
# Uses 3x averaged runs to remove noise
# Save as: C:\Aravindh\spark_optimizer_proof.py

import pandas as pd
import joblib
import json
import numpy as np
import subprocess
import re
import os
from datetime import datetime
from feature_builder import build_feature_vector

MODEL_PATH    = "C:/Aravindh/models/spark_optimizer.pkl"
FEATURES_PATH = "C:/Aravindh/models/feature_columns.json"
OUTPUT_PATH   = "C:/Aravindh/data/optimization_proof.csv"
SPARK_SUBMIT  = r"C:\spark\bin\spark-submit.cmd"
DATA_DIR      = "C:/Aravindh/data/"

RUNS_PER_CONFIG = 3

TEST_SCENARIOS = [
    {"job_name": "join",        "script": r"C:\Aravindh\day11_skew_join.py",        "dataset": "txn_5m.csv"},
    {"job_name": "aggregation", "script": r"C:\Aravindh\day12_aggregation_job.py",  "dataset": "txn_5m.csv"},
    {"job_name": "filter",      "script": r"C:\Aravindh\day13_filter_job.py",       "dataset": "txn_5m.csv"},
    {"job_name": "window",      "script": r"C:\Aravindh\day14_window_job.py",       "dataset": "txn_5m.csv"},
]

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
        "WHERE amount > 1000 AND is_fraud = 1 GROUP BY location, channel, day_of_week",
    "window":
        "SELECT account_id, amount, RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "DENSE_RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "PERCENT_RANK() OVER(PARTITION BY account_id ORDER BY amount), "
        "SUM(amount) OVER(PARTITION BY merchant_id ORDER BY timestamp) "
        "FROM transactions",
}

SHUFFLE_CANDIDATES = [4, 8, 16, 32, 64]
MEMORY_CANDIDATES  = ["2g", "3g"]
CORES_CANDIDATES   = [2, 4]
DEFAULT_SHUFFLE    = 200
DEFAULT_MEMORY     = "1g"
DEFAULT_CORES      = 2

model           = joblib.load(MODEL_PATH)
feature_columns = json.load(open(FEATURES_PATH))

def predict_time(features_dict):
    df = pd.DataFrame([features_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    import numpy as np
    return np.expm1(model.predict(df)[0])  # inverse log transform

def run_spark_averaged(script, dataset_path, shuffle, memory, cores, runs=3, timeout=300):
    times = []
    for i in range(runs):
        cmd = [
            SPARK_SUBMIT, "--master", "local[*]",
            "--executor-memory", memory,
            "--conf", f"spark.sql.shuffle.partitions={shuffle}",
            "--conf", f"spark.executor.cores={cores}",
            script, dataset_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            match  = re.search(r"EXECUTION_TIME:([\d.]+)", result.stdout)
            if match:
                t = float(match.group(1))
                times.append(t)
                print(f"      Run {i+1}: {t:.2f}s")
            else:
                print(f"      Run {i+1}: FAILED")
        except subprocess.TimeoutExpired:
            print(f"      Run {i+1}: TIMEOUT")

    if not times:
        return -1.0, []
    avg = round(sum(times) / len(times), 3)
    print(f"      Average: {avg:.2f}s ({len(times)} runs)")
    return avg, times

# ── Main ──────────────────────────────────────────────────────────────────────
rows = []

print("=" * 65)
print(f"  ML OPTIMIZER PROOF — Banking Dataset ({RUNS_PER_CONFIG}x Averaged)")
print("=" * 65)

for scenario in TEST_SCENARIOS:
    job_name     = scenario["job_name"]
    script       = scenario["script"]
    dataset      = scenario["dataset"]
    dataset_path = DATA_DIR + dataset

    print(f"\n{'─'*65}")
    print(f"📌 Job: {job_name.upper()} | Dataset: {dataset}")

    # ── ML recommendation ──────────────────────────────────────────
    sql  = SQL_TEMPLATES[job_name]
    feat = build_feature_vector(sql, dataset_path)
    feat["job_type"] = job_name

    best_predicted = float("inf")
    best_shuffle   = None
    best_memory    = None
    best_cores     = None

    print("  🔍 Scanning configurations...")
    for shuffle in SHUFFLE_CANDIDATES:
        for memory in MEMORY_CANDIDATES:
            for cores in CORES_CANDIDATES:
                f = feat.copy()
                f["shuffle_partitions"]  = shuffle
                f["executor_memory_gb"]  = float(memory.replace("g",""))
                f["executor_cores"]      = cores
                predicted = predict_time(f)
                if predicted < best_predicted:
                    best_predicted = predicted
                    best_shuffle   = shuffle
                    best_memory    = memory
                    best_cores     = cores

    print(f"  ✅ ML recommends: shuffle={best_shuffle}, "
          f"memory={best_memory}, cores={best_cores}")
    print(f"     Predicted time: {best_predicted:.2f}s")

    # ── Run DEFAULT config ─────────────────────────────────────────
    print(f"\n  ⏱ DEFAULT (shuffle={DEFAULT_SHUFFLE}, "
          f"mem={DEFAULT_MEMORY}, cores={DEFAULT_CORES}) × {RUNS_PER_CONFIG} runs:")
    default_avg, default_runs = run_spark_averaged(
        script, dataset_path,
        DEFAULT_SHUFFLE, DEFAULT_MEMORY, DEFAULT_CORES,
        runs=RUNS_PER_CONFIG
    )

    # ── Run ML-RECOMMENDED config ──────────────────────────────────
    print(f"\n  ⏱ ML-OPTIMIZED (shuffle={best_shuffle}, "
          f"mem={best_memory}, cores={best_cores}) × {RUNS_PER_CONFIG} runs:")
    optimized_avg, optimized_runs = run_spark_averaged(
        script, dataset_path,
        best_shuffle, best_memory, best_cores,
        runs=RUNS_PER_CONFIG
    )

    # ── Results ────────────────────────────────────────────────────
    if default_avg > 0 and optimized_avg > 0:
        improvement_pct = round((default_avg - optimized_avg) / default_avg * 100, 2)
        time_saved      = round(default_avg - optimized_avg, 2)
    else:
        improvement_pct = 0
        time_saved      = 0

    status = "✅ IMPROVED" if improvement_pct > 0 else "❌ WORSE"
    print(f"\n  {status}: {improvement_pct:+.1f}% ({abs(time_saved):.2f}s)")

    rows.append({
        "job_name":          job_name,
        "dataset":           dataset,
        "runs_per_config":   RUNS_PER_CONFIG,
        "default_shuffle":   DEFAULT_SHUFFLE,
        "default_memory":    DEFAULT_MEMORY,
        "default_cores":     DEFAULT_CORES,
        "default_time_avg":  default_avg,
        "default_all_runs":  str(default_runs),
        "ml_shuffle":        best_shuffle,
        "ml_memory":         best_memory,
        "ml_cores":          best_cores,
        "ml_time_avg":       optimized_avg,
        "ml_all_runs":       str(optimized_runs),
        "improvement_pct":   improvement_pct,
        "time_saved_sec":    time_saved,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ── Summary ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*65}")
print("  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  {'Job':<14} {'Default':>9} {'ML Opt':>9} {'Improvement':>12}")
print(f"  {'─'*48}")
for _, row in df.iterrows():
    arrow = "✅" if row['improvement_pct'] > 0 else "❌"
    print(f"  {row['job_name']:<14} "
          f"{row['default_time_avg']:>8.2f}s "
          f"{row['ml_time_avg']:>8.2f}s "
          f"{row['improvement_pct']:>+11.1f}% {arrow}")

improved     = df[df["improvement_pct"] > 0]
avg_all      = df["improvement_pct"].mean()
avg_improved = improved["improvement_pct"].mean() if len(improved) > 0 else 0

print(f"\n  Jobs improved:           {len(improved)}/{len(df)}")
print(f"  Avg improvement (all):   {avg_all:+.1f}%")
print(f"  Avg improvement (wins):  {avg_improved:+.1f}%")
print(f"  Best result:             {df['improvement_pct'].max():+.1f}%")
print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nNext step: python run_aqe_baseline.py")