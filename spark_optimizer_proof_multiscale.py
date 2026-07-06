# spark_optimizer_proof_multiscale.py
# Runs optimizer proof across ALL dataset sizes
# Shows ML improvement scales with data size
# Save as: C:\Aravindh\spark_optimizer_proof_multiscale.py

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
OUTPUT_PATH   = "C:/Aravindh/data/multiscale_proof.csv"
SPARK_SUBMIT  = r"C:\spark\bin\spark-submit.cmd"
DATA_DIR      = "C:/Aravindh/data/"

# ALL dataset sizes — this is your key differentiator
DATASETS = [
    "txn_500k.csv",
    "txn_1m.csv",
    "txn_5m.csv",
    "txn_10m.csv",
    "transactions_25m.csv",   # largest — strongest signal
]

JOBS = {
    "join":        r"C:\Aravindh\day11_skew_join.py",
    "aggregation": r"C:\Aravindh\day12_aggregation_job.py",
    "filter":      r"C:\Aravindh\day13_filter_job.py",
    "window":      r"C:\Aravindh\day14_window_job.py",
}

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
RUNS_PER_CONFIG    = 2        # 2 runs per config to save time
TIMEOUT            = 360      # 6 min timeout for 25m jobs

model           = joblib.load(MODEL_PATH)
feature_columns = json.load(open(FEATURES_PATH))

def predict_time(features_dict):
    df = pd.DataFrame([features_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return np.expm1(model.predict(df)[0])

def run_spark_averaged(script, dataset_path, shuffle, memory, cores,
                        runs=2, timeout=360):
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
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            match = re.search(r"EXECUTION_TIME:([\d.]+)", result.stdout)
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
    print(f"      Avg: {avg:.2f}s")
    return avg, times

# ── Main ──────────────────────────────────────────────────────────────────────
rows  = []
total = len(JOBS) * len(DATASETS)
count = 0

print("=" * 65)
print("  MULTI-SCALE OPTIMIZER PROOF")
print(f"  {len(JOBS)} jobs × {len(DATASETS)} dataset sizes")
print(f"  Total scenarios: {total}")
print("=" * 65)

for dataset in DATASETS:
    dataset_path = DATA_DIR + dataset
    size_mb      = os.path.getsize(dataset_path) / 1024 / 1024

    print(f"\n{'═'*65}")
    print(f"📦 DATASET: {dataset} ({size_mb:.0f} MB)")
    print(f"{'═'*65}")

    for job_name, script in JOBS.items():
        count += 1
        print(f"\n  [{count}/{total}] {job_name.upper()}")

        # ML recommendation
        sql  = SQL_TEMPLATES[job_name]
        feat = build_feature_vector(sql, dataset_path)
        feat["job_type"] = job_name

        best_predicted = float("inf")
        best_cfg       = None

        for shuffle in SHUFFLE_CANDIDATES:
            for memory in MEMORY_CANDIDATES:
                for cores in CORES_CANDIDATES:
                    f = feat.copy()
                    f["shuffle_partitions"] = shuffle
                    f["executor_memory_gb"] = float(memory.replace("g", ""))
                    f["executor_cores"]     = cores
                    pred = predict_time(f)
                    if pred < best_predicted:
                        best_predicted = pred
                        best_cfg = (shuffle, memory, cores)

        bs, bm, bc = best_cfg
        print(f"  ML recommends: shuffle={bs}, mem={bm}, cores={bc} "
              f"(predicted {best_predicted:.1f}s)")

        # Run DEFAULT
        print(f"  Default (s={DEFAULT_SHUFFLE} m={DEFAULT_MEMORY} "
              f"c={DEFAULT_CORES}):")
        default_avg, _ = run_spark_averaged(
            script, dataset_path,
            DEFAULT_SHUFFLE, DEFAULT_MEMORY, DEFAULT_CORES,
            runs=RUNS_PER_CONFIG, timeout=TIMEOUT
        )

        # Run ML-OPTIMIZED
        print(f"  ML-Opt (s={bs} m={bm} c={bc}):")
        ml_avg, _ = run_spark_averaged(
            script, dataset_path,
            bs, bm, bc,
            runs=RUNS_PER_CONFIG, timeout=TIMEOUT
        )

        if default_avg > 0 and ml_avg > 0:
            pct       = round((default_avg - ml_avg) / default_avg * 100, 2)
            saved     = round(default_avg - ml_avg, 2)
            status    = "✅" if pct > 0 else "❌"
        else:
            pct    = 0
            saved  = 0
            status = "⚠️ "

        print(f"  {status} {pct:+.1f}% ({saved:+.2f}s saved)")

        rows.append({
            "dataset":         dataset,
            "dataset_mb":      round(size_mb, 1),
            "job_name":        job_name,
            "default_time":    default_avg,
            "ml_time":         ml_avg,
            "ml_shuffle":      bs,
            "ml_memory":       bm,
            "ml_cores":        bc,
            "improvement_pct": pct,
            "time_saved_sec":  saved,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# ── Save + Summary ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*65}")
print("  MULTI-SCALE SUMMARY")
print(f"{'='*65}")
print(f"\n  {'Dataset':<25} {'Job':<14} {'Default':>8} "
      f"{'ML':>8} {'Improv':>9}")
print(f"  {'─'*66}")

for _, row in df.iterrows():
    arrow = "✅" if row["improvement_pct"] > 0 else "❌"
    print(f"  {row['dataset']:<25} {row['job_name']:<14} "
          f"{row['default_time']:>7.1f}s "
          f"{row['ml_time']:>7.1f}s "
          f"{row['improvement_pct']:>+8.1f}% {arrow}")

# Summary by dataset size
print(f"\n  Average improvement by dataset size:")
print(f"  {'─'*40}")
for ds, grp in df.groupby("dataset"):
    valid = grp[grp["improvement_pct"] != 0]
    if len(valid) > 0:
        avg_imp = valid["improvement_pct"].mean()
        wins    = len(valid[valid["improvement_pct"] > 0])
        print(f"  {ds:<25} {avg_imp:>+6.1f}% avg  "
              f"({wins}/{len(valid)} jobs improved)")

overall = df[df["improvement_pct"] != 0]["improvement_pct"].mean()
print(f"\n  Overall average improvement: {overall:+.1f}%")
print(f"  Total jobs improved: "
      f"{len(df[df['improvement_pct']>0])}/{len(df)}")

print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nNext: python run_aqe_baseline.py")