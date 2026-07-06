# run_experiments.py  — UPDATED for banking dataset
# Now tests 3 config parameters (was 2):
#   shuffle_partitions, executor_memory, executor_cores
# Total: 4 jobs × 5 datasets × 5 shuffle × 2 memory × 2 cores = 400 experiments
#
# Save as: C:\Aravindh\run_experiments.py  (REPLACE old version)

import subprocess
import csv
import re
import os
import time
from datetime import datetime

SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
HISTORY_PATH = "C:/Aravindh/data/job_history.csv"
DATA_DIR     = "C:/Aravindh/data/"

# ── Dataset variants ──────────────────────────────────────────────────────────
DATASETS = {
    "txn_500k.csv" : DATA_DIR + "txn_500k.csv",
    "txn_1m.csv"   : DATA_DIR + "txn_1m.csv",
    "txn_5m.csv"   : DATA_DIR + "txn_5m.csv",
    "txn_10m.csv"  : DATA_DIR + "txn_10m.csv",
    "txn_25m.csv"  : DATA_DIR + "transactions_25m.csv",
}

# ── Spark jobs ────────────────────────────────────────────────────────────────
JOBS = {
    "join":        r"C:\Aravindh\day11_skew_join.py",
    "aggregation": r"C:\Aravindh\day12_aggregation_job.py",
    "filter":      r"C:\Aravindh\day13_filter_job.py",
    "window":      r"C:\Aravindh\day14_window_job.py",
}

# ── Config grid (3 parameters now) ───────────────────────────────────────────
SHUFFLE_CONFIGS = [4, 8, 16, 32, 64]
MEMORY_CONFIGS  = ["2g", "3g"]
CORES_CONFIGS   = [2, 4]          # NEW: executor cores

TIMEOUT = 300  # seconds

# ── Clear old results ─────────────────────────────────────────────────────────
if os.path.exists(HISTORY_PATH):
    os.remove(HISTORY_PATH)
    print(f"🗑  Cleared old {HISTORY_PATH}")

# ── Calculate total ───────────────────────────────────────────────────────────
total = (len(JOBS) * len(DATASETS) *
         len(SHUFFLE_CONFIGS) * len(MEMORY_CONFIGS) * len(CORES_CONFIGS))
count = 0

print("=" * 65)
print(f"  BANKING EXPERIMENT RUNNER")
print(f"  Total experiments: {total}")
print(f"  Jobs × Datasets × Shuffle × Memory × Cores")
print(f"  {len(JOBS)} × {len(DATASETS)} × {len(SHUFFLE_CONFIGS)} × "
      f"{len(MEMORY_CONFIGS)} × {len(CORES_CONFIGS)}")
print("=" * 65)

with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "job_name", "dataset", "shuffle_partitions",
        "executor_memory", "executor_cores",
        "execution_time_sec", "timestamp"
    ])

    for job_name, script in JOBS.items():
        for dataset_name, dataset_path in DATASETS.items():

            # Skip large datasets for quick first run
            # Comment this out for full overnight run
            # if dataset_name in ["txn_10m.csv", "txn_25m.csv"]:
            #     continue

            for shuffle in SHUFFLE_CONFIGS:
                for memory in MEMORY_CONFIGS:
                    for cores in CORES_CONFIGS:
                        count += 1
                        pct = count / total * 100

                        print(f"\n[{count:>3}/{total}] ({pct:4.1f}%) "
                              f"{job_name} | {dataset_name} | "
                              f"s={shuffle} m={memory} c={cores}")

                        cmd = [
                            SPARK_SUBMIT,
                            "--master", "local[*]",
                            "--executor-memory", memory,
                            "--conf", f"spark.sql.shuffle.partitions={shuffle}",
                            "--conf", f"spark.executor.cores={cores}",
                            script, dataset_path
                        ]

                        try:
                            result = subprocess.run(
                                cmd, capture_output=True,
                                text=True, timeout=TIMEOUT
                            )
                            match   = re.search(r"EXECUTION_TIME:([\d.]+)", result.stdout)
                            elapsed = float(match.group(1)) if match else -1.0
                        except subprocess.TimeoutExpired:
                            elapsed = -2.0
                            print("  ⏱ TIMEOUT")

                        writer.writerow([
                            job_name, dataset_name, shuffle,
                            memory, cores, elapsed,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])
                        f.flush()

                        status = f"{elapsed:.2f}s" if elapsed > 0 else "FAILED"
                        print(f"  ✓ {status}")

print(f"\n✅ All {count} experiments complete")
print(f"✅ Saved to: {HISTORY_PATH}")
print("\nNext step: python ml_dataset_builder.py")