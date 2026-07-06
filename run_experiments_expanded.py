# run_experiments_expanded.py
# Expanded config space — 5 parameters instead of 3
# Adds: spark.default.parallelism + spark.memory.fraction
# Total: 4 jobs × 5 datasets × 5 shuffle × 2 memory × 2 cores
#        × 2 parallelism × 2 memory fraction = 800 experiments
# Run overnight — appends to existing job_history.csv
# Save as: C:\Aravindh\run_experiments_expanded.py

import subprocess, csv, re, os
from datetime import datetime

SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
HISTORY_PATH = "C:/Aravindh/data/job_history_expanded.csv"
DATA_DIR     = "C:/Aravindh/data/"

DATASETS = {
    "txn_500k.csv":        DATA_DIR + "txn_500k.csv",
    "txn_1m.csv":          DATA_DIR + "txn_1m.csv",
    "txn_5m.csv":          DATA_DIR + "txn_5m.csv",
    "txn_10m.csv":         DATA_DIR + "txn_10m.csv",
    "transactions_25m.csv": DATA_DIR + "transactions_25m.csv",
}

JOBS = {
    "join":        r"C:\Aravindh\day11_skew_join.py",
    "aggregation": r"C:\Aravindh\day12_aggregation_job.py",
    "filter":      r"C:\Aravindh\day13_filter_job.py",
    "window":      r"C:\Aravindh\day14_window_job.py",
}

# ── 5 config parameters ───────────────────────────────────────────
SHUFFLE_CONFIGS      = [4, 8, 16, 32, 64]
MEMORY_CONFIGS       = ["2g", "3g"]
CORES_CONFIGS        = [2, 4]
PARALLELISM_CONFIGS  = [8, 32]        # NEW: spark.default.parallelism
MEM_FRACTION_CONFIGS = [0.6, 0.8]    # NEW: spark.memory.fraction

TIMEOUT = 300

total = (len(JOBS) * len(DATASETS) * len(SHUFFLE_CONFIGS) *
         len(MEMORY_CONFIGS) * len(CORES_CONFIGS) *
         len(PARALLELISM_CONFIGS) * len(MEM_FRACTION_CONFIGS))

print("=" * 65)
print(f"  EXPANDED EXPERIMENT RUNNER — 5 Parameters")
print(f"  shuffle × memory × cores × parallelism × mem_fraction")
print(f"  {len(SHUFFLE_CONFIGS)} × {len(MEMORY_CONFIGS)} × {len(CORES_CONFIGS)} × "
      f"{len(PARALLELISM_CONFIGS)} × {len(MEM_FRACTION_CONFIGS)} = "
      f"{len(SHUFFLE_CONFIGS)*len(MEMORY_CONFIGS)*len(CORES_CONFIGS)*len(PARALLELISM_CONFIGS)*len(MEM_FRACTION_CONFIGS)} configs/job")
print(f"  Total experiments: {total}")
print(f"  ⚠️  This is a long run — leave overnight")
print("=" * 65)

count = 0

with open(HISTORY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "job_name", "dataset",
        "shuffle_partitions", "executor_memory", "executor_cores",
        "default_parallelism", "memory_fraction",
        "execution_time_sec", "timestamp"
    ])

    for job_name, script in JOBS.items():
        for dataset_name, dataset_path in DATASETS.items():
            for shuffle in SHUFFLE_CONFIGS:
                for memory in MEMORY_CONFIGS:
                    for cores in CORES_CONFIGS:
                        for parallelism in PARALLELISM_CONFIGS:
                            for mem_frac in MEM_FRACTION_CONFIGS:
                                count += 1
                                pct = count / total * 100

                                print(f"\n[{count:>4}/{total}] ({pct:4.1f}%) "
                                      f"{job_name} | {dataset_name} | "
                                      f"s={shuffle} m={memory} c={cores} "
                                      f"p={parallelism} mf={mem_frac}")

                                cmd = [
                                    SPARK_SUBMIT,
                                    "--master", "local[*]",
                                    "--executor-memory", memory,
                                    "--conf", f"spark.sql.shuffle.partitions={shuffle}",
                                    "--conf", f"spark.executor.cores={cores}",
                                    "--conf", f"spark.default.parallelism={parallelism}",
                                    "--conf", f"spark.memory.fraction={mem_frac}",
                                    script, dataset_path
                                ]

                                try:
                                    result = subprocess.run(
                                        cmd, capture_output=True,
                                        text=True, timeout=TIMEOUT
                                    )
                                    match   = re.search(
                                        r"EXECUTION_TIME:([\d.]+)", result.stdout
                                    )
                                    elapsed = float(match.group(1)) if match else -1.0
                                except subprocess.TimeoutExpired:
                                    elapsed = -2.0
                                    print("  ⏱ TIMEOUT")

                                writer.writerow([
                                    job_name, dataset_name,
                                    shuffle, memory, cores,
                                    parallelism, mem_frac, elapsed,
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ])
                                f.flush()

                                status = f"{elapsed:.2f}s" if elapsed > 0 else "FAILED"
                                print(f"  ✓ {status}")

print(f"\n✅ Expanded experiments complete → {HISTORY_PATH}")