# run_experiments_5m.py
import subprocess
import csv
import time
import os
from datetime import datetime

spark_submit  = r"C:\spark\bin\spark-submit.cmd"
HISTORY_PATH  = "C:/Aravindh/data/job_history.csv"

# ONLY 5m dataset tonight
datasets = {
    "emp_5m.csv": "C:/Aravindh/data/emp_5m.csv",
}

shuffle_configs = [4, 8, 16, 32, 64]
memory_configs  = ["2g", "4g"]

spark_jobs = {
    "join":        r"C:\Aravindh\day11_skew_join.py",
    "aggregation": r"C:\Aravindh\day12_aggregation_job.py",
    "filter":      r"C:\Aravindh\day13_filter_job.py",
    "window":      r"C:\Aravindh\day14_window_job.py",
}

# Append to existing history — don't overwrite!
log_file = open(HISTORY_PATH, "a", newline="")
writer   = csv.writer(log_file)

total = len(spark_jobs) * len(datasets) * len(shuffle_configs) * len(memory_configs)
count = 0

print(f"🌙 Overnight run starting — {total} jobs")
print(f"   Appending to: {HISTORY_PATH}\n")

for job_name, spark_script in spark_jobs.items():
    for dataset_name, dataset_path in datasets.items():
        for shuffle in shuffle_configs:
            for memory in memory_configs:

                count += 1
                print(f"[{count}/{total}] job={job_name} | {dataset_name} "
                      f"| shuffle={shuffle} | memory={memory}")

                cmd = [
                    spark_submit,
                    "--master",          "local[*]",
                    "--executor-memory", memory,
                    "--conf", f"spark.sql.shuffle.partitions={shuffle}",
                    spark_script,
                    dataset_path
                ]

                start   = time.time()
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elapsed = round(time.time() - start, 2)

                writer.writerow([
                    job_name, dataset_name, shuffle,
                    memory, elapsed,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
                log_file.flush()
                print(f"    ✓ {elapsed}s")

log_file.close()
print(f"\n✅ Done! {total} experiments complete.")