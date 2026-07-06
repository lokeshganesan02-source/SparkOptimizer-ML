# run_aqe_baseline.py — UPDATED for banking dataset
# Compares: Default | AQE ON | ML-Optimized | ML+AQE
# Save as: C:\Aravindh\run_aqe_baseline.py

import subprocess, csv, re, os
from datetime import datetime

SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
OUTPUT_PATH  = "C:/Aravindh/data/aqe_comparison.csv"
DATA_DIR     = "C:/Aravindh/data/"

SCENARIOS = [
    {"job_name": "join",        "script": r"C:\Aravindh\day11_skew_join.py",        "dataset": "txn_5m.csv"},
    {"job_name": "aggregation", "script": r"C:\Aravindh\day12_aggregation_job.py",  "dataset": "txn_5m.csv"},
    {"job_name": "filter",      "script": r"C:\Aravindh\day13_filter_job.py",       "dataset": "txn_5m.csv"},
    {"job_name": "window",      "script": r"C:\Aravindh\day14_window_job.py",       "dataset": "txn_5m.csv"},
]

# Per-job ML best config from optimizer proof results
ML_BEST = {
    "join":        {"shuffle": 16, "memory": "2g", "cores": 4},
    "aggregation": {"shuffle": 16, "memory": "2g", "cores": 4},
    "filter":      {"shuffle": 32, "memory": "3g", "cores": 2},
    "window":      {"shuffle": 16, "memory": "2g", "cores": 4},
}

def run_job(script, dataset_path, shuffle, memory, cores, aqe, timeout=240):
    cmd = [
        SPARK_SUBMIT, "--master", "local[*]",
        "--executor-memory", memory,
        "--conf", f"spark.sql.shuffle.partitions={shuffle}",
        "--conf", f"spark.executor.cores={cores}",
        "--conf", f"spark.sql.adaptive.enabled={aqe}",
        script, dataset_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        match  = re.search(r"EXECUTION_TIME:([\d.]+)", result.stdout)
        return float(match.group(1)) if match else -1.0
    except subprocess.TimeoutExpired:
        print("    ⏱ TIMEOUT")
        return -1.0

rows   = []
total  = len(SCENARIOS) * 4   # 4 configs per job
count  = 0

print("=" * 65)
print("  AQE BASELINE COMPARISON — Banking Dataset")
print("  Default | AQE ON | ML-Optimized | ML+AQE")
print("=" * 65)

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "job_name", "dataset", "config",
        "shuffle", "memory", "cores", "aqe",
        "execution_time_sec", "timestamp"
    ])

    for s in SCENARIOS:
        job      = s["job_name"]
        script   = s["script"]
        dataset  = s["dataset"]
        dpath    = DATA_DIR + dataset
        ml       = ML_BEST[job]

        print(f"\n📌 {job.upper()} | {dataset}")

        configs = [
            ("Default",  200,           "1g",          2,           "false"),
            ("AQE_ON",   200,           "1g",          2,           "true"),
            ("ML_OPT",   ml["shuffle"], ml["memory"],  ml["cores"], "false"),
            ("ML+AQE",   ml["shuffle"], ml["memory"],  ml["cores"], "true"),
        ]

        times = {}
        for label, shuffle, memory, cores, aqe in configs:
            count += 1
            print(f"  [{label}] shuffle={shuffle} mem={memory} "
                  f"cores={cores} aqe={aqe}...")
            t = run_job(script, dpath, shuffle, memory, cores, aqe)
            times[label] = t
            status = f"{t:.2f}s" if t > 0 else "FAILED"
            print(f"    ✓ {status}")

            writer.writerow([
                job, dataset, label,
                shuffle, memory, cores, aqe, t,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
            f.flush()

        # Per-job summary
        default = times["Default"]
        aqe_t   = times["AQE_ON"]
        ml_t    = times["ML_OPT"]
        mlaqe_t = times["ML+AQE"]

        if default > 0:
            print(f"\n  vs Default:")
            if aqe_t   > 0: print(f"    AQE:    {(default-aqe_t)/default*100:+.1f}%")
            if ml_t    > 0: print(f"    ML:     {(default-ml_t)/default*100:+.1f}%")
            if mlaqe_t > 0: print(f"    ML+AQE: {(default-mlaqe_t)/default*100:+.1f}%")

        rows.append({
            "job":     job,
            "default": default,
            "aqe":     aqe_t,
            "ml":      ml_t,
            "ml_aqe":  mlaqe_t,
        })

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL COMPARISON TABLE")
print(f"{'='*65}")
print(f"  {'Job':<14} {'Default':>9} {'AQE':>9} {'ML':>9} {'ML+AQE':>9}")
print(f"  {'─'*54}")
for r in rows:
    print(f"  {r['job']:<14} "
          f"{r['default']:>8.2f}s "
          f"{r['aqe']:>8.2f}s "
          f"{r['ml']:>8.2f}s "
          f"{r['ml_aqe']:>8.2f}s")

# Best config per job
print(f"\n  Best config per job:")
for r in rows:
    times_d = {"Default": r["default"], "AQE": r["aqe"],
                "ML": r["ml"], "ML+AQE": r["ml_aqe"]}
    valid   = {k: v for k, v in times_d.items() if v > 0}
    if valid:
        best = min(valid, key=valid.get)
        print(f"    {r['job']:<14} → {best} ({valid[best]:.2f}s)")

print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nNext step: python business_cost_analysis.py")