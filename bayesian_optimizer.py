# bayesian_optimizer.py
# Bayesian optimization using ML model as surrogate
# Compares: Grid Search vs Random Search vs Bayesian
# Save as: C:\Aravindh\bayesian_optimizer.py

import pandas as pd
import numpy as np
import joblib
import json
import time
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

MODEL_PATH    = "C:/Aravindh/models/spark_optimizer.pkl"
FEATURES_PATH = "C:/Aravindh/models/feature_columns.json"
OUT_DIR       = "C:/Aravindh/shap_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

model           = joblib.load(MODEL_PATH)
feature_columns = json.load(open(FEATURES_PATH))

print("=" * 65)
print("  BAYESIAN OPTIMIZATION — Config Search Comparison")
print("  Grid Search vs Random Search vs Bayesian")
print("=" * 65)

# ── Define search space ───────────────────────────────────────────
SHUFFLE_OPTIONS  = [4, 8, 16, 32, 64]
MEMORY_OPTIONS   = [2.0, 3.0]
CORES_OPTIONS    = [2, 4]

# Total grid points
TOTAL_GRID = len(SHUFFLE_OPTIONS) * len(MEMORY_OPTIONS) * len(CORES_OPTIONS)
print(f"\n  Config space: {TOTAL_GRID} total combinations")

def predict_execution_time(job_type, dataset_path, shuffle, memory_gb, cores):
    """Use trained ML model to predict execution time"""
    from feature_builder import build_feature_vector
    import sys
    sys.path.insert(0, "C:/Aravindh")

    feat = build_feature_vector(
        get_sql(job_type), dataset_path
    )
    feat["job_type"]           = job_type
    feat["shuffle_partitions"] = shuffle
    feat["executor_memory_gb"] = memory_gb
    feat["executor_cores"]     = cores

    df = pd.DataFrame([feat])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return float(np.expm1(model.predict(df)[0]))

def get_sql(job_type):
    sqls = {
        "join":
            "SELECT category, account_type, country, COUNT(*), SUM(amount), "
            "AVG(amount), SUM(is_fraud), AVG(credit_score) "
            "FROM transactions JOIN accounts ON account_id "
            "JOIN merchants ON merchant_id GROUP BY category, account_type, country",
        "aggregation":
            "SELECT merchant_id, COUNT(*), SUM(amount), AVG(amount), "
            "STDDEV(amount), SUM(is_fraud), COUNTDISTINCT(account_id) "
            "FROM transactions GROUP BY merchant_id",
        "filter":
            "SELECT location, channel, COUNT(*), SUM(amount), SUM(is_fraud) "
            "FROM transactions WHERE amount > 1000 AND is_fraud = 1 "
            "GROUP BY location, channel",
        "window":
            "SELECT account_id, RANK() OVER(PARTITION BY account_id ORDER BY amount), "
            "SUM(amount) OVER(PARTITION BY merchant_id ORDER BY timestamp) "
            "FROM transactions",
    }
    return sqls.get(job_type, "")

TEST_JOBS = [
    {"job_type": "join",        "dataset": "C:/Aravindh/data/txn_5m.csv"},
    {"job_type": "aggregation", "dataset": "C:/Aravindh/data/txn_5m.csv"},
    {"job_type": "filter",      "dataset": "C:/Aravindh/data/txn_5m.csv"},
    {"job_type": "window",      "dataset": "C:/Aravindh/data/txn_5m.csv"},
]

all_results = []

for job_info in TEST_JOBS:
    job_type    = job_info["job_type"]
    dataset     = job_info["dataset"]

    print(f"\n{'─'*65}")
    print(f"  Job: {job_type.upper()}")
    print(f"{'─'*65}")

    # ── 1. Grid Search (exhaustive) ───────────────────────────────
    t0         = time.time()
    grid_times = []
    grid_best  = float("inf")
    grid_cfg   = None

    for shuffle in SHUFFLE_OPTIONS:
        for memory in MEMORY_OPTIONS:
            for cores in CORES_OPTIONS:
                t = predict_execution_time(
                    job_type, dataset, shuffle, memory, cores
                )
                grid_times.append(t)
                if t < grid_best:
                    grid_best = t
                    grid_cfg  = (shuffle, memory, cores)

    grid_time = time.time() - t0
    print(f"  Grid Search:   best={grid_best:.2f}s  "
          f"cfg={grid_cfg}  evals={TOTAL_GRID}  time={grid_time:.3f}s")

    # ── 2. Random Search (10 random samples) ─────────────────────
    t0           = time.time()
    N_RANDOM     = 10
    random_best  = float("inf")
    random_cfg   = None
    np.random.seed(42)

    for _ in range(N_RANDOM):
        shuffle = np.random.choice(SHUFFLE_OPTIONS)
        memory  = np.random.choice(MEMORY_OPTIONS)
        cores   = np.random.choice(CORES_OPTIONS)
        t = predict_execution_time(
            job_type, dataset, float(shuffle), float(memory), int(cores)
        )
        if t < random_best:
            random_best = t
            random_cfg  = (shuffle, memory, cores)

    random_time = time.time() - t0
    print(f"  Random Search: best={random_best:.2f}s  "
          f"cfg={random_cfg}  evals={N_RANDOM}  time={random_time:.3f}s")

    # ── 3. Bayesian Optimization ──────────────────────────────────
    t0 = time.time()

    space = [
        Integer(0, len(SHUFFLE_OPTIONS)-1, name="shuffle_idx"),
        Integer(0, len(MEMORY_OPTIONS)-1,  name="memory_idx"),
        Integer(0, len(CORES_OPTIONS)-1,   name="cores_idx"),
    ]

    bayes_history = []

    def objective(params):
        shuffle_idx, memory_idx, cores_idx = params
        shuffle = SHUFFLE_OPTIONS[shuffle_idx]
        memory  = MEMORY_OPTIONS[memory_idx]
        cores   = CORES_OPTIONS[cores_idx]
        t = predict_execution_time(
            job_type, dataset, shuffle, memory, cores
        )
        bayes_history.append(t)
        return t

    result = gp_minimize(
        objective,
        space,
        n_calls=12,
        n_initial_points=4,
        random_state=42,
        verbose=False
    )

    bayes_time = time.time() - t0
    best_idx   = result.x
    bayes_best = result.fun
    bayes_cfg  = (
        SHUFFLE_OPTIONS[best_idx[0]],
        MEMORY_OPTIONS[best_idx[1]],
        CORES_OPTIONS[best_idx[2]]
    )

    print(f"  Bayesian Opt:  best={bayes_best:.2f}s  "
          f"cfg={bayes_cfg}  evals=12  time={bayes_time:.3f}s")

    # Efficiency
    random_gap = ((random_best - grid_best) / grid_best * 100)
    bayes_gap  = ((bayes_best  - grid_best) / grid_best * 100)

    print(f"\n  Optimality gap vs exhaustive:")
    print(f"    Random Search: {random_gap:+.1f}%")
    print(f"    Bayesian:      {bayes_gap:+.1f}%")

    all_results.append({
        "job_type":    job_type,
        "grid_best":   round(grid_best, 3),
        "grid_cfg":    str(grid_cfg),
        "random_best": round(random_best, 3),
        "random_cfg":  str(random_cfg),
        "bayes_best":  round(bayes_best, 3),
        "bayes_cfg":   str(bayes_cfg),
        "bayes_history": bayes_history,
    })

# ── Convergence plot ──────────────────────────────────────────────
print("\n  Generating convergence plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors_map = {"join": "#2E75B6", "aggregation": "#70AD47",
              "filter": "#ED7D31", "window": "#7030A0"}

for idx, res in enumerate(all_results):
    ax          = axes[idx]
    job_type    = res["job_type"]
    history     = res["bayes_history"]
    running_min = [min(history[:i+1]) for i in range(len(history))]

    ax.plot(range(1, len(running_min)+1), running_min,
            "o-", color=colors_map[job_type],
            linewidth=2, markersize=5, label="Bayesian")
    ax.axhline(y=res["grid_best"], color="red",
               linestyle="--", alpha=0.7, label=f"Grid best ({res['grid_best']:.1f}s)")
    ax.axhline(y=res["random_best"], color="orange",
               linestyle=":", alpha=0.7, label=f"Random best ({res['random_best']:.1f}s)")

    ax.set_title(f"{job_type.upper()} — Bayesian Convergence",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Evaluations", fontsize=9)
    ax.set_ylabel("Best Time (s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, len(running_min)+0.5)

plt.suptitle("Bayesian vs Grid vs Random Search Convergence\n(Using ML Model as Surrogate)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR + "bayesian_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: bayesian_convergence.png")

# ── Summary table ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  COMPARISON SUMMARY")
print(f"{'='*65}")
print(f"  {'Job':<14} {'Grid':>8} {'Random':>8} {'Bayesian':>10} {'Bayes Gap':>10}")
print(f"  {'─'*54}")

for r in all_results:
    gap = (r["bayes_best"] - r["grid_best"]) / r["grid_best"] * 100
    print(f"  {r['job_type']:<14} "
          f"{r['grid_best']:>7.2f}s "
          f"{r['random_best']:>7.2f}s "
          f"{r['bayes_best']:>9.2f}s "
          f"{gap:>+9.1f}%")

print(f"\n  Key insight: Bayesian finds near-optimal in 12 evals")
print(f"  vs Grid Search exhaustive {TOTAL_GRID} evals")
print(f"  Efficiency gain: {(1 - 12/TOTAL_GRID)*100:.0f}% fewer evaluations")
print(f"\n✅ Plot saved: {OUT_DIR}bayesian_convergence.png")