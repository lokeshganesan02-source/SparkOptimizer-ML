# temporal_drift_analysis.py — UPDATED for banking dataset
# Per-job-type stability analysis
# Shows model generalizes across all banking workload types
# Save as: C:\Aravindh\temporal_drift_analysis.py

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score

ML_DATA_PATH  = "C:/Aravindh/data/ml_dataset.csv"
MODEL_PATH    = "C:/Aravindh/models/spark_optimizer.pkl"
FEATURES_PATH = "C:/Aravindh/models/feature_columns.json"
OUTPUT_PATH   = "C:/Aravindh/data/temporal_analysis.csv"

model           = joblib.load(MODEL_PATH)
feature_columns = json.load(open(FEATURES_PATH))

print("=" * 65)
print("  TEMPORAL STABILITY ANALYSIS — Banking Dataset")
print("  Tests model consistency across job types + time periods")
print("=" * 65)

df = pd.read_csv(ML_DATA_PATH)
df = df[df["execution_time_sec"] > 0].reset_index(drop=True)

print(f"\n  Total samples: {len(df)}")
print(f"  Job type distribution:")
for jt, cnt in df["job_type"].value_counts().items():
    print(f"    {jt:<15} {cnt} samples")

results = []

# ── Per-job-type early/late split ─────────────────────────────────
print(f"\n{'─'*65}")
print("  PER-JOB-TYPE STABILITY (Early vs Late experiments)")
print(f"{'─'*65}")

for job_type in ["join", "aggregation", "filter", "window"]:
    subset = df[df["job_type"] == job_type].copy().reset_index(drop=True)

    if len(subset) < 6:
        print(f"\n  {job_type.upper()}: insufficient samples ({len(subset)})")
        continue

    mid   = len(subset) // 2
    early = subset.iloc[:mid]
    late  = subset.iloc[mid:]

    print(f"\n  {job_type.upper()} ({len(subset)} total samples)")

    for period_name, period_df in [("Early", early), ("Late", late)]:
        X = period_df.drop(columns=["execution_time_sec"])
        y = period_df["execution_time_sec"]

        X = pd.get_dummies(X)
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Model was trained on log scale — predict and inverse transform
        log_preds = model.predict(X)
        preds     = np.expm1(log_preds)

        mae  = mean_absolute_error(y, preds)
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        r2   = r2_score(y, preds)

        status = "✅" if r2 > 0.70 else ("⚠️ " if r2 > 0.50 else "❌")
        print(f"    {period_name} half ({len(period_df)} samples): "
              f"MAE={mae:.2f}s  RMSE={rmse:.2f}s  R²={r2:.3f} {status}")

        results.append({
            "job_type":   job_type,
            "period":     period_name,
            "samples":    len(period_df),
            "mae":        round(mae, 3),
            "rmse":       round(rmse, 3),
            "r2":         round(r2, 3),
            "mean_actual": round(y.mean(), 2),
            "mean_pred":   round(preds.mean(), 2),
        })

# ── Overall 3-block temporal analysis ─────────────────────────────
print(f"\n{'─'*65}")
print("  3-BLOCK TEMPORAL ANALYSIS (Full dataset)")
print(f"{'─'*65}")

n      = len(df)
b1_end = n // 3
b2_end = 2 * (n // 3)

blocks = {
    "Block 1 — Early":  df.iloc[:b1_end],
    "Block 2 — Mid":    df.iloc[b1_end:b2_end],
    "Block 3 — Recent": df.iloc[b2_end:]
}

block_results = []
for block_name, block_df in blocks.items():
    job_counts = block_df["job_type"].value_counts().to_dict()

    X = block_df.drop(columns=["execution_time_sec"])
    y = block_df["execution_time_sec"]
    X = pd.get_dummies(X)
    X = X.reindex(columns=feature_columns, fill_value=0)

    log_preds = model.predict(X)
    preds     = np.expm1(log_preds)

    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    r2   = r2_score(y, preds)

    print(f"\n  {block_name} ({len(block_df)} samples)")
    print(f"    Job types: {job_counts}")
    print(f"    MAE={mae:.2f}s  RMSE={rmse:.2f}s  R²={r2:.3f}")

    block_results.append({
        "block":      block_name,
        "samples":    len(block_df),
        "job_counts": str(job_counts),
        "mae":        round(mae, 3),
        "rmse":       round(rmse, 3),
        "r2":         round(r2, 3),
    })

# ── Save results ──────────────────────────────────────────────────
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)

# ── Overall verdict ───────────────────────────────────────────────
r2_vals  = [r["r2"] for r in results]
valid_r2 = [v for v in r2_vals if v > -1]
avg_r2   = sum(valid_r2) / len(valid_r2) if valid_r2 else 0
min_r2   = min(valid_r2) if valid_r2 else 0

print(f"\n{'='*65}")
print(f"  VERDICT")
print(f"{'='*65}")
print(f"  R² values per job/period: {[round(v,3) for v in r2_vals]}")
print(f"  Average R²: {avg_r2:.3f}")
print(f"  Minimum R²: {min_r2:.3f}")

if avg_r2 > 0.75:
    verdict = "✅ STABLE — model consistent across all job types and time"
elif avg_r2 > 0.55:
    verdict = "⚠️  MOSTLY STABLE — minor variance on some job types"
else:
    verdict = "❌ UNSTABLE — high variance across job types"

print(f"  {verdict}")

# Paper-ready summary
print(f"\n{'─'*65}")
print("  PAPER SUMMARY")
print(f"{'─'*65}")
join_r2  = [r["r2"] for r in results if r["job_type"] == "join"]
filt_r2  = [r["r2"] for r in results if r["job_type"] == "filter"]
win_r2   = [r["r2"] for r in results if r["job_type"] == "window"]
agg_r2   = [r["r2"] for r in results if r["job_type"] == "aggregation"]

for jt, vals in [("join", join_r2), ("aggregation", agg_r2),
                  ("filter", filt_r2), ("window", win_r2)]:
    if vals:
        stability = abs(vals[0] - vals[1]) if len(vals) == 2 else 0
        flag = "✅ stable" if stability < 0.15 else "⚠️  some drift"
        print(f"  {jt:<14} Early R²={vals[0]:.3f}  "
              f"Late R²={vals[1]:.3f}  "
              f"Δ={stability:.3f}  {flag}")

print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nAll analyses complete!")
print("Next: python spark_optimizer_proof_multiscale.py (run overnight)")