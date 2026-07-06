# shap_analysis.py
# SHAP feature importance analysis — per job type
# Generates interpretability evidence for paper
# Save as: C:\Aravindh\shap_analysis.py

import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ML_DATA_PATH  = "C:/Aravindh/data/ml_dataset.csv"
MODEL_PATH    = "C:/Aravindh/models/spark_optimizer.pkl"
FEATURES_PATH = "C:/Aravindh/models/feature_columns.json"
OUT_DIR       = "C:/Aravindh/shap_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

model           = joblib.load(MODEL_PATH)
feature_columns = json.load(open(FEATURES_PATH))

print("=" * 65)
print("  SHAP FEATURE IMPORTANCE ANALYSIS")
print("  Per job type + global summary")
print("=" * 65)

df = pd.read_csv(ML_DATA_PATH)
df = df[df["execution_time_sec"] > 0].reset_index(drop=True)

X_full = df.drop(columns=["execution_time_sec"])
X_full = pd.get_dummies(X_full)
X_full = X_full.reindex(columns=feature_columns, fill_value=0)

# ── Global SHAP summary ───────────────────────────────────────────
print("\n[1/5] Computing global SHAP values...")
explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_full)

# Summary plot — all jobs
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_full,
    plot_type="bar",
    show=False,
    max_display=15
)
plt.title("Global Feature Importance (SHAP) — All Job Types", 
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR + "shap_global_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: shap_global_bar.png")

# Beeswarm plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_full, show=False, max_display=15)
plt.title("SHAP Beeswarm — Feature Impact Distribution", 
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR + "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: shap_global_beeswarm.png")

# ── Per-job-type SHAP ─────────────────────────────────────────────
job_types = ["join", "aggregation", "filter", "window"]
colors    = {"join": "#2E75B6", "aggregation": "#70AD47",
             "filter": "#ED7D31", "window": "#7030A0"}

print("\n[2/5] Per-job-type SHAP analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

job_shap_summary = {}

for idx, job_type in enumerate(job_types):
    subset = df[df["job_type"] == job_type].copy()
    if len(subset) < 5:
        continue

    X_job = subset.drop(columns=["execution_time_sec"])
    X_job = pd.get_dummies(X_job)
    X_job = X_job.reindex(columns=feature_columns, fill_value=0)

    sv_job = explainer.shap_values(X_job)

    # Mean absolute SHAP per feature
    mean_shap = pd.Series(
        np.abs(sv_job).mean(axis=0),
        index=feature_columns
    ).sort_values(ascending=False).head(8)

    job_shap_summary[job_type] = mean_shap.to_dict()

    ax = axes[idx]
    bars = ax.barh(
        mean_shap.index[::-1],
        mean_shap.values[::-1],
        color=colors.get(job_type, "#999999"),
        alpha=0.85
    )
    ax.set_title(f"{job_type.upper()} Job — Top Features (SHAP)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    # Value labels
    for bar, val in zip(bars, mean_shap.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)

plt.suptitle("Per-Job-Type SHAP Feature Importance — SparkOptimizer-ML",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR + "shap_per_job_type.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: shap_per_job_type.png")

# ── Cross-job comparison heatmap ──────────────────────────────────
print("\n[3/5] SHAP cross-job heatmap...")

top_features = list(pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_columns
).sort_values(ascending=False).head(10).index)

heatmap_data = {}
for job_type in job_types:
    subset = df[df["job_type"] == job_type]
    X_job  = subset.drop(columns=["execution_time_sec"])
    X_job  = pd.get_dummies(X_job)
    X_job  = X_job.reindex(columns=feature_columns, fill_value=0)
    sv     = explainer.shap_values(X_job)
    mean_s = pd.Series(np.abs(sv).mean(axis=0), index=feature_columns)
    heatmap_data[job_type] = mean_s[top_features]

heatmap_df = pd.DataFrame(heatmap_data).T

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(heatmap_df.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(job_types)))
ax.set_yticklabels([j.upper() for j in job_types], fontsize=10)

for i in range(len(job_types)):
    for j in range(len(top_features)):
        ax.text(j, i, f"{heatmap_df.values[i,j]:.3f}",
                ha="center", va="center", fontsize=7,
                color="black" if heatmap_df.values[i,j] < 0.1 else "white")

plt.colorbar(im, ax=ax, label="Mean |SHAP value|")
ax.set_title("SHAP Feature Importance Heatmap by Job Type",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR + "shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: shap_heatmap.png")

# ── Predicted vs Actual scatter ───────────────────────────────────
print("\n[4/5] Predicted vs Actual plot...")
preds_log = model.predict(X_full)
preds     = np.expm1(preds_log)
actual    = df["execution_time_sec"].values

fig, ax = plt.subplots(figsize=(8, 6))
job_list = df["job_type"].values

for job_type in job_types:
    mask = job_list == job_type
    ax.scatter(actual[mask], preds[mask],
               label=job_type.upper(), alpha=0.6,
               color=colors[job_type], s=30)

max_val = max(actual.max(), preds.max())
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect prediction")
ax.set_xlabel("Actual Execution Time (s)", fontsize=11)
ax.set_ylabel("Predicted Execution Time (s)", fontsize=11)
ax.set_title("Predicted vs Actual Execution Time\n(XGBoost, log-transformed target)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

from sklearn.metrics import r2_score
r2 = r2_score(actual, preds)
ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig(OUT_DIR + "predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: predicted_vs_actual.png")

# ── Text summary for paper ────────────────────────────────────────
print("\n[5/5] SHAP text summary for paper...")
print(f"\n{'='*65}")
print("  KEY SHAP FINDINGS FOR PAPER")
print(f"{'='*65}")

for job_type, feat_dict in job_shap_summary.items():
    top3 = list(feat_dict.items())[:3]
    print(f"\n  {job_type.upper()}:")
    for feat, val in top3:
        print(f"    {feat:<30} SHAP={val:.4f}")

print(f"\n  Global top feature: {feature_columns[np.abs(shap_values).mean(axis=0).argmax()]}")
print(f"\n✅ All plots saved to: {OUT_DIR}")
print("\nFiles to include in paper:")
print("  shap_global_bar.png      → Figure 2")
print("  shap_per_job_type.png    → Figure 3")
print("  shap_heatmap.png         → Figure 4")
print("  predicted_vs_actual.png  → Figure 5")