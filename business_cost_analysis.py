# business_cost_analysis.py — UPDATED for banking dataset
# Calculates real business cost savings
# Save as: C:\Aravindh\business_cost_analysis.py

import pandas as pd
import os

PROOF_PATH  = "C:/Aravindh/data/optimization_proof.csv"
AQE_PATH    = "C:/Aravindh/data/aqe_comparison.csv"
OUTPUT_PATH = "C:/Aravindh/data/business_impact.csv"

# Enterprise assumptions
COST_PER_HOUR        = 0.10    # $0.10/hour compute
JOBS_PER_DAY         = 500     # enterprise batch jobs
WORKING_DAYS         = 22      # per month
ENGINEER_RATE        = 50      # $/hour
TUNING_HOURS         = 5       # hours/month manual tuning
FRAUD_COST_PER_MISS  = 150     # $ cost per missed fraud detection

print("=" * 65)
print("  BUSINESS COST IMPACT ANALYSIS — Banking Dataset")
print("=" * 65)

def seconds_to_cost(seconds):
    return (seconds / 3600) * COST_PER_HOUR

# ── Load optimizer proof ──────────────────────────────────────────
if not os.path.exists(PROOF_PATH):
    print("❌ Run spark_optimizer_proof.py first!")
    exit()

df = pd.read_csv(PROOF_PATH)
df = df[df["default_time_avg"] > 0]

print("\n📊 Per-Job Savings (txn_5m.csv):")
print("-" * 65)

total_default_cost    = 0
total_optimized_cost  = 0

for _, row in df.iterrows():
    dc  = seconds_to_cost(row["default_time_avg"])
    oc  = seconds_to_cost(row["ml_time_avg"])
    sav = dc - oc
    total_default_cost   += dc
    total_optimized_cost += oc

    arrow = "✅" if sav > 0 else "❌"
    print(f"  {row['job_name'].upper():<14} "
          f"Default: {row['default_time_avg']:>6.2f}s (${dc:.4f}) | "
          f"ML: {row['ml_time_avg']:>6.2f}s (${oc:.4f}) | "
          f"{row['improvement_pct']:>+6.1f}% {arrow}")

# ── Monthly calculations ──────────────────────────────────────────
monthly_jobs          = JOBS_PER_DAY * WORKING_DAYS
avg_default_sec       = df["default_time_avg"].mean()
avg_ml_sec            = df["ml_time_avg"].mean()
avg_improvement       = df["improvement_pct"].mean()

monthly_default_cost  = seconds_to_cost(avg_default_sec) * monthly_jobs
monthly_ml_cost       = seconds_to_cost(avg_ml_sec)      * monthly_jobs
monthly_compute_saved = monthly_default_cost - monthly_ml_cost
monthly_tuning_saved  = TUNING_HOURS * ENGINEER_RATE
total_monthly_savings = monthly_compute_saved + monthly_tuning_saved

# ── AQE comparison savings ────────────────────────────────────────
aqe_savings = None
if os.path.exists(AQE_PATH):
    aqe_df    = pd.read_csv(AQE_PATH)
    default_t = aqe_df[aqe_df["config"] == "Default"]["execution_time_sec"].mean()
    ml_t      = aqe_df[aqe_df["config"] == "ML_OPT"]["execution_time_sec"].mean()
    aqe_t     = aqe_df[aqe_df["config"] == "AQE_ON"]["execution_time_sec"].mean()

    if default_t > 0 and ml_t > 0 and aqe_t > 0:
        ml_vs_default = (default_t - ml_t) / default_t * 100
        ml_vs_aqe     = (aqe_t - ml_t) / aqe_t * 100
        aqe_savings   = {"ml_vs_default": ml_vs_default, "ml_vs_aqe": ml_vs_aqe}

print(f"\n{'='*65}")
print(f"  MONTHLY IMPACT ({monthly_jobs:,} jobs/month)")
print(f"{'='*65}")
print(f"  Avg runtime — Default:    {avg_default_sec:.2f}s")
print(f"  Avg runtime — ML Opt:     {avg_ml_sec:.2f}s")
print(f"  Avg improvement:          {avg_improvement:+.1f}%")
print(f"\n  Compute savings:          ${monthly_compute_saved:.2f}/month")
print(f"  Manual tuning savings:    ${monthly_tuning_saved:.2f}/month")
print(f"  {'─'*45}")
print(f"  TOTAL MONTHLY SAVINGS:    ${total_monthly_savings:.2f}/month")
print(f"  ANNUAL SAVINGS:           ${total_monthly_savings * 12:,.2f}/year")

if aqe_savings:
    print(f"\n  ML vs Default (AQE test): {aqe_savings['ml_vs_default']:+.1f}%")
    print(f"  ML vs AQE:                {aqe_savings['ml_vs_aqe']:+.1f}%")

# ── Fraud-specific impact ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FRAUD DETECTION PIPELINE IMPACT")
print(f"{'='*65}")
fraud_row = df[df["job_name"] == "filter"]
if len(fraud_row) > 0:
    fraud_default = fraud_row.iloc[0]["default_time_avg"]
    fraud_ml      = fraud_row.iloc[0]["ml_time_avg"]
    fraud_saved   = fraud_default - fraud_ml
    fraud_pct     = fraud_row.iloc[0]["improvement_pct"]

    # Faster fraud detection = catch fraud earlier
    # Every second saved = potentially catching fraud before more damage
    monthly_fraud_jobs    = monthly_jobs * 0.10   # 10% are fraud queries
    monthly_fraud_time_saved = fraud_saved * monthly_fraud_jobs
    print(f"  Fraud query speedup:      {fraud_pct:+.1f}%")
    print(f"  Time saved per query:     {fraud_saved:.2f}s")
    print(f"  Monthly fraud queries:    {monthly_fraud_jobs:,.0f}")
    print(f"  Total time saved:         {monthly_fraud_time_saved/3600:.1f} hours/month")
    print(f"  Faster fraud detection → reduced financial exposure")

# ── Window job impact (best result) ──────────────────────────────
window_row = df[df["job_name"] == "window"]
if len(window_row) > 0:
    w_pct = window_row.iloc[0]["improvement_pct"]
    w_def = window_row.iloc[0]["default_time_avg"]
    w_ml  = window_row.iloc[0]["ml_time_avg"]
    print(f"\n  Window (fraud ranking):   {w_def:.1f}s → {w_ml:.1f}s ({w_pct:+.1f}%)")
    print(f"  Use case: Real-time account risk ranking for fraud analysts")

# ── Save ──────────────────────────────────────────────────────────
report = {
    "avg_default_sec":       avg_default_sec,
    "avg_ml_sec":            avg_ml_sec,
    "avg_improvement_pct":   avg_improvement,
    "monthly_compute_saved": round(monthly_compute_saved, 2),
    "monthly_tuning_saved":  monthly_tuning_saved,
    "total_monthly_savings": round(total_monthly_savings, 2),
    "annual_savings":        round(total_monthly_savings * 12, 2),
}
if aqe_savings:
    report.update(aqe_savings)

pd.DataFrame([report]).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nNext step: python temporal_drift_analysis.py")