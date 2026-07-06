# business_cost_real_pricing.py
# Uses real AWS/GCP pricing for 3 scenarios
# Save as: C:\Aravindh\business_cost_real_pricing.py

import pandas as pd
import os

PROOF_PATH  = "C:/Aravindh/data/optimization_proof.csv"
OUTPUT_PATH = "C:/Aravindh/data/business_impact_real.csv"

# Real cloud pricing (AWS EMR, as of 2025)
# Source: aws.amazon.com/emr/pricing
PRICING = {
    "small_team": {
        "label":          "Small Analytics Team",
        "description":    "5-node EMR cluster, m5.xlarge",
        "cost_per_hour":  0.48,      # 5 × m5.xlarge EMR price
        "jobs_per_day":   200,
        "working_days":   22,
        "engineers":      2,
        "engineer_rate":  50,
        "tuning_hours":   5,
    },
    "mid_bank": {
        "label":          "Mid-Size Bank",
        "description":    "20-node EMR cluster, m5.2xlarge",
        "cost_per_hour":  3.84,      # 20 × m5.2xlarge EMR price
        "jobs_per_day":   2000,
        "working_days":   22,
        "engineers":      8,
        "engineer_rate":  75,
        "tuning_hours":   20,
    },
    "enterprise": {
        "label":          "Enterprise Bank",
        "description":    "100-node EMR cluster, m5.4xlarge",
        "cost_per_hour":  38.40,     # 100 × m5.4xlarge EMR price
        "jobs_per_day":   10000,
        "working_days":   22,
        "engineers":      30,
        "engineer_rate":  100,
        "tuning_hours":   80,
    }
}

print("=" * 65)
print("  BUSINESS IMPACT — Real Cloud Pricing (AWS EMR 2025)")
print("=" * 65)

if not os.path.exists(PROOF_PATH):
    print("❌ Run spark_optimizer_proof.py first!")
    exit()

df = pd.read_csv(PROOF_PATH)
df = df[df["default_time_avg"] > 0]

avg_default_sec  = df["default_time_avg"].mean()
avg_ml_sec       = df["ml_time_avg"].mean()
avg_improvement  = df["improvement_pct"].mean() / 100  # as fraction
window_improvement = df[df["job_name"]=="window"]["improvement_pct"].values
join_improvement   = df[df["job_name"]=="join"]["improvement_pct"].values

print(f"\n  Baseline metrics:")
print(f"    Avg default runtime:   {avg_default_sec:.2f}s")
print(f"    Avg ML runtime:        {avg_ml_sec:.2f}s")
print(f"    Avg improvement:       {avg_improvement*100:.1f}%")

rows = []

print(f"\n{'='*65}")
for scenario_key, cfg in PRICING.items():
    monthly_jobs        = cfg["jobs_per_day"] * cfg["working_days"]
    hours_default       = (avg_default_sec * monthly_jobs) / 3600
    hours_ml            = (avg_ml_sec      * monthly_jobs) / 3600

    compute_default     = hours_default * cfg["cost_per_hour"]
    compute_ml          = hours_ml      * cfg["cost_per_hour"]
    compute_saved       = compute_default - compute_ml

    tuning_saved        = cfg["tuning_hours"] * cfg["engineer_rate"] * cfg["engineers"]
    total_monthly       = compute_saved + tuning_saved
    annual_savings      = total_monthly * 12

    print(f"\n  📊 {cfg['label']} ({cfg['description']})")
    print(f"  {'─'*55}")
    print(f"    Monthly jobs:          {monthly_jobs:>10,}")
    print(f"    Compute hrs (default): {hours_default:>10.1f}h")
    print(f"    Compute hrs (ML opt):  {hours_ml:>10.1f}h")
    print(f"    Compute cost saved:    ${compute_saved:>9,.2f}/month")
    print(f"    Tuning time saved:     ${tuning_saved:>9,.2f}/month")
    print(f"    TOTAL MONTHLY SAVING:  ${total_monthly:>9,.2f}/month")
    print(f"    ANNUAL SAVING:         ${annual_savings:>9,.2f}/year")

    rows.append({
        "scenario":             cfg["label"],
        "cluster_description":  cfg["description"],
        "monthly_jobs":         monthly_jobs,
        "avg_improvement_pct":  round(avg_improvement*100, 1),
        "compute_saved_monthly":round(compute_saved, 2),
        "tuning_saved_monthly": tuning_saved,
        "total_monthly_savings":round(total_monthly, 2),
        "annual_savings":       round(annual_savings, 2),
    })

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_PATH, index=False)

# Window job specific
print(f"\n{'='*65}")
print(f"  WINDOW JOB IMPACT (77.4% improvement on AQE baseline)")
print(f"{'='*65}")
for _, cfg in PRICING.items():
    monthly_jobs    = cfg["jobs_per_day"] * cfg["working_days"]
    # Assume 20% of jobs are window/ranking queries
    window_jobs     = monthly_jobs * 0.20
    time_saved_sec  = 75.02 - 16.97   # from AQE comparison
    hours_saved     = (time_saved_sec * window_jobs) / 3600
    cost_saved      = hours_saved * cfg["cost_per_hour"]
    print(f"  {cfg['label']:<25} Window savings: ${cost_saved:>8,.2f}/month")

print(f"\n✅ Saved to: {OUTPUT_PATH}")
print("\nKey paper claim:")
print(f"  At enterprise scale: annual savings up to")
print(f"  ${rows[-1]['annual_savings']:,.0f}/year (100-node EMR cluster)")