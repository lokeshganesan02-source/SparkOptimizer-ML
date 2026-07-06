# generate_banking_data.py
# Generates 3 realistic banking tables:
#   transactions_25m.csv  — 25 million rows (~1.5 GB)
#   accounts_500k.csv     — 500,000 accounts
#   merchants_100k.csv    — 100,000 merchants
#
# Run: python generate_banking_data.py
# Time: ~3-5 minutes

import pandas as pd
import numpy as np
import os
import time

OUT_DIR = "C:/Aravindh/data/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_TRANSACTIONS = 25_000_000
N_ACCOUNTS     = 500_000
N_MERCHANTS    = 100_000
FRAUD_RATE     = 0.0017   # 0.17% — same as real ULB dataset

print("=" * 65)
print("  BANKING DATASET GENERATOR")
print(f"  Transactions : {N_TRANSACTIONS:,}")
print(f"  Accounts     : {N_ACCOUNTS:,}")
print(f"  Merchants    : {N_MERCHANTS:,}")
print(f"  Fraud rate   : {FRAUD_RATE*100:.2f}%")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: accounts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Generating accounts...")
t0 = time.time()

account_types = ["savings", "checking", "credit", "business", "premium"]
countries     = ["US", "UK", "IN", "DE", "FR", "CA", "AU", "SG", "JP", "BR"]
account_type_weights = [0.35, 0.30, 0.20, 0.10, 0.05]

accounts_df = pd.DataFrame({
    "account_id"    : [f"ACC{str(i).zfill(8)}" for i in range(N_ACCOUNTS)],
    "customer_name" : [f"Customer_{i}" for i in range(N_ACCOUNTS)],
    "account_type"  : np.random.choice(account_types, N_ACCOUNTS, p=account_type_weights),
    "balance"       : np.round(np.random.exponential(5000, N_ACCOUNTS), 2),
    "credit_score"  : np.random.randint(300, 851, N_ACCOUNTS),
    "country"       : np.random.choice(countries, N_ACCOUNTS),
    "age"           : np.random.randint(18, 80, N_ACCOUNTS),
    "is_high_risk"  : np.random.choice([0, 1], N_ACCOUNTS, p=[0.95, 0.05]),
})

path = OUT_DIR + "accounts_500k.csv"
accounts_df.to_csv(path, index=False)
print(f"   ✅ {N_ACCOUNTS:,} rows → {os.path.getsize(path)/1024/1024:.1f} MB ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: merchants
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Generating merchants...")
t0 = time.time()

merchant_categories = [
    "grocery", "restaurant", "electronics", "clothing", "fuel",
    "pharmacy", "travel", "entertainment", "online", "atm"
]
risk_levels = ["low", "medium", "high"]
risk_weights = [0.70, 0.20, 0.10]

merchants_df = pd.DataFrame({
    "merchant_id"   : [f"MER{str(i).zfill(7)}" for i in range(N_MERCHANTS)],
    "merchant_name" : [f"Merchant_{i}" for i in range(N_MERCHANTS)],
    "category"      : np.random.choice(merchant_categories, N_MERCHANTS),
    "risk_level"    : np.random.choice(risk_levels, N_MERCHANTS, p=risk_weights),
    "country"       : np.random.choice(countries, N_MERCHANTS),
    "avg_txn_value" : np.round(np.random.exponential(80, N_MERCHANTS), 2),
    "monthly_volume": np.random.randint(100, 50000, N_MERCHANTS),
})

path = OUT_DIR + "merchants_100k.csv"
merchants_df.to_csv(path, index=False)
print(f"   ✅ {N_MERCHANTS:,} rows → {os.path.getsize(path)/1024/1024:.1f} MB ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: transactions (25M rows — built in chunks to save RAM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Generating transactions (this takes ~3-4 minutes)...")

CHUNK_SIZE = 1_000_000
txn_path   = OUT_DIR + "transactions_25m.csv"
total_written = 0
t_start = time.time()

transaction_types = ["purchase", "transfer", "withdrawal", "deposit", "refund"]
channels          = ["online", "mobile", "pos", "atm", "branch"]
statuses          = ["completed", "pending", "failed", "flagged"]
type_weights      = [0.55, 0.20, 0.10, 0.10, 0.05]
channel_weights   = [0.35, 0.30, 0.20, 0.10, 0.05]
status_weights    = [0.88, 0.05, 0.04, 0.03]

# Base timestamp: 2 years of data
base_ts   = pd.Timestamp("2022-01-01")
end_ts    = pd.Timestamp("2023-12-31")
ts_range  = int((end_ts - base_ts).total_seconds())

with open(txn_path, "w") as f:
    # Header
    f.write("transaction_id,account_id,merchant_id,amount,transaction_type,"
            "timestamp,is_fraud,location,channel,status,hour_of_day,day_of_week\n")

    for chunk_idx in range(N_TRANSACTIONS // CHUNK_SIZE):
        n = CHUNK_SIZE

        # Fraud flag — 0.17% rate with skew toward high-risk merchants
        is_fraud = (np.random.random(n) < FRAUD_RATE).astype(int)

        # Fraud transactions have higher amounts
        base_amount = np.random.exponential(150, n)
        fraud_boost = is_fraud * np.random.uniform(500, 5000, n)
        amounts     = np.round(base_amount + fraud_boost, 2)

        # Timestamps spread across 2 years
        seconds_offset = np.random.randint(0, ts_range, n)
        timestamps     = [str(base_ts + pd.Timedelta(seconds=int(s)))[:19]
                          for s in seconds_offset]

        # Hour and day features (useful for window functions)
        ts_arr      = pd.to_datetime(timestamps)
        hour_of_day = ts_arr.hour
        day_of_week = ts_arr.dayofweek

        chunk = pd.DataFrame({
            "transaction_id"  : [f"TXN{str(chunk_idx*n + i).zfill(10)}" for i in range(n)],
            "account_id"      : [f"ACC{str(np.random.randint(0, N_ACCOUNTS)).zfill(8)}" for _ in range(n)],
            "merchant_id"     : [f"MER{str(np.random.randint(0, N_MERCHANTS)).zfill(7)}" for _ in range(n)],
            "amount"          : amounts,
            "transaction_type": np.random.choice(transaction_types, n, p=type_weights),
            "timestamp"       : timestamps,
            "is_fraud"        : is_fraud,
            "location"        : np.random.choice(countries, n),
            "channel"         : np.random.choice(channels, n, p=channel_weights),
            "status"          : np.random.choice(statuses, n, p=status_weights),
            "hour_of_day"     : hour_of_day,
            "day_of_week"     : day_of_week,
        })

        chunk.to_csv(f, index=False, header=False)
        total_written += n

        elapsed  = time.time() - t_start
        progress = total_written / N_TRANSACTIONS * 100
        rate     = total_written / elapsed
        eta      = (N_TRANSACTIONS - total_written) / rate
        print(f"   {progress:5.1f}% | {total_written:>12,} rows | "
              f"{elapsed:5.1f}s elapsed | ETA: {eta:.0f}s")

size_mb = os.path.getsize(txn_path) / 1024 / 1024
print(f"\n   ✅ {N_TRANSACTIONS:,} rows → {size_mb:.1f} MB ({time.time()-t_start:.1f}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DATASET SUMMARY")
print("=" * 65)
for fname in ["transactions_25m.csv", "accounts_500k.csv", "merchants_100k.csv"]:
    fpath = OUT_DIR + fname
    mb    = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {fname:<30} {mb:>8.1f} MB")

total_mb = sum(
    os.path.getsize(OUT_DIR + f) / 1024 / 1024
    for f in ["transactions_25m.csv", "accounts_500k.csv", "merchants_100k.csv"]
)
print(f"  {'TOTAL':30} {total_mb:>8.1f} MB")
print(f"\n  Fraud transactions: ~{int(N_TRANSACTIONS * FRAUD_RATE):,} ({FRAUD_RATE*100:.2f}%)")
print(f"\n✅ All files saved to: {OUT_DIR}")
print("\nNext step: Run generate_banking_subsets.py to create size variants")