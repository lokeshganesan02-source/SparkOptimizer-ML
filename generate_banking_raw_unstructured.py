# generate_banking_raw_unstructured.py
#
# Generates 3 banking tables as RAW / UNSTRUCTURED files — ready for Spark ingestion.
#
#   transactions_25m.jsonl   — 25 M rows, JSON Lines (one JSON object per line)
#   accounts_500k.pipe       — 500 K rows, pipe-delimited flat file, no header, dirty whitespace
#   merchants_100k.log       — 100 K rows, raw application-log format (key=value pairs)
#
# Spark will need to:
#   • Parse JSON Lines          → spark.read.json(...)
#   • Split pipe-delimited text → spark.read.option("delimiter", "|").csv(...) or manual split
#   • Regex-extract log fields  → UDFs / regexp_extract
#
# Run : python generate_banking_raw_unstructured.py
# Time: ~4-6 minutes

import os
import time
import json
import random
import numpy as np

OUT_DIR = "C:/Aravindh/data/raw/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_TRANSACTIONS = 25_000_000
N_ACCOUNTS     = 500_000
N_MERCHANTS    = 100_000
FRAUD_RATE     = 0.0017
CHUNK_SIZE     = 500_000

print("=" * 65)
print("  RAW UNSTRUCTURED BANKING DATA GENERATOR (Spark target)")
print(f"  transactions_25m.jsonl  — JSON Lines")
print(f"  accounts_500k.pipe      — Pipe-delimited, no header, dirty")
print(f"  merchants_100k.log      — App-log key=value format")
print("=" * 65)

ACCOUNT_TYPES   = ["savings", "checking", "credit", "business", "premium"]
COUNTRIES       = ["US", "UK", "IN", "DE", "FR", "CA", "AU", "SG", "JP", "BR"]
MERCHANT_CATS   = ["grocery","restaurant","electronics","clothing","fuel",
                   "pharmacy","travel","entertainment","online","atm"]
RISK_LEVELS     = ["low", "medium", "high"]
TXN_TYPES       = ["purchase","transfer","withdrawal","deposit","refund"]
CHANNELS        = ["online","mobile","pos","atm","branch"]
STATUSES        = ["completed","pending","failed","flagged"]

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 — accounts_500k.pipe
# Format (pipe-separated, NO header, intentional dirty whitespace & mixed case):
#   account_id | customer_name | account_type | balance | credit_score |
#   country | age | is_high_risk
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Writing accounts_500k.pipe  (pipe-delimited, no header)...")
t0 = time.time()

acct_type_weights = [0.35, 0.30, 0.20, 0.10, 0.05]
risk_weights_acct = [0.95, 0.05]

path_acct = OUT_DIR + "accounts_500k.pipe"
with open(path_acct, "w", encoding="utf-8") as f:
    for i in range(N_ACCOUNTS):
        acct_id    = f"ACC{str(i).zfill(8)}"
        name       = f"  Customer_{i}  "          # dirty leading/trailing spaces
        acct_type  = np.random.choice(ACCOUNT_TYPES, p=acct_type_weights)
        balance    = round(np.random.exponential(5000), 2)
        credit     = np.random.randint(300, 851)
        country    = random.choice(COUNTRIES)
        age        = np.random.randint(18, 80)
        high_risk  = np.random.choice([0, 1], p=risk_weights_acct)

        # Randomly uppercase / lowercase account_type to add dirtiness
        if random.random() < 0.15:
            acct_type = acct_type.upper()

        line = f"{acct_id}|{name}|{acct_type}|{balance}|{credit}|{country}|{age}|{high_risk}\n"
        f.write(line)

size_mb = os.path.getsize(path_acct) / 1024 / 1024
print(f"   ✅ {N_ACCOUNTS:,} rows → {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")
print(f"   Sample line: ACC00000000|  Customer_0  |savings|3241.87|712|US|34|0")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2 — merchants_100k.log
# Format: raw application-log style, one merchant per line
#   2024-01-15T08:23:11 [MERCHANT_REGISTRY] id=MER0000001 name="Merchant_1"
#   category=grocery risk=low country=US avg_txn=82.30 monthly_vol=14220
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Writing merchants_100k.log  (raw log key=value format)...")
t0 = time.time()

risk_weights_mer = [0.70, 0.20, 0.10]
path_mer = OUT_DIR + "merchants_100k.log"

import datetime
base_log_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)

with open(path_mer, "w", encoding="utf-8") as f:
    for i in range(N_MERCHANTS):
        ts        = base_log_dt + datetime.timedelta(seconds=i * 3)
        ts_str    = ts.strftime("%Y-%m-%dT%H:%M:%S")
        mer_id    = f"MER{str(i).zfill(7)}"
        name      = f"Merchant_{i}"
        category  = random.choice(MERCHANT_CATS)
        risk      = np.random.choice(RISK_LEVELS, p=risk_weights_mer)
        country   = random.choice(COUNTRIES)
        avg_txn   = round(np.random.exponential(80), 2)
        monthly   = np.random.randint(100, 50000)

        # Occasionally inject a WARN prefix to make it dirtier
        level = "WARN " if random.random() < 0.05 else "INFO "

        line = (f"{ts_str} [{level}MERCHANT_REGISTRY] "
                f"id={mer_id} name=\"{name}\" category={category} "
                f"risk={risk} country={country} "
                f"avg_txn={avg_txn} monthly_vol={monthly}\n")
        f.write(line)

size_mb = os.path.getsize(path_mer) / 1024 / 1024
print(f"   ✅ {N_MERCHANTS:,} rows → {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")
print(f"   Sample line: 2024-01-01T00:00:00 [INFO  MERCHANT_REGISTRY] id=MER0000000 name=\"Merchant_0\" ...")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — transactions_25m.jsonl
# Format: JSON Lines — one raw JSON object per line, NO schema enforcement.
#   Fields vary slightly (some rows have extra noise keys, some missing optional keys)
#   to simulate real unstructured ingestion.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Writing transactions_25m.jsonl  (JSON Lines — ~3-4 min)...")

type_weights    = [0.55, 0.20, 0.10, 0.10, 0.05]
channel_weights = [0.35, 0.30, 0.20, 0.10, 0.05]
status_weights  = [0.88, 0.05, 0.04, 0.03]

base_ts  = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_ts   = datetime.datetime(2023, 12, 31, 23, 59, 59)
ts_range = int((end_ts - base_ts).total_seconds())

path_txn      = OUT_DIR + "transactions_25m.jsonl"
total_written = 0
t_start       = time.time()

with open(path_txn, "w", encoding="utf-8") as f:
    for chunk_idx in range(N_TRANSACTIONS // CHUNK_SIZE):
        n = CHUNK_SIZE

        is_fraud    = (np.random.random(n) < FRAUD_RATE).astype(int)
        base_amount = np.random.exponential(150, n)
        fraud_boost = is_fraud * np.random.uniform(500, 5000, n)
        amounts     = np.round(base_amount + fraud_boost, 2)

        seconds_off = np.random.randint(0, ts_range, n)
        txn_types   = np.random.choice(TXN_TYPES,   n, p=type_weights)
        channels    = np.random.choice(CHANNELS,    n, p=channel_weights)
        statuses    = np.random.choice(STATUSES,    n, p=status_weights)
        locations   = np.random.choice(COUNTRIES,   n)
        acct_ids    = np.random.randint(0, N_ACCOUNTS,  n)
        mer_ids     = np.random.randint(0, N_MERCHANTS, n)

        for i in range(n):
            global_i = chunk_idx * n + i
            ts_obj   = base_ts + datetime.timedelta(seconds=int(seconds_off[i]))

            record = {
                "transaction_id"  : f"TXN{str(global_i).zfill(10)}",
                "account_id"      : f"ACC{str(acct_ids[i]).zfill(8)}",
                "merchant_id"     : f"MER{str(mer_ids[i]).zfill(7)}",
                "amount"          : float(amounts[i]),
                "transaction_type": txn_types[i],
                "timestamp"       : ts_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "is_fraud"        : int(is_fraud[i]),
                "location"        : locations[i],
                "channel"         : channels[i],
                "status"          : statuses[i],
                "hour_of_day"     : ts_obj.hour,
                "day_of_week"     : ts_obj.weekday(),
            }

            # ~3% of records: inject a random noise key (simulates schema drift)
            if random.random() < 0.03:
                record["_raw_device_id"] = f"DEV-{random.randint(1000,9999)}"

            # ~1% of records: omit status (simulate missing field)
            if random.random() < 0.01:
                del record["status"]

            f.write(json.dumps(record) + "\n")

        total_written += n
        elapsed  = time.time() - t_start
        progress = total_written / N_TRANSACTIONS * 100
        rate     = total_written / elapsed
        eta      = (N_TRANSACTIONS - total_written) / rate
        print(f"   {progress:5.1f}% | {total_written:>13,} rows | "
              f"{elapsed:5.1f}s elapsed | ETA {eta:.0f}s")

size_mb = os.path.getsize(path_txn) / 1024 / 1024
print(f"\n   ✅ {N_TRANSACTIONS:,} rows → {size_mb:.1f} MB  ({time.time()-t_start:.1f}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  OUTPUT SUMMARY")
print("=" * 65)
files = [
    ("transactions_25m.jsonl", "JSON Lines"),
    ("accounts_500k.pipe",     "Pipe-delimited, no header"),
    ("merchants_100k.log",     "App-log key=value"),
]
total_mb = 0
for fname, fmt in files:
    fpath = OUT_DIR + fname
    mb    = os.path.getsize(fpath) / 1024 / 1024
    total_mb += mb
    print(f"  {fname:<28} {mb:>8.1f} MB   ({fmt})")
print(f"  {'TOTAL':28} {total_mb:>8.1f} MB")

print("""
  HOW TO READ WITH SPARK
  ──────────────────────
  # JSON Lines
  txn_df = spark.read.json("transactions_25m.jsonl")

  # Pipe-delimited (no header)
  acct_df = (spark.read
               .option("delimiter", "|")
               .option("header", "false")
               .csv("accounts_500k.pipe")
               .toDF("account_id","customer_name","account_type",
                     "balance","credit_score","country","age","is_high_risk"))
  # Trim dirty whitespace
  from pyspark.sql.functions import trim
  acct_df = acct_df.withColumn("customer_name", trim(acct_df.customer_name))

  # App-log key=value
  from pyspark.sql.functions import regexp_extract
  raw_mer = spark.read.text("merchants_100k.log")
  mer_df  = (raw_mer
               .withColumn("id",          regexp_extract("value", r"id=(\\S+)",        1))
               .withColumn("name",        regexp_extract("value", r'name="([^"]+)"',   1))
               .withColumn("category",    regexp_extract("value", r"category=(\\S+)",  1))
               .withColumn("risk",        regexp_extract("value", r"risk=(\\S+)",       1))
               .withColumn("country",     regexp_extract("value", r"country=(\\S+)",   1))
               .withColumn("avg_txn",     regexp_extract("value", r"avg_txn=(\\S+)",   1))
               .withColumn("monthly_vol", regexp_extract("value", r"monthly_vol=(\\S+)",1)))
""")

print(f"✅  All raw files saved to: {OUT_DIR}")
print("    Next step: load into Spark and run your transformations / fraud ML pipeline.")