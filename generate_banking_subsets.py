# generate_banking_subsets.py
# Creates smaller transaction subsets from the full 25m file
# Needed for multi-dataset experiments (like your original 100k→5m range)
#
# Run AFTER generate_banking_data.py
# Output: txn_500k, txn_1m, txn_5m, txn_10m, txn_25m (already exists)

import pandas as pd
import os
import time

IN_PATH = "C:/Aravindh/data/transactions_25m.csv"
OUT_DIR = "C:/Aravindh/data/"

SUBSETS = [
    ("txn_500k.csv",  500_000),
    ("txn_1m.csv",  1_000_000),
    ("txn_5m.csv",  5_000_000),
    ("txn_10m.csv",10_000_000),
]

print("=" * 55)
print("  BANKING SUBSET GENERATOR")
print("=" * 55)
print(f"\n  Source: {IN_PATH}")

for fname, nrows in SUBSETS:
    t0   = time.time()
    path = OUT_DIR + fname
    print(f"\n  Creating {fname} ({nrows:,} rows)...")

    df = pd.read_csv(IN_PATH, nrows=nrows)
    df.to_csv(path, index=False)

    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✅ {mb:.1f} MB saved in {time.time()-t0:.1f}s")

print("\n" + "=" * 55)
print("  ALL SUBSETS CREATED")
print("=" * 55)
for fname, nrows in SUBSETS:
    path = OUT_DIR + fname
    mb   = os.path.getsize(path) / 1024 / 1024
    print(f"  {fname:<20} {nrows:>12,} rows   {mb:>8.1f} MB")

print(f"\n✅ Ready for experiments!")
print("Next step: Run the 4 updated Spark job scripts")