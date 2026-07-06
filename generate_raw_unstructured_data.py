# generate_raw_unstructured_data.py
# Creates realistic UNSTRUCTURED banking data files
# Shows what data looks like BEFORE Spark processing
# Perfect for video demo — contrast raw vs clean
# Save as: C:\Aravindh\generate_raw_unstructured_data.py

import pandas as pd
import numpy as np
import json
import csv
import os
import random
import string
from datetime import datetime, timedelta

OUT_DIR = "C:/Aravindh/data/raw_unstructured/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

print("=" * 65)
print("  GENERATING UNSTRUCTURED RAW BANKING DATA FILES")
print("  Shows real-world data chaos before Spark processing")
print("=" * 65)

# ── 1. ATM Raw Log File (.log) ────────────────────────────────────
# Simulates raw ATM machine logs — completely unstructured text
print("\n[1/6] ATM raw log file...")

atm_locations = ["Chennai_Anna_Nagar", "Mumbai_Bandra", "Delhi_CP",
                 "Bangalore_Koramangala", "Hyderabad_Jubilee"]
atm_errors    = ["TIMEOUT", "CARD_READ_ERROR", "NETWORK_FAIL",
                 "CASH_DISPENSED", "TXN_APPROVED", "PIN_MISMATCH"]

with open(OUT_DIR + "atm_raw_logs.log", "w") as f:
    f.write("=== ATM TRANSACTION LOG DUMP ===\n")
    f.write(f"EXPORT_TIME: {datetime.now()}\n")
    f.write("SOURCE: ATM_NETWORK_v2.3.1\n\n")

    for i in range(200):
        ts       = datetime(2024, random.randint(1,12),
                            random.randint(1,28),
                            random.randint(0,23),
                            random.randint(0,59))
        atm_id   = f"ATM{random.randint(1000,9999)}"
        loc      = random.choice(atm_locations)
        amount   = random.choice([500, 1000, 2000, 5000, 10000])
        status   = random.choice(atm_errors)
        card_num = f"****-****-****-{random.randint(1000,9999)}"
        noise    = "".join(random.choices(string.ascii_uppercase, k=4))

        # Real ATM logs are messy — mixed formats, noise bytes
        if i % 15 == 0:
            f.write(f"[HEARTBEAT] {atm_id} OK ts={ts.timestamp()}\n")
        elif i % 20 == 0:
            f.write(f"##CORRUPT_RECORD## {noise} {ts}\n")
        elif i % 25 == 0:
            f.write(f"  \n")  # blank line noise
        else:
            f.write(f"[{ts.strftime('%Y%m%d%H%M%S')}] "
                    f"ATM={atm_id} "
                    f"LOC={loc} "
                    f"CARD={card_num} "
                    f"AMT={amount} "
                    f"STATUS={status} "
                    f"SEQ={i:06d}\n")

print(f"  ✅ atm_raw_logs.log")

# ── 2. GPay / UPI JSON Dump ───────────────────────────────────────
# Simulates raw mobile payment API responses — nested JSON
print("[2/6] GPay/UPI JSON dump...")

records = []
for i in range(300):
    ts = datetime(2024, random.randint(1,12),
                  random.randint(1,28),
                  random.randint(0,23),
                  random.randint(0,59))

    record = {
        "event_id":       f"EVT{i:08d}",
        "event_type":     random.choice(["UPI_TXN", "GPAY_TXN",
                                          "PHONEPE_TXN", "PAYTM_TXN"]),
        "timestamp_ms":   int(ts.timestamp() * 1000),
        "ts_human":       ts.isoformat(),
        "payload": {
            "from_vpa":   f"user{random.randint(1,999)}@okicici",
            "to_vpa":     f"merchant{random.randint(1,99)}@oksbi",
            "amount_paise": random.randint(100, 500000),  # paise not rupees!
            "currency":   "INR",
            "remarks":    random.choice(["Food", "Shopping", "Rent",
                                          "Transfer", "Bill Payment",
                                          None, "", "test", "????"]),
            "device_info": {
                "os":      random.choice(["Android 12", "iOS 16",
                                           "Android 13", None]),
                "app_ver": f"{random.randint(1,5)}.{random.randint(0,9)}",
                "ip":      f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
            }
        },
        "status":   random.choice(["SUCCESS", "FAILED", "PENDING",
                                    "REVERSED", None]),
        "bank_ref": f"BNKREF{random.randint(100000,999999)}"
                    if random.random() > 0.2 else None,  # sometimes missing!
        "_raw_bytes": f"0x{random.randint(0,0xFFFF):04X}",
        "_schema_ver": random.choice(["v1", "v2", "v1.5", None]),
    }

    # Inject some schema inconsistencies
    if i % 30 == 0:
        record["AMOUNT"] = record["payload"]["amount_paise"]  # duplicate field
    if i % 40 == 0:
        record["payload"]["amount_paise"] = "NULL"  # string instead of int
    if i % 50 == 0:
        record["timestamp_ms"] = "INVALID_TIMESTAMP"

    records.append(record)

with open(OUT_DIR + "gpay_upi_raw_dump.json", "w") as f:
    json.dump({"source": "UPI_GATEWAY_v3",
               "export_date": str(datetime.now()),
               "record_count": len(records),
               "warning": "RAW_UNPROCESSED_DATA",
               "records": records}, f, indent=2, default=str)

print(f"  ✅ gpay_upi_raw_dump.json")

# ── 3. Bank CSV — inconsistent schema ────────────────────────────
# Different branches export CSV in different formats!
print("[3/6] Multi-branch bank CSV (inconsistent schema)...")

with open(OUT_DIR + "branch_transactions_raw.csv", "w", newline="") as f:
    writer = csv.writer(f)

    # Branch A — uses one date format
    writer.writerow(["=== BRANCH: Chennai Main === EXPORT: " +
                     str(datetime.now())])
    writer.writerow(["Txn_ID", "Date", "Account", "Debit",
                     "Credit", "Balance", "Remarks"])
    for i in range(50):
        writer.writerow([
            f"CHN{i:06d}",
            f"{random.randint(1,28)}/{random.randint(1,12)}/2024",  # DD/MM/YYYY
            f"SB{random.randint(10000,99999)}",
            random.randint(0, 50000) if random.random()>0.5 else "",
            random.randint(0, 50000) if random.random()>0.5 else "",
            random.randint(1000, 500000),
            random.choice(["ATM WDL", "UPI CR", "NEFT", "IMPS", "CASH DEP", ""])
        ])

    # Branch B — uses different date format AND different column names!
    writer.writerow([])
    writer.writerow(["=== BRANCH: Mumbai Bandra === EXPORT: " +
                     str(datetime.now())])
    writer.writerow(["transaction_id", "transaction_date", "acct_no",
                     "withdrawal", "deposit", "bal", "narration",
                     "IFSC", "EXTRA_COL"])  # extra column!
    for i in range(50):
        writer.writerow([
            f"MUM{i:06d}",
            f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",  # YYYY-MM-DD
            f"CA{random.randint(10000,99999)}",
            f"₹{random.randint(0,50000):,}" if random.random()>0.5 else "NIL",
            f"₹{random.randint(0,50000):,}" if random.random()>0.5 else "NIL",
            f"INR {random.randint(1000,500000)}",
            random.choice(["Cash Withdrawal", "UPI", "NEFT Cr",
                            "Salary", "EMI Debit", ""]),
            f"HDFC{random.randint(1000,9999)}",
            "N/A"  # extra column branch A doesn't have
        ])

    # Branch C — corrupted rows mixed in
    writer.writerow([])
    writer.writerow(["=== BRANCH: Delhi CP === EXPORT: " +
                     str(datetime.now())])
    writer.writerow(["id","date","account","amount","type","fraud_flag"])
    for i in range(50):
        if i % 10 == 0:
            writer.writerow(["##RECORD_CORRUPTED##", "N/A", "N/A",
                              "N/A", "N/A", "N/A"])
        elif i % 15 == 0:
            writer.writerow([])  # blank row
        else:
            writer.writerow([
                f"DEL{i:06d}",
                datetime(2024, random.randint(1,12),
                         random.randint(1,28)).strftime("%b %d, %Y"),  # Yet another format!
                f"ACC{random.randint(10000,99999)}",
                random.randint(100, 100000),
                random.choice(["DR", "CR"]),
                random.choice(["Y", "N", "SUSPECTED", "", None])
            ])

print(f"  ✅ branch_transactions_raw.csv")

# ── 4. Credit Card Serialized Binary-like file (.dat) ─────────────
# Simulates legacy mainframe export — fixed-width, binary-like
print("[4/6] Legacy mainframe credit card data (.dat)...")

with open(OUT_DIR + "creditcard_mainframe_export.dat", "w") as f:
    f.write("HDR|CCDATA|20240101|20241231|BANK_XYZ|CONFIDENTIAL\n")
    f.write("FMT|CARDNO:16|DATE:8|AMT:12|MCC:4|COUNTRY:3|"
            "STATUS:1|FRAUD:1|SPARE:10\n")

    for i in range(200):
        card    = f"{''.join([str(random.randint(0,9)) for _ in range(16)])}"
        date    = f"2024{random.randint(1,12):02d}{random.randint(1,28):02d}"
        amt     = f"{random.randint(100,999999):012d}"  # amount in paise, no decimal
        mcc     = f"{random.randint(1000,9999)}"        # merchant category code
        country = random.choice(["IND","USA","GBR","SGP","ARE"])
        status  = random.choice(["A","D","R","P"])      # Approved/Declined/Reversed/Pending
        fraud   = "1" if random.random() < 0.002 else "0"
        spare   = "".join(random.choices("0123456789ABCDEF", k=10))

        # Fixed width — no headers per row, must know schema to parse
        f.write(f"DTL|{card}|{date}|{amt}|{mcc}|{country}|{status}|{fraud}|{spare}\n")

    f.write(f"TRL|TOTAL_RECORDS:{200}|CHECKSUM:{random.randint(100000,999999)}\n")

print(f"  ✅ creditcard_mainframe_export.dat")

# ── 5. Fraud Alerts — Mixed XML-like format ───────────────────────
print("[5/6] Fraud alerts raw XML feed...")

with open(OUT_DIR + "fraud_alerts_raw.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<!-- RAW FRAUD ALERT FEED - UNPROCESSED -->\n')
    f.write('<FraudAlerts source="FRAUD_ENGINE_v4" '
            f'generated="{datetime.now()}">\n')

    for i in range(100):
        ts       = datetime(2024, random.randint(1,12),
                            random.randint(1,28),
                            random.randint(0,23),
                            random.randint(0,59))
        score    = round(random.uniform(0, 1), 4)
        amount   = random.randint(1000, 500000)
        merchant = f"MER{random.randint(1000,9999)}"

        if i % 20 == 0:
            # Malformed XML record
            f.write(f'  <Alert id="{i}" -- MALFORMED>\n')
            f.write(f'    <Score>{score}</Score\n')  # missing closing >
            f.write(f'  </Alert>\n')
        elif i % 30 == 0:
            # Missing fields
            f.write(f'  <Alert id="{i}">\n')
            f.write(f'    <Score>{score}</Score>\n')
            f.write(f'    <!-- Amount field missing -->\n')
            f.write(f'  </Alert>\n')
        else:
            f.write(f'  <Alert id="{i}" timestamp="{ts.isoformat()}">\n')
            f.write(f'    <Score>{score}</Score>\n')
            f.write(f'    <Amount currency="INR">{amount}</Amount>\n')
            f.write(f'    <Merchant id="{merchant}"/>\n')
            f.write(f'    <Decision>{"BLOCK" if score>0.7 else "ALLOW"}</Decision>\n')
            f.write(f'    <RulesFired>{random.randint(1,8)}</RulesFired>\n')
            f.write(f'  </Alert>\n')

    f.write('</FraudAlerts>\n')

print(f"  ✅ fraud_alerts_raw.xml")

# ── 6. Summary comparison ─────────────────────────────────────────
print("[6/6] Creating clean vs unstructured comparison file...")

comparison = {
    "Source": [
        "ATM Machines",
        "GPay/UPI Gateway",
        "Bank Branches",
        "Credit Card System",
        "Fraud Engine"
    ],
    "Raw File": [
        "atm_raw_logs.log",
        "gpay_upi_raw_dump.json",
        "branch_transactions_raw.csv",
        "creditcard_mainframe_export.dat",
        "fraud_alerts_raw.xml"
    ],
    "Format": [
        "Unstructured text log",
        "Nested JSON with nulls",
        "3 different CSV schemas",
        "Fixed-width mainframe",
        "Malformed XML"
    ],
    "Problems": [
        "Mixed formats, corrupt records, heartbeat noise",
        "Wrong types, missing fields, duplicate keys",
        "3 date formats, extra columns, empty rows",
        "No headers, binary-like, encoded amounts",
        "Malformed tags, missing fields, encoding issues"
    ],
    "After Spark →": [
        "transactions_25m.csv",
        "transactions_25m.csv",
        "transactions_25m.csv",
        "transactions_25m.csv",
        "transactions_25m.csv"
    ]
}

pd.DataFrame(comparison).to_csv(
    OUT_DIR + "raw_vs_clean_summary.csv", index=False
)

print(f"  ✅ raw_vs_clean_summary.csv")

# ── Final summary ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  FILES GENERATED")
print(f"{'='*65}")
for fname in os.listdir(OUT_DIR):
    fpath = OUT_DIR + fname
    kb    = os.path.getsize(fpath) / 1024
    print(f"  {fname:<45} {kb:>8.1f} KB")

print(f"\n✅ All files saved to: {OUT_DIR}")
print("""
VIDEO USAGE:
  Scene 2 — Show these files as 'raw data arriving from sources'
  Then show transactions_25m.csv as 'after Spark processing'
  
  Sequence:
  1. Show atm_raw_logs.log  → "ATM machines dump this"
  2. Show gpay_upi_raw_dump.json → "Mobile payments send this"
  3. Show branch_transactions_raw.csv → "Branches export this"
  4. Show creditcard_mainframe_export.dat → "Legacy systems produce this"
  5. THEN show transactions_25m.csv → "Spark unifies all of this"
""")