# ============================================================
# day13_filter_job.py  — FILTER job (banking version)
# High-value fraud detection pipeline
# ============================================================
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(dataset_path):
    spark = SparkSession.builder \
        .appName("banking_filter") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    start = time.time()

    txn = spark.read.csv(dataset_path, header=True, inferSchema=True)

    result = txn \
        .filter(
            (F.col("amount") > 1000) &
            (F.col("status") != "failed") &
            (F.col("transaction_type").isin(["purchase", "transfer", "withdrawal"]))
        ) \
        .filter(
            (F.col("hour_of_day").between(0, 5)) |
            (F.col("is_fraud") == 1)
        ) \
        .groupBy("location", "channel", "day_of_week") \
        .agg(
            F.count("transaction_id").alias("suspicious_txn_count"),
            F.sum("amount").alias("suspicious_total"),
            F.sum("is_fraud").alias("confirmed_fraud"),
            F.avg("amount").alias("avg_suspicious_amount")
        ) \
        .filter(F.col("suspicious_txn_count") > 5)

    count = result.count()
    elapsed = round(time.time() - start, 3)

    print(f"EXECUTION_TIME:{elapsed}")
    print(f"Result rows: {count}")
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/Aravindh/data/txn_5m.csv")


# ============================================================
# SAVE AS: C:\Aravindh\day13_filter_job.py
# ============================================================
