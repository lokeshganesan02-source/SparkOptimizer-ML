# ============================================================
# day12_aggregation_job.py  — AGGREGATION job (banking version)
# Per-merchant fraud analytics — realistic analytics pipeline
# ============================================================
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(dataset_path):
    spark = SparkSession.builder \
        .appName("banking_aggregation") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    start = time.time()

    txn = spark.read.csv(dataset_path, header=True, inferSchema=True)

    result = txn \
        .groupBy("merchant_id", "transaction_type", "channel") \
        .agg(
            F.count("transaction_id").alias("txn_count"),
            F.sum("amount").alias("total_amount"),
            F.avg("amount").alias("avg_amount"),
            F.stddev("amount").alias("stddev_amount"),
            F.min("amount").alias("min_amount"),
            F.max("amount").alias("max_amount"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
            F.countDistinct("account_id").alias("unique_accounts")
        ) \
        .filter(F.col("txn_count") > 10) \
        .orderBy(F.col("fraud_count").desc())

    count = result.count()
    elapsed = round(time.time() - start, 3)

    print(f"EXECUTION_TIME:{elapsed}")
    print(f"Result rows: {count}")
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/Aravindh/data/txn_5m.csv")


# ============================================================
# SAVE AS: C:\Aravindh\day12_aggregation_job.py
# ============================================================