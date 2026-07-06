# ============================================================
# day11_skew_join.py  — JOIN job (banking version)
# Joins transactions + accounts + merchants
# Realistic: fraud analysis requires all 3 tables
# ============================================================
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(dataset_path):
    spark = SparkSession.builder \
        .appName("banking_join") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    DATA_DIR = "C:/Aravindh/data/"

    # Load all 3 tables
    txn       = spark.read.csv(dataset_path, header=True, inferSchema=True)
    accounts  = spark.read.csv(DATA_DIR + "accounts_500k.csv",  header=True, inferSchema=True)
    merchants = spark.read.csv(DATA_DIR + "merchants_100k.csv", header=True, inferSchema=True)

    # Fix: rename duplicate 'country' columns before joining
    accounts  = accounts.withColumnRenamed("country", "account_country")
    merchants = merchants.withColumnRenamed("country", "merchant_country")

    start = time.time()

    # Join transactions → accounts → merchants
    result = txn \
        .join(accounts,  on="account_id",  how="left") \
        .join(merchants, on="merchant_id", how="left") \
        .groupBy("category", "account_type", "account_country") \
        .agg(
            F.count("transaction_id").alias("txn_count"),
            F.sum("amount").alias("total_amount"),
            F.avg("amount").alias("avg_amount"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("credit_score").alias("avg_credit_score")
        ) \
        .filter(F.col("txn_count") > 10)

    count = result.count()
    elapsed = round(time.time() - start, 3)

    print(f"EXECUTION_TIME:{elapsed}")
    print(f"Result rows: {count}")
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/Aravindh/data/txn_5m.csv")


# ============================================================
# SAVE AS: C:\Aravindh\day11_skew_join.py
# ============================================================