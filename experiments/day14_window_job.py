# ============================================================
# day14_window_job.py  — WINDOW job (banking version)
# Per-account fraud ranking — running totals, risk scoring
# ============================================================
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def main(dataset_path):
    spark = SparkSession.builder \
        .appName("banking_window") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    start = time.time()

    txn = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # Window by account — rank transactions by amount
    w_account = Window.partitionBy("account_id").orderBy(F.col("amount").desc())
    # Window by merchant — rank by fraud count
    w_merchant = Window.partitionBy("merchant_id").orderBy(F.col("timestamp"))

    result = txn \
        .withColumn("rank_in_account",       F.rank().over(w_account)) \
        .withColumn("dense_rank_in_account",  F.dense_rank().over(w_account)) \
        .withColumn("pct_rank_in_account",    F.percent_rank().over(w_account)) \
        .withColumn("running_total",          F.sum("amount").over(
            w_merchant.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )) \
        .filter(F.col("rank_in_account") <= 10)

    count = result.count()
    elapsed = round(time.time() - start, 3)

    print(f"EXECUTION_TIME:{elapsed}")
    print(f"Result rows: {count}")
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/Aravindh/data/txn_5m.csv")


# ============================================================
# SAVE AS: C:\Aravindh\day14_window_job.py
# ============================================================