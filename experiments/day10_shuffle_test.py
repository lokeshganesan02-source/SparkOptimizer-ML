from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time

spark = (
    SparkSession.builder
    .appName("Day10-Shuffle-Test")
    .getOrCreate()
)

# Log level low to reduce noise
spark.sparkContext.setLogLevel("ERROR")

print("Shuffle partitions:",
      spark.conf.get("spark.sql.shuffle.partitions"))

# Input data (same dataset you used earlier)
DATA_PATH = "C:/Aravindh/data/emp_1m.csv"

# Read CSV
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(DATA_PATH)
)

# Force a shuffle-heavy operation
start = time.time()

result = (
    df.groupBy("Department")
      .count()
      .orderBy(col("count").desc())
)

result.show()

end = time.time()

print(f"Execution Time (sec): {round(end - start, 2)}")

spark.stop()
