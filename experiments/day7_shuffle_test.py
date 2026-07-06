import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession.builder \
    .appName("ShuffleOptimization") \
    .getOrCreate()

input_path = sys.argv[1]

df = spark.read.csv(input_path, header=True, inferSchema=True)

# Small dimension table
dept_df = spark.createDataFrame(
    [("Engineering", "Tech"),
     ("HR", "Admin"),
     ("Marketing", "Business")],
    ["Department", "Category"]
)

print("\n=== BEFORE OPTIMIZATION (SHUFFLE JOIN) ===")
start = time.time()

joined_df = df.join(dept_df, "Department")
joined_df.groupBy("Category").avg("Salary").explain(True)
joined_df.groupBy("Category").avg("Salary").show()

end = time.time()
print("Execution Time (sec):", round(end - start, 2))

print("\n=== AFTER OPTIMIZATION (BROADCAST JOIN) ===")
start = time.time()

opt_df = df.join(broadcast(dept_df), "Department")
opt_df.groupBy("Category").avg("Salary").explain(True)
opt_df.groupBy("Category").avg("Salary").show()

end = time.time()
print("Execution Time (sec):", round(end - start, 2))

spark.stop()
