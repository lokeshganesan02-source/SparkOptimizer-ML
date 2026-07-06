import time
from pyspark.sql import SparkSession

start_time = time.time()

spark = SparkSession.builder \
    .appName("BaselineKPI") \
    .master("local-cluster[2,2,1024]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    r"C:\Aravindh\employees.csv",
    header=True,
    inferSchema=True
)

df.createOrReplaceTempView("employees")

spark.sql("""
    SELECT Department, AVG(Salary)
    FROM employees
    GROUP BY Department
""").show()

end_time = time.time()

execution_time_sec = end_time - start_time

# Extract executor info
conf = spark.sparkContext.getConf()
executors = conf.get("spark.executor.instances", "unknown")

print("\n BASELINE METRICS")
print(f"Execution Time (sec): {execution_time_sec:.2f}")
print(f"Executor Instances: {executors}")

spark.stop()
