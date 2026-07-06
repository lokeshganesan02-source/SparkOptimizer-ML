from pyspark.sql import SparkSession
import os

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', 'NOT SET')}")
print(f"SPARK_HOME: {os.environ.get('SPARK_HOME', 'NOT SET')}")
print(f"HADOOP_HOME: {os.environ.get('HADOOP_HOME', 'NOT SET')}")
print("=" * 60)

spark = SparkSession.builder \
    .appName("WindowsTest") \
    .master("local[*]") \
    .config("spark.sql.warehouse.dir", "C:/tmp/spark-warehouse") \
    .getOrCreate()

print(f"\n✅ Spark Version: {spark.version}")
print(f"✅ Expected: 3.5.8")

data = [("Alice", 34), ("Bob", 45), ("Charlie", 28)]
df = spark.createDataFrame(data, ["Name", "Age"])

print("\n✅ DataFrame created!")
df.show()

filtered = df.filter(df.Age > 30)
print(f"✅ Filtered count: {filtered.count()}")

avg_age = df.agg({"Age": "avg"}).collect()[0][0]
print(f"✅ Average age: {avg_age:.1f}")

spark.stop()
print("\n✅ ALL TESTS PASSED!")