from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col

spark = SparkSession.builder.appName("Day1Practice").getOrCreate()

# Sample employee dataset
data = [
    ("Alice", "Engineering", 90000),
    ("Bob", "Engineering", 85000),
    ("Charlie", "HR", 60000),
    ("David", "HR", 65000),
    ("Eve", "Marketing", 70000),
    ("Frank", "Marketing", 72000),
]

columns = ["Name", "Department", "Salary"]
df = spark.createDataFrame(data, columns)

print("\n📋 Original Data")
df.show()

print("\n👥 Employees per Department")
df.groupBy("Department").agg(count("*").alias("Employee_Count")).show()

print("\n💰 Average Salary per Department")
df.groupBy("Department").agg(avg("Salary").alias("Avg_Salary")).show()

print("\n🏆 Highest Paid Employee")
df.orderBy(col("Salary").desc()).show(1)

print("\n💎 Employees earning more than 70,000")
df.filter(col("Salary") > 70000).show()

print("\n📊 Department with Highest Average Salary")
df.groupBy("Department") \
  .agg(avg("Salary").alias("Avg_Salary")) \
  .orderBy(col("Avg_Salary").desc()) \
  .show(1)

print("\n📈 Salary Range (Min & Max)")
from pyspark.sql.functions import min, max
df.agg(min("Salary").alias("Min_Salary"),
       max("Salary").alias("Max_Salary")).show()


spark.stop()
