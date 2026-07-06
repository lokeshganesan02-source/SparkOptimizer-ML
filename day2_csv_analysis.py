from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, max

spark = SparkSession.builder.appName("CSVAnalysis").getOrCreate()

df = spark.read.csv(r"C:\Aravindh\employees.csv", header=True, inferSchema=True)

print("\n📂 CSV Data")
df.show()

print("\n👥 Employees per Department")
df.groupBy("Department").agg(count("*").alias("Total")).show()

print("\n💰 Avg Salary by Department")
df.groupBy("Department").agg(avg("Salary").alias("Avg_Salary")).show()

print("\n🏆 Most Experienced Employee")
df.orderBy(col("Experience").desc()).show(1)

print("\n📈 Highest Salary in Company")
df.agg(max("Salary").alias("Max_Salary")).show()

spark.stop()
