from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PartitionBaseline").getOrCreate()

df = spark.read.option("header", True).csv("C:/Aravindh/data/emp_1m.csv")

print("Initial partitions:", df.rdd.getNumPartitions())

df.groupBy("Department").count().show()

print("Default shuffle partitions:",
      spark.conf.get("spark.sql.shuffle.partitions"))


spark.stop()
