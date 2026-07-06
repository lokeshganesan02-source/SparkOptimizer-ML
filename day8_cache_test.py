import time
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

spark = SparkSession.builder.appName("CacheVsNoCache").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

path = "C:/Aravindh/data/emp_1m.csv"

# READ
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

# FILTER FIRST (important)
df_filt = df.filter(df.Department.isNotNull())

# ---------- RUN 1: NO CACHE ----------
t1 = time.time()
df_filt.groupBy("Department").avg("Salary").show()
df_filt.groupBy("Department").count().show()
print("NO CACHE Time:", round(time.time() - t1, 2), "sec")

# ---------- RUN 2: CACHE ----------
df_cached = df_filt.cache()
df_cached.count()  # materialize cache

t2 = time.time()
df_cached.groupBy("Department").avg("Salary").show()
df_cached.groupBy("Department").count().show()
print("CACHE (MEMORY_ONLY) Time:", round(time.time() - t2, 2), "sec")

# ---------- RUN 3: PERSIST ----------
df_persist = df_filt.persist(StorageLevel.MEMORY_AND_DISK)
df_persist.count()

t3 = time.time()
df_persist.groupBy("Department").avg("Salary").show()
df_persist.groupBy("Department").count().show()
print("PERSIST (MEMORY_AND_DISK) Time:", round(time.time() - t3, 2), "sec")

spark.stop()
