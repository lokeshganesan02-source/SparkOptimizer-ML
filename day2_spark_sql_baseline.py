from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkSQLBaseline-Local") \
    .master("local[*]") \
    .getOrCreate()


# Reduce log noise
spark.sparkContext.setLogLevel("ERROR")

# Load CSV
df = spark.read.csv(
    r"C:\Aravindh\employees.csv",
    header=True,
    inferSchema=True
)

# Register as SQL table
df.createOrReplaceTempView("employees")

print("\n📂 Employee Table")
spark.sql("SELECT * FROM employees").show()

print("\n👥 Employees per Department")
spark.sql("""
    SELECT Department, COUNT(*) AS total_employees
    FROM employees
    GROUP BY Department
""").show()

print("\n💰 Average Salary per Department")
spark.sql("""
    SELECT Department, AVG(Salary) AS avg_salary
    FROM employees
    GROUP BY Department
""").show()

print("\n🏆 Most Experienced Employee")
spark.sql("""
    SELECT *
    FROM employees
    ORDER BY Experience DESC
    LIMIT 1
""").show()

print("\n📈 Highest Salary in Company")
spark.sql("""
    SELECT MAX(Salary) AS max_salary
    FROM employees
""").show()

spark.stop()
