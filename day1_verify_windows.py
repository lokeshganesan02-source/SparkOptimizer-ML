import sys
import subprocess
import os

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

print("=" * 70)
print("DAY 1 ENVIRONMENT VERIFICATION (WINDOWS - Spark 3.5.8)")
print("=" * 70)

# Environment variables
print("\n📌 ENVIRONMENT VARIABLES:")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME', '❌ NOT SET')}")
print(f"SPARK_HOME: {os.environ.get('SPARK_HOME', '❌ NOT SET')}")
print(f"HADOOP_HOME: {os.environ.get('HADOOP_HOME', '❌ NOT SET')}")

# Java
print("\n☕ JAVA:")
java_out = run_command("java -version")
print(java_out.split('\n')[0] if java_out else "❌ NOT FOUND")

# Spark
print("\n⚡ SPARK:")
spark_home = os.environ.get("SPARK_HOME", "")
spark_submit = os.path.join(spark_home, "bin", "spark-submit.cmd")

if os.path.exists(spark_submit):
    spark_out = run_command(f'"{spark_submit}" --version')
else:
    spark_out = "❌ spark-submit not found"

for line in spark_out.split('\n'):
    if 'version' in line.lower():
        print(line.strip())
        break

# Python
print("\n🐍 PYTHON:")
print(f"Version: {sys.version.split()[0]}")
print(f"Path: {sys.executable}")

# PySpark
print("\n📦 PYSPARK:")
try:
    import pyspark
    version = pyspark.__version__
    print(f"Version: {version}")
    
    if version == "3.5.8":
        print("\n" + "=" * 70)
        print("✅ ✅ ✅ ENVIRONMENT READY! ✅ ✅ ✅")
        print("=" * 70)
    else:
        print(f"\n⚠️  Expected 3.5.8, found {version}")
except ImportError:
    print("❌ NOT INSTALLED")

# Winutils check
hadoop_home = os.environ.get("HADOOP_HOME", "")
winutils_path = os.path.join(hadoop_home, "bin", "winutils.exe")
print(f"\n🔧 WINUTILS:")
if os.path.exists(winutils_path):
    print(f"✅ Found at {winutils_path}")
else:
    print(f"❌ NOT FOUND at {winutils_path}")
    print("   Download from: https://github.com/cdarlint/winutils")

print("\n" + "=" * 70)