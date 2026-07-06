import pandas as pd
import numpy as np
import os

print("Generating emp_5m.csv...")

np.random.seed(42)
n = 5_000_000

departments = ["HR", "Engineering", "Finance", "Marketing", "Operations"]
categories  = ["A", "B", "C", "D"]

df = pd.DataFrame({
    "EmployeeID":  np.arange(1, n + 1),
    "Department":  np.random.choice(departments, n),
    "Category":    np.random.choice(categories, n),
    "Salary":      np.random.randint(30000, 120000, n),
    "Age":         np.random.randint(22, 60, n),
    "YearsExp":    np.random.randint(0, 35, n),
})

out = "C:/Aravindh/data/emp_5m.csv"
os.makedirs("C:/Aravindh/data", exist_ok=True)
df.to_csv(out, index=False)
print(f"✅ Saved {n:,} rows to {out}")
print(f"   File size: {os.path.getsize(out)/1024/1024:.1f} MB")