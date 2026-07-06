# save as: regenerate_datasets.py
import pandas as pd
import numpy as np
import os

np.random.seed(42)
departments = ["HR", "Engineering", "Finance", "Marketing", "Operations"]
categories  = ["A", "B", "C", "D"]

sizes = {
    "emp_100k.csv":   100_000,
    "emp_500k.csv":   500_000,
    "emp_1m.csv":   1_000_000,
    "emp_2m.csv":   2_000_000,
}

for filename, n in sizes.items():
    print(f"Generating {filename} ({n:,} rows)...")
    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n + 1),
        "Department": np.random.choice(departments, n),
        "Category":   np.random.choice(categories, n),
        "Salary":     np.random.randint(30000, 120000, n),
        "Age":        np.random.randint(22, 60, n),
        "YearsExp":   np.random.randint(0, 35, n),
    })
    out = f"C:/Aravindh/data/{filename}"
    df.to_csv(out, index=False)
    print(f"  ✅ {os.path.getsize(out)/1024/1024:.1f} MB")

print("\n✅ All datasets regenerated with consistent schema!")