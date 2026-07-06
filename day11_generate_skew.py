import pandas as pd
import numpy as np

rows = 1_000_000

# 90% of rows go to one department (Engineering)
departments = np.random.choice(
    ["Engineering", "HR", "Finance", "Sales", "Marketing"],
    size=rows,
    p=[0.9, 0.025, 0.025, 0.025, 0.025]
)

df = pd.DataFrame({
    "emp_id": range(rows),
    "Department": departments,
    "salary": np.random.randint(30000, 120000, size=rows)
})

df.to_csv("C:/Aravindh/data/emp_skewed.csv", index=False)

print("Skewed dataset generated.")
