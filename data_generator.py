import argparse
import csv
import random

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

departments = ["Engineering", "HR", "Marketing", "Finance", "Sales"]

with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Department", "Salary", "Experience"])

    for i in range(args.rows):
        writer.writerow([
            f"Emp{i}",
            random.choice(departments),
            random.randint(30000, 150000),
            random.randint(0, 15)
        ])

print(f"Generated {args.rows} rows at {args.out}")
