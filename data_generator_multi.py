import pandas as pd
import random

departments = ["Engineering", "HR", "Finance", "Sales", "Marketing"]

def generate_dataset(rows, output_path):

    data = []

    for i in range(rows):

        # Create skew
        if random.random() < 0.9:
            dept = "Engineering"
        else:
            dept = random.choice(departments)

        data.append({
            "emp_id": i,
            "salary": random.randint(30000, 120000),
            "Department": dept
        })

    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False)

    print("Generated:", output_path)


if __name__ == "__main__":

    generate_dataset(100000, "C:/Aravindh/data/emp_100k.csv")
    generate_dataset(500000, "C:/Aravindh/data/emp_500k.csv")
    generate_dataset(1000000, "C:/Aravindh/data/emp_1m.csv")
    generate_dataset(2000000, "C:/Aravindh/data/emp_2m.csv")