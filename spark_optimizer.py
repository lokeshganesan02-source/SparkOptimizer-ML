import joblib
import pandas as pd
from feature_builder import build_feature_vector

MODEL_PATH = "C:/Aravindh/models/spark_optimizer.pkl"
DATA_PATH = "C:/Aravindh/data/emp_skewed.csv"
DATASET_PATH = "C:/Aravindh/data/ml_dataset.csv"

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(MODEL_PATH)


# Load training dataset to get feature order`1`
train_df = pd.read_csv(DATASET_PATH)

X_train = train_df.drop(columns=["execution_time_sec"])
X_train = pd.get_dummies(X_train, drop_first=True)

training_columns = X_train.columns


# -----------------------------
# Recommendation function
# -----------------------------
def recommend_shuffle(sql_query):

    features = build_feature_vector(sql_query, DATA_PATH)

    df = pd.DataFrame([features])

    # Apply same encoding as training
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=training_columns, fill_value=0)

    configs = [4, 8, 16, 32, 64]

    results = []

    for shuffle in configs:

        df["shuffle_partitions"] = shuffle

        pred = model.predict(df)[0]

        results.append((shuffle, pred))

    best = sorted(results, key=lambda x: x[1])[0]

    return best, results


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":

    sql = """
    SELECT d.Category, COUNT(*)
    FROM employees e
    JOIN department d
    ON e.Department = d.Department
    GROUP BY d.Category
    """

    best, results = recommend_shuffle(sql)

    print("\nPredictions:")
    for r in results:
        print("shuffle =", r[0], "predicted_time =", round(r[1],3))

    print("\nRecommended shuffle:", best[0])