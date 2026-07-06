import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt


# ---------------------------------------
# PATHS
# ---------------------------------------

DATASET_PATH = "C:/Aravindh/data/ml_dataset.csv"
MODEL_PATH = "C:/Aravindh/models/spark_optimizer.pkl"


# ---------------------------------------
# LOAD DATA
# ---------------------------------------

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)


# ---------------------------------------
# DEFINE FEATURES AND TARGET
# ---------------------------------------

target = "execution_time_sec"

drop_cols = [
    "execution_time_sec",
    "file_format"
]

X = df.drop(columns=drop_cols)
y = df[target]


# ---------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------

model = joblib.load(MODEL_PATH)


# ---------------------------------------
# PREDICTIONS
# ---------------------------------------

preds = model.predict(X)


# ---------------------------------------
# METRICS
# ---------------------------------------

mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
r2 = r2_score(y, preds)

print("\nMODEL PERFORMANCE")
print("----------------------------")
print("MAE  :", round(mae,3))
print("RMSE :", round(rmse,3))
print("R2   :", round(r2,3))


# ---------------------------------------
# CROSS VALIDATION
# ---------------------------------------

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

cv_scores = -cv_scores

print("\nCROSS VALIDATION (MAE)")
print("----------------------------")
print("Fold scores:", np.round(cv_scores,3))
print("Average CV MAE:", round(cv_scores.mean(),3))


# ---------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("----------------------------")
print(importance)


# ---------------------------------------
# BASELINE vs OPTIMIZED PERFORMANCE
# ---------------------------------------

baseline_shuffle = 200

baseline_rows = df[df["shuffle_partitions"] == baseline_shuffle]

if len(baseline_rows) > 0:

    baseline_runtime = baseline_rows["execution_time_sec"].mean()

else:

    baseline_runtime = df["execution_time_sec"].mean()

optimized_runtime = df["execution_time_sec"].min()

improvement = ((baseline_runtime - optimized_runtime) / baseline_runtime) * 100

print("\nOPTIMIZER PERFORMANCE")
print("----------------------------")
print("Baseline Runtime :", round(baseline_runtime,3))
print("Best Runtime     :", round(optimized_runtime,3))
print("Improvement (%)  :", round(improvement,2))


# ---------------------------------------
# PLOT 1 — Feature Importance
# ---------------------------------------

plt.figure()

importance.plot(kind="bar")

plt.title("Feature Importance")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ---------------------------------------
# PLOT 2 — Prediction vs Actual
# ---------------------------------------

plt.figure()

plt.scatter(y, preds)

plt.xlabel("Actual Runtime")

plt.ylabel("Predicted Runtime")

plt.title("Prediction vs Actual Runtime")

plt.tight_layout()

plt.show()


# ---------------------------------------
# PLOT 3 — Shuffle vs Runtime
# ---------------------------------------

plt.figure()

plt.scatter(df["shuffle_partitions"], df["execution_time_sec"])

plt.xlabel("Shuffle Partitions")

plt.ylabel("Execution Time")

plt.title("Shuffle Partitions vs Runtime")

plt.tight_layout()

plt.show()


print("\nEvaluation Complete.")