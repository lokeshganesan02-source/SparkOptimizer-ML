# train_optimizer_model.py — UPDATED for banking dataset
# Includes executor_cores feature
# Uses log-transform for runtime prediction
# Compares RandomForest, GradientBoosting, XGBoost

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed — run: pip install xgboost")

ML_DATA_PATH  = "C:/Aravindh/data/ml_dataset.csv"
MODEL_PATH    = "C:/Aravindh/models/spark_optimizer.pkl"
FEATURES_PATH = "C:/Aravindh/models/feature_columns.json"
RESULTS_PATH  = "C:/Aravindh/data/model_comparison.csv"

os.makedirs("C:/Aravindh/models", exist_ok=True)

print("=" * 65)
print("  SPARK ML OPTIMIZER — MODEL TRAINING")
print("  Banking Dataset Version")
print("=" * 65)

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv(ML_DATA_PATH)
print(f"\nLoaded dataset: {len(df)} rows × {len(df.columns)} columns")

# ─────────────────────────────────────────────
# Remove outliers
# ─────────────────────────────────────────────
before = len(df)
mean = df["execution_time_sec"].mean()
std  = df["execution_time_sec"].std()

df = df[abs(df["execution_time_sec"] - mean) < 3 * std].reset_index(drop=True)

print(f"Outliers removed: {before - len(df)}")
print(f"Training samples: {len(df)}")

# ─────────────────────────────────────────────
# Features and target
# ─────────────────────────────────────────────
X = df.drop(columns=["execution_time_sec"])
y = df["execution_time_sec"]

# Log transform target
y_original = y.copy()
y = np.log1p(y)

print("\nTarget transformed to log scale")
print(f"Original range: {y_original.min():.2f}s – {y_original.max():.2f}s")
print(f"Log range: {y.min():.3f} – {y.max():.3f}")

# One-hot encode categorical features
X = pd.get_dummies(X)

feature_columns = X.columns.tolist()

print(f"\nTotal features: {len(feature_columns)}")

# ─────────────────────────────────────────────
# Cross validation
# ─────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
models = {

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        min_samples_leaf=10,
        random_state=42
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

print("\n" + "─" * 65)
print("MODEL COMPARISON")
print("─" * 65)

comparison_rows = []
best_model = None
best_r2 = -999
best_name = None

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
for name, model in models.items():

    print(f"\nTraining {name}...")

    # Cross validation (log scale)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    # Train full model
    model.fit(X, y)

    # Predictions (log scale)
    preds_log = model.predict(X)

    # Convert back to seconds
    preds_original = np.expm1(preds_log)
    y_original_vals = np.expm1(y)

    # Metrics on original scale
    mae  = mean_absolute_error(y_original_vals, preds_original)
    rmse = np.sqrt(np.mean((y_original_vals - preds_original) ** 2))
    r2   = r2_score(y_original_vals, preds_original)

    print(f"MAE:  {mae:.3f}s")
    print(f"RMSE: {rmse:.3f}s")
    print(f"R²:   {r2:.3f}")
    print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    comparison_rows.append({
        "model": name,
        "mae": round(mae,3),
        "rmse": round(rmse,3),
        "r2": round(r2,3),
        "cv_r2": round(cv_scores.mean(),3),
        "cv_std": round(cv_scores.std(),3)
    })

    if cv_scores.mean() > best_r2:
        best_r2 = cv_scores.mean()
        best_model = model
        best_name = name

# ─────────────────────────────────────────────
# Save comparison results
# ─────────────────────────────────────────────
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(RESULTS_PATH, index=False)

print("\nBest model:", best_name, f"(CV R² = {best_r2:.3f})")

# ─────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):

    importance = pd.Series(
        best_model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    print("\nTop 10 features:")

    for feat, val in importance.head(10).items():
        bar = "█" * int(val * 40)
        print(f"{feat:<30} {val:.4f} {bar}")

# ─────────────────────────────────────────────
# Save best model
# ─────────────────────────────────────────────
joblib.dump(best_model, MODEL_PATH)

with open(FEATURES_PATH, "w") as f:
    json.dump(feature_columns, f)

print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)

for r in comparison_rows:
    print(f"{r['model']:<18} R²={r['r2']:.3f}  CV={r['cv_r2']:.3f}±{r['cv_std']:.3f}")

print("\nSaved files:")
print("Model:", MODEL_PATH)
print("Features:", FEATURES_PATH)
print("Comparison:", RESULTS_PATH)

print("\nNext step:")
print("python spark_optimizer_proof.py")