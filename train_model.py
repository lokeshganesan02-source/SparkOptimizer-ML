import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from ml_dataset_builder import build_ml_dataset


def train():

    df = build_ml_dataset()

    # Drop non-numeric columns
    df = df.drop(columns=["file_format"])

    # Separate features and target
    X = df.drop(columns=["execution_time_sec"])
    y = df["execution_time_sec"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    print("\nModel Evaluation")
    print("----------------")
    print("MAE:", round(mae, 4))

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False))

    return model


if __name__ == "__main__":
    train()