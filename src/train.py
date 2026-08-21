import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """Train a model, track it in MLflow, and persist its outputs."""
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    for name, frame in (("training", df_train), ("evaluation", df_eval)):
        if "target" not in frame.columns:
            raise ValueError(f"{name} data is missing the required 'target' column")

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # CI and remote tracking servers can override the local defaults.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "income-classification"))

    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w", encoding="utf-8") as report_file:
            json.dump({"f1_score": f1, "accuracy": acc}, report_file, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)
    train(params)
