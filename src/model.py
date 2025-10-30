import os
import json
import yaml
import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_model(train: pd.DataFrame, params: dict):
    X = train.drop("high_quality", axis=1)
    y = train["high_quality"]
    model = RandomForestClassifier(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def evaluate(model, test: pd.DataFrame) -> dict:
    X = test.drop("high_quality", axis=1)
    y = test["high_quality"]
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds)
    }

@click.command()
def train():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("wine_quality")

    with mlflow.start_run(run_name="rf_baseline"):
        params = yaml.safe_load(open("params.yaml"))["train"]
        mlflow.log_params(params)

        train_df = load_split("data/processed/train.csv")
        test_df = load_split("data/processed/test.csv")

        model = train_model(train_df, params)
        metrics = evaluate(model, test_df)
        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)
        model_path = "models/rf.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/train.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if metrics["f1"] > 0.70:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="WineQualityRF"
            )

if __name__ == "__main__":
    train()