"""Train one model and log parameters, metrics and the model to MLflow."""
import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from src import ml

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "wine-classifier"


def main():
    parser = argparse.ArgumentParser(description="Train and track a model")
    parser.add_argument("--n-estimators", type=int, help="override params.yaml")
    parser.add_argument("--max-depth", type=int, help="override params.yaml")
    parser.add_argument("--run-name", help="name shown in the MLflow UI")
    parser.add_argument("--export-dir",
                        help="also save the model in this folder (used to test the quality gate)")
    args = parser.parse_args()

    params = ml.load_params()
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth

    X, y = ml.load_data()
    X_train, X_test, y_train, y_test = ml.split(X, y, params["seed"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({**params, "data_sha256": ml.data_sha256()})

        model = ml.build_model(params)
        cv_accuracy = cross_val_score(model, X_train.values, y_train, cv=5).mean()
        model.fit(X_train.values, y_train)
        predictions = model.predict(X_test.values)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        mlflow.log_metrics(
            {"cv_accuracy": cv_accuracy, "accuracy": accuracy, "f1_macro": f1}
        )
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        if args.export_dir:
            out = Path(args.export_dir)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, out / "model.joblib")
            (out / "metadata.json").write_text(
                json.dumps({"model_version": 0, "seed": params["seed"]}), encoding="utf-8")

        print(f"run id       : {run.info.run_id}")
        print(f"n_estimators : {params['n_estimators']}   max_depth: {params['max_depth']}")
        print(f"cv_accuracy  : {cv_accuracy:.3f}")
        print(f"accuracy     : {accuracy:.3f}   (hold-out test set)")
        print(f"f1_macro     : {f1:.3f}")


if __name__ == "__main__":
    main()
