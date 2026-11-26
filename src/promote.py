"""Register the best run in the MLflow Model Registry, give it the alias
'champion' and export it to the model/ folder for packaging."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src import ml
from src.train import EXPERIMENT, TRACKING_URI

MODEL_NAME = "wine-classifier"
ALIAS = "champion"
OUT_DIR = Path(os.getenv("MODEL_DIR", "model"))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit("No experiment found. Run src.train first.")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["metrics.cv_accuracy DESC", "attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No runs found. Run src.train first.")
    best = runs[0]

    # MLflow 3 stores the model as a "logged model" linked to the run.
    model_id = best.outputs.model_outputs[0].model_id
    version = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    print(f"Registered {MODEL_NAME} version {version.version} from run {best.info.run_id}")
    print(f"Alias '{ALIAS}' now points to version {version.version}")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.joblib")

    X, _ = ml.load_data()
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": int(version.version),
        "run_id": best.info.run_id,
        "accuracy": round(best.data.metrics["accuracy"], 4),
        "cv_accuracy": round(best.data.metrics["cv_accuracy"], 4),
        "f1_macro": round(best.data.metrics["f1_macro"], 4),
        "n_estimators": int(best.data.params["n_estimators"]),
        "max_depth": int(best.data.params["max_depth"]),
        "seed": int(best.data.params["seed"]),
        "data_sha256": best.data.params["data_sha256"],
        "feature_names": list(X.columns),
        "target_names": ml.TARGET_NAMES,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported the champion to {OUT_DIR}/")


if __name__ == "__main__":
    main()
