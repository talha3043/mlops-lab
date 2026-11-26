"""Quality gate: the exported model must reach a minimum accuracy on the
hold-out test set, otherwise the pipeline stops."""
import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

from src import ml


def main():
    parser = argparse.ArgumentParser(description="Check the model quality")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    X, y = ml.load_data()
    _, X_test, _, y_test = ml.split(X, y, meta["seed"])
    accuracy = accuracy_score(y_test, model.predict(X_test.values))

    print(f"model version : {meta['model_version']}")
    print(f"test accuracy : {accuracy:.3f}")
    print(f"required      : {args.min_accuracy:.3f}")
    if accuracy < args.min_accuracy:
        print("QUALITY GATE FAILED: the model is not good enough to ship.")
        sys.exit(1)
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
