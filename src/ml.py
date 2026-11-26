"""Shared helpers: data loading, the train/test split and the model definition."""
import hashlib
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "wine.csv"
PARAMS_PATH = ROOT / "params.yaml"
TARGET = "target"
TARGET_NAMES = ["class_0", "class_1", "class_2"]


def load_params(path=PARAMS_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def data_sha256(path=DATA_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split(X, y, seed):
    """The same seed always gives the same split, so results can be reproduced."""
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def build_model(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["seed"],
    )
