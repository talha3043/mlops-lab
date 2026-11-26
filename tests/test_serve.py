import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src import ml
from src.serve import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    X, y = ml.load_data()
    params = {"n_estimators": 20, "max_depth": 5, "seed": 42}
    joblib.dump(ml.build_model(params).fit(X.values, y), model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_version": 1, "target_names": ml.TARGET_NAMES}),
        encoding="utf-8",
    )
    return TestClient(create_app(model_dir))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_a_class(client):
    X, y = ml.load_data()
    response = client.post("/predict", json={"features": X.iloc[0].tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1, 2)
    assert abs(sum(body["probabilities"]) - 1.0) < 0.01


def test_predict_rejects_the_wrong_number_of_features(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422
