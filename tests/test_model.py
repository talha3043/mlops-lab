from sklearn.metrics import accuracy_score

from src import ml


def test_the_training_code_can_reach_a_good_model():
    # Fixed parameters on purpose: this test checks the code and the data,
    # not the values in params.yaml. The quality gate checks params.yaml.
    params = {"n_estimators": 100, "max_depth": 8, "seed": 42}
    X, y = ml.load_data()
    X_train, X_test, y_train, y_test = ml.split(X, y, params["seed"])
    model = ml.build_model(params).fit(X_train.values, y_train)
    assert accuracy_score(y_test, model.predict(X_test.values)) >= 0.90


def test_the_split_is_reproducible():
    X, y = ml.load_data()
    first = ml.split(X, y, 42)[1].index.tolist()
    second = ml.split(X, y, 42)[1].index.tolist()
    assert first == second
