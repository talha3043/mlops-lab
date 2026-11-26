from src import ml


def test_data_has_the_expected_shape():
    X, y = ml.load_data()
    assert X.shape == (178, 13)
    assert len(y) == 178


def test_data_has_no_missing_values():
    X, y = ml.load_data()
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_labels_are_the_three_wine_classes():
    _, y = ml.load_data()
    assert set(y.unique()) == {0, 1, 2}
