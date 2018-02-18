import numpy as np

from ttt.weight_utils import stat_from_weight, predict
from ttt.sklearn_helper import rf_to_weights_and_stats

def test_predict(weight, postive_x, negative_x, pred_positive, pred_negative):
    actual = predict(postive_x, weight)
    assert actual == pred_positive

    actual = predict(negative_x, weight)
    assert actual == pred_negative

def test_stat_from_weight(weight, stat):
    actual = stat_from_weight(weight)
    assert actual == stat


def test_predict(dataset, clf):
    weights, _ = rf_to_weights_and_stats(clf)
    X, y = dataset
    for i in range(clf.n_estimators):
        actual = predict(X, weights[i])
        actual = np.array([i for i in actual])
        actual = np.where(actual[:,0]> actual[:,1], 0.0, 1.0)
        expected = clf.estimators_[i].predict(X)
        assert np.array_equal(actual, expected)
