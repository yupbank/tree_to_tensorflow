import numpy as np
import tensorflow as tf
import pytest

from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ttt.sk_inference import TreeClassificationInference, TreeRegressionInference, ForestClassifierInference, ForestRegressorInference


@pytest.fixture(scope="function", params=[(DecisionTreeClassifier(), TreeClassificationInference),
                                          (RandomForestClassifier(n_estimators=4), ForestClassifierInference)],
                ids=['DecisionTreeClassifier', 'RandomForestClassifier'])
def classifiers(request):
    param = request.param
    yield param


@pytest.fixture(scope="function", params=[(DecisionTreeRegressor(), TreeRegressionInference),
                                          (RandomForestRegressor(n_estimators=4), ForestRegressorInference)],
                ids=['DecisionTreeRegressor', 'RandomForestRegressor'])
def regressors(request):
    param = request.param
    yield param


def test_classification(classifiers, binary_classification_dataset):
    clf, inferencer = classifiers
    x, y = binary_classification_dataset
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, y)
    tf_infer = inferencer(clf)
    predict_res = tf_infer.predict(d)
    predict_prob_res = tf_infer.predict_prob(d)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), predict_res.eval({d: x}))
        np.testing.assert_allclose(clf.predict_proba(
            x), predict_prob_res.eval({d: x}))

    clf.fit(x, np.vstack([y, y]).T)
    tf_infer = inferencer(clf)
    predict_res = tf_infer.predict(d)
    predict_prob_res = tf_infer.predict_prob(d)

    def reshape_proba(x): return np.concatenate(
        [r[:, np.newaxis, :] for r in clf.predict_proba(x)], axis=1)

    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), predict_res.eval({d: x}))
        np.testing.assert_allclose(reshape_proba(
            x), predict_prob_res.eval({d: x}))


def test_regression(regressors):
    x, y = make_regression()
    clf, inferencer = regressors
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, y)
    tf_infer = inferencer(clf)
    predict_res = tf_infer.predict(d)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), predict_res.eval({d: x}))

    clf.fit(x, np.vstack([y, y]).T)
    tf_infer = inferencer(clf)
    predict_res = tf_infer.predict(d)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), predict_res.eval({d: x}))