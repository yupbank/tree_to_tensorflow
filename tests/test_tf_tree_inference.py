import numpy as np
import tensorflow as tf

from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ttt.tf_tree_inference import decision_tree_inference_in_tf as inference_tf


def test_classification():
    x, y = make_classification()
    clf = DecisionTreeClassifier()
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, y)
    res = inference_tf(d, clf)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), res.eval({d: x}))


def test_classification_multi_output():
    x, y = make_classification()
    clf = DecisionTreeClassifier()
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, np.vstack([y, y]).T)
    res = inference_tf(d, clf)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), res.eval({d: x}))


def test_regression():
    x, y = make_regression()
    clf = DecisionTreeRegressor()
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, y)
    res = inference_tf(d, clf)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), res.eval({d: x}))


def test_regression_multi_output():
    x, y = make_regression()
    clf = DecisionTreeRegressor()
    d = tf.placeholder(tf.float64, [None, x.shape[1]])
    clf.fit(x, np.vstack([y, y]).T)
    res = inference_tf(d, clf)
    with tf.Session():
        np.testing.assert_allclose(clf.predict(x), res.eval({d: x}))
