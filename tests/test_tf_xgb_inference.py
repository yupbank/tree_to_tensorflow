import xgboost as xgb
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression, make_classification

from ttt.tf_xgb_inference import TreeRegressionInference, TreeClassificationInference


def test_regression_inference():
    x, y = make_regression(random_state=10)
    model = xgb.XGBRegressor(random_state=10).fit(x, y)
    a = TreeRegressionInference(model)
    with tf.Session() as sess:
      np.testing.assert_array_almost_equal(
          sess.run(a.predict(x)), model.predict(x), decimal=3)


def test_classification_inference_2_class():
    x, y = make_classification(random_state=10)
    model = xgb.XGBClassifier(random_state=10).fit(x, y)
    b = TreeClassificationInference(model)
    with tf.Session() as sess:
      np.testing.assert_array_almost_equal(
          sess.run(b.predict_proba(x)), model.predict_proba(x), decimal=3)


def test_classification_inference_3_class():
    x, y = make_classification(random_state=10, n_classes=3, n_informative=10)
    model = xgb.XGBClassifier(random_state=10).fit(x, y)
    b = TreeClassificationInference(model)
    with tf.Session() as sess:
      np.testing.assert_array_almost_equal(
          sess.run(b.predict_proba(x)), model.predict_proba(x), decimal=3)
