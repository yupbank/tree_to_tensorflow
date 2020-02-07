import xgboost as xgb
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression

from ttt.tf_xgb_inference import TreeRegressionInference


def test_regression_inference():
    x, y = make_regression()
    model = xgb.XGBRegressor().fit(x, y)
    a = TreeRegressionInference(model)
    with tf.Session() as sess:
      np.testing.assert_array_almost_equal(
          sess.run(a.predict(x)), model.predict(x), decimal=3)
