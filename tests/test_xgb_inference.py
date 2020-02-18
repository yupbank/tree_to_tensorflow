import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.datasets import make_regression, make_classification

from ttt.xgb_inference import RegressionInference, ClassificationInference


def test_regression_inference():
    x, y = make_regression(random_state=10)
    model = xgb.XGBRegressor(random_state=10).fit(x, y)
    a = RegressionInference(model)
    with tf.Session() as sess:
        np.testing.assert_array_almost_equal(
            sess.run(a.predict(x)), model.predict(x), decimal=3
        )


def test_classification_inference_2_class(binary_classification_dataset):
    x, y = binary_classification_dataset
    model = xgb.XGBClassifier(random_state=10).fit(x, y)
    b = ClassificationInference(model)
    with tf.Session() as sess:
        np.testing.assert_array_almost_equal(
            sess.run(b.predict_proba(x)), model.predict_proba(x), decimal=3
        )
        np.testing.assert_array_almost_equal(sess.run(b.predict(x)), model.predict(x))


def test_classification_inference_3_class(multiclass_classification_dataset):
    x, y = multiclass_classification_dataset
    model = xgb.XGBClassifier(random_state=10).fit(x, y)
    b = ClassificationInference(model)
    with tf.Session() as sess:
        np.testing.assert_array_almost_equal(
            sess.run(b.predict_proba(x)), model.predict_proba(x), decimal=3
        )
        np.testing.assert_array_almost_equal(sess.run(b.predict(x)), model.predict(x))
