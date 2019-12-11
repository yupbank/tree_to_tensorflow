import numpy as np
import tensorflow as tf
import os

from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ttt.saved_model_export import export_decision_tree


def test_export_decision_tree(tmpdir, dlf, dataset):
    x, y = dataset
    path = os.path.join(tmpdir.dirname, tmpdir.basename)
    with tf.Graph().as_default():
        d = tf.placeholder(tf.float64, [None, x.shape[1]])
        export_decision_tree(dlf, {'features': d}, path)
        assert len(tmpdir.listdir()) == 1
