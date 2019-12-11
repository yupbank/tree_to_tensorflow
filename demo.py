from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import json

from ttt import export_decision_tree

tf.app.flags.DEFINE_string('output_dir', '/tmp/tree_savedmodel',
                           """Directory where to export tree saved model.""")

FLAGS = tf.app.flags.FLAGS


def main(_):
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    features = {'features': tf.placeholder(tf.float64, [None, X.shape[1]])}
    export_decision_tree(clf, features, FLAGS.output_dir)
    # base64.b64encode(tf.train.Example(features=tf.train.Features(feature={'features':tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 2.0, 1.0]))})).SerializeToString())


if __name__ == "__main__":
    tf.app.run()
