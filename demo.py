from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import json

from ttt import export_sklearn_rf


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoint',
                           """Directory where to save tree model checkpoint.""")

tf.app.flags.DEFINE_string('output_dir', '/tmp/tree_savedmodel',
                           """Directory where to export tree saved model.""")

FLAGS = tf.app.flags.FLAGS

def main(_):
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X = np.asarray(X, dtype=np.float32)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    tf_estimator = export_sklearn_rf(clf, FLAGS.checkpoint_dir)
    assert isinstance(tf_estimator, tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator)

    feature_spec = {'features': tf.FixedLenFeature([4], tf.float32)}
    export_func = tf.contrib.learn.build_parsing_serving_input_fn(feature_spec)
    tf_estimator.export_savedmodel(FLAGS.output_dir, export_func)
    pred = tf_estimator.evaluate(X, y)
    print(pred)
    print y[0], y[1]
    print json.dumps(dict(instances=[dict(features=X[0].tolist()), dict(features=X[1].tolist())]))
    # base64.b64encode(tf.train.Example(features=tf.train.Features(feature={'features':tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0, 2.0, 1.0]))})).SerializeToString())


if __name__ == "__main__":
    tf.app.run()
