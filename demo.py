from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tree_to_tensorflow.sklearn_exporter import export_random_forest_classier
from tree_to_tensorflow.tf_helper import RandomForestInferenceGraphs
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import tensorflow as tf
import numpy as np


def export(tree_weights, params, model_dir, X, y):
    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        graph = RandomForestInferenceGraphs(params, tree_weights)
        logits, tree_paths, regression_variance = graph.inference_graph(dx)
        init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

        with tf.Session() as sess:
            sess.run(init_vars)
            pred = sess.run(logits, feed_dict={dx: X})
            print 'tensorflow', classification_report(y, np.where((pred[:, 0] - pred[:, 1]) > 0, 0, 1))
            saver = tf.train.Saver()
            saver.save(sess, model_dir)

def import_model(model_dir, params, X, y):
    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        graph = RandomForestInferenceGraphs(params, None)
        logits, tree_paths, regression_variance = graph.inference_graph(dx)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
            pred = sess.run(logits, feed_dict={dx: X})
            print 'imported tensorflow', classification_report(y, np.where((pred[:, 0] - pred[:, 1]) > 0, 0, 1))


def main():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X = np.asarray(X, dtype=np.float32)
    model_dir = 'tmp/one'
    clf = RandomForestClassifier()
    clf.fit(X, y)
    print 'sklearn', classification_report(y, clf.predict(X))
    params = tensor_forest.ForestHParams(
        num_classes=2,
        num_features=4,
        num_trees=10,
        max_nodes=100)
    tree_weights = export_random_forest_classier(clf)
    export(tree_weights, params, model_dir, X, y)
    import_model(model_dir, params, X, y)
    # graph = RandomForestInferenceGraphs(params, tree_weights)
    # with tf.Graph().as_default() as g:
    #     dx = tf.placeholder(tf.float32, [None, 4])
    #     graph = RandomForestInferenceGraphs(params, tree_weights)
    #     logits, tree_paths, regression_variance = graph.inference_graph(dx)
    #     init_vars = tf.group(tf.global_variables_initializer(),  resources.initialize_resources(resources.shared_resources()))
    #     with tf.Session() as sess:
    #         sess.run(init_vars)
    #         print sess.run(logits, feed_dict={dx: X})


if __name__ == "__main__":
    main()
