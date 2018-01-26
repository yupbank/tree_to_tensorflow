from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tree_to_tensorflow.sklearn_exporter import export_random_forest_classier
from tree_to_tensorflow.tf_helper import tree_weights_into_proto
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
import tensorflow as tf
import numpy as np


def export(clf, est, X, y):
    tree_weights = map(tree_weights_into_proto, 
                        export_random_forest_classier(clf))
    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        dy = tf.placeholder(tf.int32, [None])
        pred = est._model_fn(dx, dy, 'infer')
        classes = pred.predictions['classes']

        g.clear_collection('resources')
        tree_variables = filter(lambda r: isinstance(r, tf.contrib.tensor_forest.python.ops.model_ops.TreeVariableSavable), g.get_collection('saveable_objects'))
        for i in tree_variables:
            g.add_to_collection('to_keep', i)
        g.clear_collection('saveable_objects')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            ops = [i.restore(tf.constant(j, shape=[1]), None)
                        for i, j in zip(tree_variables, tree_weights)]
            sess.run(ops)
            pred = sess.run(classes, feed_dict={dx: X})
            print 'tensorflow', classification_report(y, pred)
            print 'tensorflow', classification_report(y, np.where(pred==1, 0, 1))
            saver = tf.train.Saver()
            saver.save(sess, est.model_dir)

def import_model(model_dir, est, X, y):
    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        dy = tf.placeholder(tf.int32, [None])
        pred = est._model_fn(dx, dy, 'infer')
        classes = pred.predictions['classes']

        g.clear_collection('resources')
        tree_variables = filter(lambda r: isinstance(r, tf.contrib.tensor_forest.python.ops.model_ops.TreeVariableSavable), g.get_collection('saveable_objects'))
        for i in tree_variables:
            g.add_to_collection('to_keep', i)
        g.clear_collection('saveable_objects')
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
            pred = sess.run(classes, feed_dict={dx: X})
            print 'tensorflow', classification_report(y, pred)


def main():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    model_dir = 'tmp/one'
    clf = RandomForestClassifier()
    clf.fit(X, y)
    print 'sklearn', classification_report(y, clf.predict(X))
    graph_builder_class = tensor_forest.RandomForestGraphs
    params = tensor_forest.ForestHParams(
        num_classes=2,
        num_features=4,
        num_trees=10,
        max_nodes=100)
    est = random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class, model_dir=model_dir)
    export(clf, est, X, y)
    import_model(model_dir, est, X, y)
    #with tf.Graph().as_default() as g:
    #    with tf.Session() as sess:
    #        pred = est._model_fn(dx, dy, 'infer')
    #        saver.restore(sess, est.model_dir)
    #        classes = pred.predictions['classes']
    #        tensorflow_pred = sess.run(classes, feed_dict={dx: X})
    #        print 'tensorflow', classification_report(y, tensorflow_pred)


if __name__ == "__main__":
    main()
