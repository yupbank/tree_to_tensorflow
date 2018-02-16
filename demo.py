from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tree_to_tensorflow.sklearn_exporter import export_random_forest_classier
from tree_to_tensorflow.tf_helper import RandomForestInferenceGraphs, tree_proto_into_weights, tree_path_proto_into_dict, tree_weight_into_proto
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import resources
import tensorflow as tf
import numpy as np

from tree_to_tensorflow.weight_utils import predict

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

def _import_model(tree_number, model_dir, params, x):
    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        processed_dense_features, processed_sparse_features, data_spec = tensor_forest.data_ops.ParseDataTensorOrDict(dx)
        graph = RandomForestInferenceGraphs(params, None)
        logits, tree_paths, regression_variance = graph.inference_graph(dx)
        dpred = graph.trees[tree_number].inference_graph(processed_dense_features, data_spec, sparse_features=processed_sparse_features)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_dir)
            pred = sess.run(dpred, feed_dict={dx:x})
            tree_paths = map(tree_path_proto_into_dict, pred.tree_paths)
            predictions = pred.predictions
            return predictions, tree_paths
            #print 'imported tensorflow', classification_report(y, np.where((pred[:, 0] - pred[:, 1]) > 0, 0, 1))

def _tf_by_tree_number(sess, graph, processed_dense_features, data_spec, sparse_features, tree_number, dx, x):
    tree = graph.trees[tree_number]
    pred = sess.run(tree.inference_graph(processed_dense_features, data_spec, sparse_features=sparse_features), feed_dict={dx:x})
    tree_paths = map(tree_path_proto_into_dict, pred.tree_paths)
    predictions = pred.predictions
    return predictions, tree_paths


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
        inference_tree_paths=True,
        max_nodes=100)
    tree_weights, tree_stats = export_random_forest_classier(clf)
    # export(tree_weights, params, model_dir, X, y)
    # import_model(model_dir, params, X, y)

    with tf.Graph().as_default() as g:
        dx = tf.placeholder(tf.float32, [None, 4])
        graph = RandomForestInferenceGraphs(params.fill(), tree_weights, tree_stats)
        logits, tree_paths, regression_variance = graph.inference_graph(dx)
        init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
        processed_dense_features, processed_sparse_features, data_spec =  tensor_forest.data_ops.ParseDataTensorOrDict(dx)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_vars)
            import ipdb; ipdb.set_trace()
            weights_by_tree_number = lambda tree_number, instance_number: predict(X[instance_number], tree_weights[tree_number], True)
            sklearn_by_tree_number = lambda tree_number, instance_number: clf.estimators_[tree_number].tree_.predict(X[instance_number:instance_number+1])
            tf_by_tree_number = lambda tree_number, instance_number: _tf_by_tree_number(sess, graph, processed_dense_features, data_spec, processed_sparse_features, tree_number, dx, X[instance_number:instance_number+1])
            #sess.run(graph.tress[0].variables.tree.graph.get_tensor_by_name('tree-1/TreeSerialize:0'))
            #print sess.run(logits, feed_dict={dx: X})
            saver.save(sess, model_dir, global_step=1)
            reader = pywrap_tensorflow.NewCheckpointReader('%s-1'%model_dir)
            saved_weights_by_tree_number = lambda tree_number, instance_number: predict(X[instance_number], tree_proto_into_weights(reader.get_tensor('tree-%s:0'%tree_number)), True)
            import_model = lambda tree_number, instance_number: _import_model(tree_number, model_dir+'-1', params, X[instance_number:instance_number+1])
            print reader.get_variable_to_shape_map()


if __name__ == "__main__":
    main()
