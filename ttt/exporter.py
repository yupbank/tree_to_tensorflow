import tensorflow as tf
from tensorflow.contrib.tensor_forest as tensor_forest

import ttt.sklearn_exporter as sk 
from ttt.tf_helper import RandomForestGraphs, weight_proto_to_dict, path_proto_to_dict, weight_dict_to_proto


def export_sklearn_rf(clf, model_dir):
    tree_weights, tree_stats = sk.rf_to_weights_and_stats(clf)
    hparams = sk.rf_to_hparams(clf)
    params = tensor_forest.tensor_forest.ForestHParams(**hparams)
    return export_tf_estimator(tree_weights, tree_stats, params, model_dir)


def export_tf_graph(tree_weights, tree_stats, params, model_dir, feature_dimension, input_func=None):
    from tensorflow.python.ops import resources
    with tf.Graph().as_default() as g:
        if input_func is None:
            dx = tf.placeholder(tf.float32, [None, feature_dimension])
        graph = RandomForestGraphs(params, tree_weights, tree_stats)
        logits, tree_paths, regression_variance = graph.inference_graph(dx)
        init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

        with tf.Session() as sess:
            sess.run(init_vars)
            saver = tf.train.Saver()
            saver.save(sess, model_dir)


def export_tf_estimator(tree_weights, tree_stats, params, model_dir):
    graph_builder_class = partial(RandomForestGraphs, tree_weights=tree_weights, tree_stats=tree_stats)
    return tensor_forest.random_forest.TensorForestEstimator(params, model_dir=model_dir, graph_builder_class=graph_builder_class)
