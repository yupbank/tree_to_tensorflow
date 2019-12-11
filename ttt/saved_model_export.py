import logging
import tensorflow as tf
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ttt.tf_tree_inference import TreeRegressionInference, TreeClassificationInference, ForestClassifierInference, ForestRegressorInference


def model_fn(clf, input_):
    if isinstance(clf, DecisionTreeRegressor):
        return TreeRegressionInference(clf).predict(input_)
    elif isinstance(clf, DecisionTreeClassifier):
        return TreeClassificationInference(clf).predict(input_)
    elif isinstance(clf, RandomForestClassifier):
        return ForestClassifierInference(clf).predict(input_)
    elif isinstance(clf, RandomForestRegressor):
        return ForestRegressorInference(clf).predict(input_)
    else:
        raise Exception("Not recognized model")


def export_deicison_tree(clf, inputs, output_dir, model_version=1):
    assert len(inputs.values()) == 1, "only one input is supported"
    input_name, input_placeholder = next(iter(inputs.items()))
    with input_placeholder.graph.as_default():
        output = model_fn(clf, input_placeholder)
        with tf.Session() as sess:
            # Export inference model.
            output_path = os.path.join(
                tf.compat.as_bytes(output_dir),
                tf.compat.as_bytes(str(model_version)))

            logging.info('Exporting trained model to %s' % output_path)
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            # Build the signature_def_map.
            inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
                input_placeholder)
            output_tensor_info = tf.saved_model.utils.build_tensor_info(
                output)
            signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        input_name: inputs_tensor_info,
                    },
                    outputs={'predict': output_tensor_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            logging.info(
                'Successfully exported model to %s' % output_dir)
