from itertools import repeat

from google.protobuf.json_format import ParseDict
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto



def binary_node(node_id, feature_id, threshold, left_child_id, right_child_id):
    payload = {
                'binaryNode': {
                                'inequalityLeftChildTest':
                                {
                                    'featureId': {'id': unicode(feature_id)},
                                    'threshold': {'floatValue': threshold}
                                },
                                'leftChildId': left_child_id,
                                'rightChildId': right_child_id
                            }
             }
    if node_id:
        payload['nodeId'] = node_id
    return payload


def leaf_node(node_id, value):
    value_payload = [{i:j} for i, j in zip(repeat('floatValue'), value)]
    return {
                'leaf':
                    {
                        'vector':
                            {'value': value_payload}
                    },
                 'nodeId': node_id
            }


def export_tree_into_dict(estimator):
    nodes = []
    base_model = {'decisionTree': {'nodes': nodes}}
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    value = estimator.tree_.value
    threshold = estimator.tree_.threshold
    stack = [0]
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            nodes.append(binary_node(node_id, feature[node_id], threshold[node_id], children_left[node_id], children_right[node_id]))
            stack.extend([children_left[node_id], children_right[node_id]])
        else:
            nodes.append(leaf_node(node_id, value[node_id].ravel()))
    return base_model


def tree_dict_into_proto(tree_dict):
    return ParseDict(tree_dict, _tree_proto.Model())

def export_random_forest_classier(rf):
    assert isinstance(rf, RandomForestClassifier), 'only scikit-learn random forest classifier supported'
    assert rf.estimators_ is not None or len(rf.estimators_) > 0, 'you have to fit it'
    tree_proto_strs = []
    for tree in rf.estimators_:
        tree_proto_strs.append(tree_dict_into_proto(export_tree_into_dict(tree)).SerializeToString())
    #params = tensor_forest.ForestHParams(
    #  num_classes=rf.n_classes_,
    #  num_features=len(rf.feature_importances_),
    #  num_trees=len(rf.estimators_),
    #  max_nodes=2**(rf.max_depth+1))
    return tree_proto_strs
