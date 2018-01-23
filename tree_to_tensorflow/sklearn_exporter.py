from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto



def export_random_forest_classier(rf):
    assert isinstance(rf, RandomForestClassifier), 'only scikit-learn random forest classifier supported'
    assert self.estimators_ is not None or len(self.estimators_) > 0, 'you have to fit it'
    params = tensor_forest.ForestHParams(
      num_classes=rf.n_classes_,
      num_features=len(rf.feature_importances_),
      num_trees=len(rf.estimators_),
      max_nodes=2**(rf.max_depth+1))
    model = _tree_proto.Model()
    return
