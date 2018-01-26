
from sklearn.ensemble import RandomForestClassifier
import weight_utils as wutil
import tf_helper


def extract_weights(estimator):
    nodes = []
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
            nodes.append(wutil.binary_node(node_id, feature[node_id], threshold[node_id], children_left[node_id], children_right[node_id]))
            stack.extend([children_left[node_id], children_right[node_id]])
        else:
            nodes.append(wutil.leaf_node(node_id, value[node_id].ravel()))
    return wutil.base_model(nodes)



def export_random_forest_classier(rf):
    assert isinstance(rf, RandomForestClassifier), 'only scikit-learn random forest classifier supported'
    assert rf.estimators_ is not None or len(rf.estimators_) > 0, 'you have to fit it'
    tree_weights = []
    for tree in rf.estimators_:
        tree_weights.append(extract_weights(tree))
    return  tree_weights
