
from sklearn.ensemble import RandomForestClassifier
import weight_utils as wutil
import tf_helper


def extract_weights(estimator):
    nodes = []
    stats = []
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    value = estimator.tree_.value
    threshold = estimator.tree_.threshold
    stack = [(0, -1)]
    depth = -1
    while len(stack) > 0:
        (node_id, depth) = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            nodes.append(wutil.binary_node(node_id, feature[node_id], threshold[node_id], children_left[node_id], children_right[node_id]))
            stack.extend([(children_left[node_id], depth+1), (children_right[node_id], depth+1)])
        else:
            nodes.append(wutil.leaf_node(node_id, value.take([node_id], axis=0).ravel()))
            stats.append(wutil.node_stats(node_id, depth))
    return wutil.base_model(sorted(nodes, key=lambda r: r.get('nodeID'))), wutil.fertile_stats(stats)



def export_random_forest_classier(rf):
    assert isinstance(rf, RandomForestClassifier), 'only scikit-learn random forest classifier supported'
    assert rf.estimators_ is not None or len(rf.estimators_) > 0, 'you have to fit it'
    tree_weights = []
    stats = []
    for tree in rf.estimators_:
        tree_weight, stat = extract_weights(tree)
        tree_weights.append(tree_weight)
        stats.append(stat)
    return  tree_weights, stats
