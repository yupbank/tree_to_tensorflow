
from sklearn.ensemble import RandomForestClassifier

import ttt.weight_utils as wutil


def tree_to_weight(tree):
    nodes = []
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    value = tree.value
    threshold = tree.threshold
    stack = [0]
    while len(stack) > 0:
        node_id = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            nodes.append(wutil.binary_node(node_id, feature[node_id], threshold[node_id], children_left[node_id], children_right[node_id]))
            stack.extend([children_left[node_id], children_right[node_id]])
        else:
            nodes.append(wutil.leaf_node(node_id, value.take([node_id], axis=0).ravel()))
    return wutil.tree_model(nodes)


def rf_to_hparams(rf):
    return dict(num_classes=rf.n_classes_,
                num_features=rf.n_features_,
                num_trees=rf.n_estimators)



def rf_to_weights_and_stats(rf):
    assert isinstance(rf, RandomForestClassifier), 'only scikit-learn random forest classifier supported'
    assert rf.estimators_ is not None or len(rf.estimators_) > 0, 'you have to fit it'
    tree_weights = []
    tree_stats = []
    for tree in rf.estimators_:
        tree_weight = tree_to_weight(tree.tree_)
        tree_stat = wutil.stat_from_weight(tree_weight)
        tree_weights.append(tree_weight)
        tree_stats.append(tree_stat)
    return  tree_weights, tree_stats
