from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pytest

import ttt.weight_utils as wutils


@pytest.fixture
def weight():
    root = wutils.binary_node(node_id=0, feature_id=0, threshold=0, left_child_id=1, right_child_id=2)
    left = wutils.leaf_node(node_id=1, value=[0.0, 1.0])
    right = wutils.leaf_node(node_id=2, value=[1.0, 0.0])
    return wutils.tree_model([root, left, right])


@pytest.fixture
def stat():
    return wutils.fertile_stat([wutils.node_stat(2, 1), wutils.node_stat(1, 1)])


@pytest.fixture
def postive_x():
    return [-1]


@pytest.fixture
def pred_positive():
    return [{'floatValue': 0.0}, {'floatValue': 1.0}]


@pytest.fixture
def negative_x():
    return [1]


@pytest.fixture
def pred_negative():
    return [{'floatValue': 1.0}, {'floatValue': 0.0}]


@pytest.fixture
def dataset():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X = np.asarray(X, dtype=np.float32)
    return X, y


@pytest.fixture
def clf(dataset):
    clf = RandomForestClassifier()
    clf.fit(*dataset)
    return clf
