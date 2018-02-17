import pytest
import sys
print(sys.path)
import ttt.weight_utils as wutils


@pytest.fixture
def weight():
    root = wutils.binary_node(node_id=0, feature_id=0, threshold=0, left_child_id=1, right_child_id=2)
    left = wutils.leaf_node(node_id=1, value=[0.0, 1.0])
    right = wutils.leaf_node(node_id=2, value=[1.0, 0.0])
    return wutils.base_model([root, left, right])

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
