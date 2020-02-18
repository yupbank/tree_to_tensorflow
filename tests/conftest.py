from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import numpy as np
import pytest


@pytest.fixture
def binary_classification_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    # X = np.asarray(X, dtype=np.float32)
    return X, y


@pytest.fixture
def single_regression_dataset():
    X, y = make_regression(
        n_samples=1000, n_features=4, n_informative=2, random_state=0, shuffle=False
    )
    X = np.asarray(X, dtype=np.float32)
    return X, y


@pytest.fixture
def multi_regression_dataset(single_regression_dataset):
    X, y = single_regression_dataset
    return X, np.vstack([y, y]).T


@pytest.fixture
def multiclass_classification_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=10,
        n_redundant=2,
        random_state=0,
        shuffle=False,
    )
    # X = np.asarray(X, dtype=np.float32)
    return X, y


@pytest.fixture
def multioutput_binary_class_dataset(binary_classification_dataset):
    X, y = binary_classification_dataset
    return X, np.vstack([y, y]).T


@pytest.fixture
def multioutput_multi_class_dataset(multiclass_classification_dataset):
    X, y = multiclass_classification_dataset
    return X, np.vstack([y, y]).T


@pytest.fixture(
    params=[
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=4),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_estimators=4),
        xgb.XGBClassifier(n_estimators=4),
        xgb.XGBRegressor(n_estimators=4),
    ],
    ids=[
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "XGBClassifier",
        "XGBRegressor",
    ],
)
def dlf(request, binary_classification_dataset):
    clf = request.param
    clf.fit(*binary_classification_dataset)
    yield clf
