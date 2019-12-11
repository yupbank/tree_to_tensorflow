from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pytest


@pytest.fixture
def dataset():
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X = np.asarray(X, dtype=np.float32)
    return X, y


@pytest.fixture(params=[DecisionTreeClassifier(), RandomForestClassifier(n_estimators=4), DecisionTreeRegressor(), RandomForestRegressor(n_estimators=4)],
                ids=['DecisionTreeClassifier', 'RandomForestClassifier', 'DecisionTreeRegressor', 'RandomForestRegressor'])
def dlf(request, dataset):
    clf = request.param
    clf.fit(*dataset)
    yield clf
