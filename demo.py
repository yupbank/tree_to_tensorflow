from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import tree_to_tensorflow.sklearn_exporter as sklearn_export
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest

def main():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    model_dir = 'x'
    clf = RandomForestClassifier()
    clf.fit(X, y)
    graph_builder_class = sklearn_export.export_random_forest_classier(clf)
    params = tensor_forest.ForestHParams(
        num_classes=2,
        num_features=4,
        num_trees=10,
        max_nodes=100)
    est = random_forest.TensorForestEstimator(
        params, graph_builder_class=graph_builder_class, model_dir=model_dir)
    print est.predict(X)


if __name__ == "__main__":
    main()
