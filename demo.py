from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from ttt import export_sklearn_rf


def main():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X = np.asarray(X, dtype=np.float32)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    tf_estimator = export_sklearn_rf(clf, 'tmp')
    pred = tf_estimator.predict(x=X)
    print [i for i in pred]

if __name__ == "__main__":
    main()
