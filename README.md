# Covert Tree Models to Tensorflow Tree.

---

[![CircleCI](https://circleci.com/gh/yupbank/tree_to_tensorflow/tree/master.svg?style=svg)](https://circleci.com/gh/yupbank/tree_to_tensorflow/tree/master)

# The Goal is to have one unified tree runtime

	* Convert a spark Tree/Forest into Tensorflow Tree/Forest Model.

	* Convert a sciki-learn Tree/Forest into Tensorflow Tree/Forest Model.


### Example

Convert a fitted `sklearn.random_forest_classifier` to `tensorflow.random_forest_estimator`

All you need to do is pass your desired `model_dir`, `'./tmp'` in  this example and a fitted classifier.


```python
    
from ttt import export_sklearn_rf

    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y)
    tf_estimator = export_sklearn_rf(clf, 'tmp')
    
    pred = tf_estimator.evaluate(X, y)
```

