# Covert Tree Models to Tensorflow Tree.

---

[![CircleCI](https://circleci.com/gh/yupbank/tree_to_tensorflow/tree/master.svg?style=svg)](https://circleci.com/gh/yupbank/tree_to_tensorflow/tree/master)
[![PyPI version](https://badge.fury.io/py/TFTree.svg)](https://badge.fury.io/py/TFTree)

# The Goal is to have one unified tree runtime

	* Convert a xgboost Tree/Forest into Tensorflow Graph.

	* Convert a sciki-learn Tree/Forest into Tensorflow Graph.


### Example

Convert fitted 

- `sklearn.DecisionTreeClassifier` 
- `sklearn.DecisionTreeRegressor`
- `sklearn.RandomForestRegressor`
- `sklearn.RandomForestClassifier`
- `xgboost.XGBClassifier`
- `xgboost.XGBRegressor`

to `tensorflow.saved_model`

All you need to do is pass your desired `model_dir`, `'./tmp'` in  this example and a fitted classifier.


```python
    
    from ttt import export_decision_tree

    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y)
    features = {'features': tf.placeholder(tf.float64, [None, X.shape[1]])}
    export_decision_tree(clf, features, 'sklearn_saved_model')

    xgb_model = xgboost.XGBClassifier().fit(X, y)
    export_decision_tree(xgb_model, features, 'xgb_saved_model')
    
```

And then you can server this model with tf/serving using 'saved_model'
