from sklearn.datasets import make_classification
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np


X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

def build_estimator(model_dir):
  """Build an estimator."""
  params = tensor_forest.ForestHParams(
      num_classes=2,
      num_features=4,
      num_trees=10,
      max_nodes=100)
  graph_builder_class = tensor_forest.RandomForestGraphs
  return random_forest.TensorForestEstimator(
      params, graph_builder_class=graph_builder_class, model_dir=model_dir)


est = build_estimator(model_dir='./tmp')


train_input_fn = numpy_io.numpy_input_fn(
  x={'images': np.asarray(X, dtype=np.float32)},
  y=y,
  batch_size=X.shape[0],
  num_epochs=None,
  shuffle=True)

est.fit(input_fn=train_input_fn, steps=None)
