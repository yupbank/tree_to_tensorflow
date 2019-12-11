import numpy as np
import tensorflow as tf


def is_regression(clf):
    return clf._estimator_type == 'regressor'


def clf_to_leaf(input_, clf):
    feature, threshold, left, right = clf.tree_.feature, clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right

    def next_node_from_previous(prev_node):
        indices = tf.gather(feature, prev_node)
        t = tf.gather(threshold, prev_node)
        row_indices = tf.cast(tf.range(tf.shape(indices)[0]), dtype=tf.int64)
        # zip row indices with column indices
        full_indices = tf.stack([row_indices, indices], axis=1)
        left_or_right = tf.gather_nd(input_, full_indices) <= t

        l = tf.gather(left, prev_node)
        r = tf.gather(right, prev_node)
        next_node = tf.where(left_or_right, l, r)
        next_is_not_leaf = tf.not_equal(
            tf.gather(left, next_node), tf.gather(right, next_node))

        return next_node, next_is_not_leaf

    def condition(prev_node):
        potential_next_node, potential_next_is_not_leaf = next_node_from_previous(
            prev_node)
        return tf.reduce_any(potential_next_is_not_leaf)

    def body(prev_node):
        potential_next_node, potential_next_is_not_leaf = next_node_from_previous(
            prev_node)
        return tf.where(potential_next_is_not_leaf, potential_next_node, prev_node)

    final = tf.while_loop(
        condition, body, [tf.zeros_like(input_[:, 0], dtype=tf.int64)], back_prop=False)
    leaf, _ = next_node_from_previous(final)
    return leaf


def clf_to_value(input_, clf):
    value = clf.tree_.value
    return tf.gather(value, clf_to_leaf(input_, clf))


class InferenceBase(object):
    def __init__(self, clf):
        self.clf = clf


class TreeRegressionInference(InferenceBase):

    def predict(self, input_):
        proba = clf_to_value(input_, self.clf)
        if self.clf.n_outputs_ == 1:
            return tf.squeeze(proba[:, 0], axis=1)
        else:
            return proba[:, :, 0]


class TreeClassificationInference(InferenceBase):
    def predict(self, input_):
        proba = clf_to_value(input_, self.clf)
        pred = tf.argmax(proba, axis=2)

        if self.clf.n_outputs_ == 1:
            return tf.gather(self.clf.classes_, pred)[:, 0]
        else:
            return tf.stack([tf.gather(self.clf.classes_[i], pred[:, i]) for i in range(self.clf.n_outputs_)], axis=1)

    def predict_prob(self, input_):
        proba = clf_to_value(input_, self.clf)

        if self.clf.n_outputs_ == 1:
            proba = tf.squeeze(proba, axis=1)
            normalizer = tf.reduce_sum(proba, axis=1, keepdims=True)
            proba = proba/normalizer
        else:
            normalizer = tf.reduce_sum(proba, axis=2, keepdims=True)
            proba = proba/normalizer

        return proba

    def predict_log_prob(self, input_):
        proba = self.predict_prob(input_)
        return tf.log(proba)


class ForestClassifierInference(InferenceBase):
    def predict_prob(self, input_):
        probs = [TreeClassificationInference(est).predict_prob(
            input_) for est in self.clf.estimators_]
        sum_probs = tf.reduce_sum(probs, axis=0)
        return sum_probs/self.clf.n_estimators

    def predict(self, input_):
        prob = self.predict_prob(input_)
        pred = tf.argmax(prob, axis=-1)

        if self.clf.n_outputs_ == 1:
            return tf.gather(self.clf.classes_, pred)
        else:
            return tf.stack([tf.gather(self.clf.classes_[i], pred[:, i]) for i in range(self.clf.n_outputs_)], axis=1)


class ForestRegressorInference(InferenceBase):
    def predict(self, input_):
        sum_probs = tf.reduce_sum([TreeRegressionInference(est).predict(
            input_) for est in self.clf.estimators_], axis=0)
        return sum_probs/self.clf.n_estimators


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.tree import DecisionTreeRegressor
    import tensorflow as tf
    import numpy as np
    from sklearn.utils import resample
    data = load_boston()
    x, y = data['data'], data['target']

    clf = DecisionTreeRegressor(random_state=10)
    clf.fit(x, y)
    data = tf.placeholder(tf.float64, [None, x.shape[1]])
    x_big = resample(x, n_samples=10000000, random_state=0)
    leafs_mine = inference_np(x_big, clf)
    leafs_sklearn = sklearn_inference(x_big, clf)
    sess = tf.InteractiveSession()
    res = inference_tf(data, clf)

    @timeit
    def tf_inference():
        rs = []
        for small in np.array_split(x_big, 50):
            rs.append(res.eval({data: small}))
        return np.hstack(rs)
    leafs_tf = tf_inference()
    np.testing.assert_allclose(leafs_mine, leafs_sklearn)
    np.testing.assert_allclose(leafs_tf, leafs_sklearn)
