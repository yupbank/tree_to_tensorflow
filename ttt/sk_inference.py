import numpy as np
import tensorflow as tf

from ttt.tf_utils import tree_to_leaf, leaf_to_value


def clf_to_leaf(input_, clf):
    feature, threshold, left, right = clf.tree_.feature, clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right
    return tree_to_leaf(input_, feature, threshold, left, right)


def clf_to_value(input_, clf):
    value = clf.tree_.value
    leaf = clf_to_leaf(input_, clf)
    return leaf_to_value(leaf, value)


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
        return tf.reduce_mean([TreeRegressionInference(est).predict(
            input_) for est in self.clf.estimators_], axis=0)
        #return sum_probs/self.clf.n_estimators
