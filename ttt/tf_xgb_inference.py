import numpy as np
import tensorflow as tf
from ttt.tf_utils import tree_to_leaf, leaf_to_value


def df_to_tree(df):
    meta_df = df.copy()
    children_left = np.array([-1 if not isinstance(r, str) else int(
        r.split('-')[1]) for r in meta_df['Yes'].values])
    children_right = np.array([-1 if not isinstance(r, str) else int(
        r.split('-')[1]) for r in meta_df['No'].values])

    features = np.array([-2 if r == 'Leaf' else int(r.strip('f'))
                         for r in meta_df['Feature'].values])
    thresholds = meta_df['Split'].fillna(-2).values
    meta_df.loc[meta_df['Feature'] != 'Leaf', 'Gain'] = -1
    values = meta_df['Gain'].values
    return features, thresholds, children_left, children_right, values


def model_to_tree_dfs(xgb_model):
    booster = xgb_model.get_booster()
    df = booster.trees_to_dataframe()
    for tree_id in df['Tree'].unique():
        yield df[df['Tree'] == tree_id]


def tree_df_to_value(input_, df):
    feature, threshold, left, right, values = df_to_tree(df)
    leaf = tree_to_leaf(input_,  feature, threshold, left, right)
    return leaf_to_value(leaf, values)


class InferenceBase(object):

    def __init__(self, clf):
        self.clf = clf


class TreeRegressionInference(InferenceBase):
    def _predict(self, input_):
        return [tree_df_to_value(input_, tree_df)
                for tree_df in model_to_tree_dfs(self.clf)]

    def predict(self, input_):
        predicted = self._predict(input_)
        return self.clf.base_score+tf.reduce_sum(predicted, axis=0)


class TreeClassificationInference(InferenceBase):

    def predict_proba(self, input_):
        predicted = self._predict(input_)
        if self.clf.n_classes_ > 2:
            predicted = tf.reshape(predicted, (-1, self.clf.n_classes_))
            predicted = tf.softmax(predicted, axis=1)
        return tf.reduce_mean(predicted, axis=0)

    def predict(self, input_):
        prob = self.predict_proba(input_)
        return tf.argmax(prob, axis=-1)
        #return tf.gather(self.clf.classes_, pred)
