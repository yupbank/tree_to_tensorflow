import numpy as np
import tensorflow as tf


def tree_to_leaf(input_, feature, threshold, left, right):
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


def leaf_to_value(leaf_id, value):
    return tf.gather(value, leaf_id)
