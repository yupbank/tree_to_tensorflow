from functools import partial

from google.protobuf.json_format import ParseDict
import tensorflow as tf
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest



class RandomForestGraphs(tensor_forest.RandomForestGraphs):
    __doc__ = tensor_forest.RandomForestGraphs.__doc__

    def fill_the_trees(self, tree_weights):
        restorable = tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        restore_ops = []
        print len(tree_weights), len(restorable)
        for n, tree in enumerate(restorable[:len(tree_weights)]):
            restore_ops.append(tree.restore(tree_weights_into_proto(tree_weights[n]), None))
        # WTF...... how do i get the session!
        tf.get_default_session().run(restore_ops)

    def fit(self, *args, **kwargs):
        raise Exception('Disabled...')

def hack_graph_builder_class(cls, tree_weights):
    ss = tree_weights
    def wappered_init(*args, **kwargs):
        obj = cls(*args, **kwargs)
        if cls == RandomForestGraphs:
            obj.fill_the_trees(ss)
        return obj
    wappered_init.__doc__ == cls.__doc__
    return wappered_init

RFGraphBuilder = partial(hack_graph_builder_class, RandomForestGraphs)


def tree_weights_into_proto(tree_weight):
    return ParseDict(tree_weight, _tree_proto.Model()).SerializeToString()
