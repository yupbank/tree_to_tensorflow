from tensorflow.contrib.tensor_forest.proto import fertile_stats_pb2 as _stats_proto
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.proto import tensor_forest_params_pb2 as _params_proto
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
import logging

from google.protobuf.json_format import ParseDict, MessageToDict
from tensorflow.contrib.tensor_forest import tensor_forest

RandomForestGraphs = tensor_forest.RandomForestGraphs


def weight_dict_to_proto(weight):
    if weight:
        return ParseDict(weight, _tree_proto.Model()).SerializeToString()
    else:
        return weight


def weight_proto_to_dict(weight_proto):
    model = _tree_proto.Model()
    model.ParseFromString(weight_proto)
    return MessageToDict(model)


def stat_dict_to_proto(stat):
    if stat:
        return ParseDict(stat, _stats_proto.FertileStats()).SerializeToString()
    else:
        return stat


def stat_proto_to_dict(stat_proto):
    stat = _stats_proto.FertileStats()
    stat.ParseFromString(stat_proto)
    return MessageToDict(stat)


def path_proto_to_dict(path_proto):
    model = _stats_proto.TreePath()
    model.ParseFromString(path_proto)
    return MessageToDict(model)
