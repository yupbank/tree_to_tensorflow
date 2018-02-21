import logging

from google.protobuf.json_format import ParseDict, MessageToDict
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
from tensorflow.contrib.tensor_forest.proto import tensor_forest_params_pb2 as _params_proto
from tensorflow.contrib.tensor_forest.python import tensor_forest

try:
    ### already merged in upstream https://github.com/tensorflow/tensorflow/pull/16566
    from tensorflow.contrib.tensor_forest.proto import fertile_stats_pb2 as _stats_proto
except Exception:
    import ttt.fertile_stats_pb2 as _stats_proto


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


#####
# Below Code already push to upstream https://github.com/tensorflow/tensorflow/pull/17070
#####

class TreeVariables(tensor_forest.TreeTrainingVariables):
    def __init__(self, params, tree_num, training, tree_config='', tree_stat=''):
        if (not hasattr(params, 'params_proto') or
                not isinstance(params.params_proto,
                               _params_proto.TensorForestParams)):
            params.params_proto = tensor_forest.build_params_proto(params)

        params.serialized_params_proto = params.params_proto.SerializeToString()
        
        self.stats = tensor_forest.stats_ops.fertile_stats_variable(
                params, tree_stat, self.get_tree_name('stats', tree_num))

        self.tree = tensor_forest.model_ops.tree_variable(
                params, tree_config, self.stats, self.get_tree_name('tree', tree_num))


class ForestVariables(tensor_forest.ForestTrainingVariables):
    def __init__(self, params, device_assigner, training=True,
            tree_variables_class=TreeVariables, tree_configs=None, tree_stats=None):
        self.variables = []
        self.device_dummies = []

        if tree_configs is not None:
            assert len(tree_configs) == params.num_trees
        if tree_stats is not None:
            assert len(tree_stats) == params.num_trees

        with tensor_forest.ops.device(device_assigner):
            for i in range(params.num_trees):
                self.device_dummies.append(tensor_forest.variable_scope.get_variable(
                    name='device_dummy_%d' % i, shape=0))
        for i in range(params.num_trees):
            with tensor_forest.ops.device(self.device_dummies[i].device):
                kwargs = {}
                if tree_configs is not None:
                    kwargs.update(dict(tree_config=tree_configs[i]))
                if tree_stats is not None:
                    kwargs.update(dict(tree_stat=tree_stats[i]))

                self.variables.append(tree_variables_class(params, i, training, **kwargs))


class RandomForestGraphs(tensor_forest.RandomForestGraphs):
    def __init__(self,
            params,
            tree_configs=None,
            tree_stats=None,
            device_assigner=None,
            variables=None,
            tree_variables_class=TreeVariables,
            tree_graphs=None,
            training=True):
        self.params = params
        self.device_assigner = (
            device_assigner or tensor_forest.framework_variables.VariableDeviceChooser())
        logging.info('Constructing forest with params = ')
        logging.info(self.params.__dict__)
        self.variables = variables or ForestVariables(
                self.params, device_assigner=self.device_assigner, training=training,
                tree_variables_class=tree_variables_class, tree_configs=tree_configs, tree_stats=tree_stats)
        tree_graph_class = tree_graphs or tensor_forest.RandomTreeGraphs
        self.trees = [
                    tree_graph_class(self.variables[i], self.params, i)
                    for i in range(self.params.num_trees)
                ]
