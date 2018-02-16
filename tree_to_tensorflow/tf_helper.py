from google.protobuf.json_format import ParseDict, MessageToDict
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
from tensorflow.contrib.tensor_forest.proto import tensor_forest_params_pb2 as _params_proto
from tensorflow.contrib.tensor_forest.python import tensor_forest
import fertile_stats_pb2 as _stats_proto


def tree_weight_into_proto(tree_weight):
    if tree_weight:
        return ParseDict(tree_weight, _tree_proto.Model()).SerializeToString()
    else:
        return tree_weight

def tree_proto_into_weights(tree_proto):
    model = _tree_proto.Model()
    model.ParseFromString(tree_proto)
    return MessageToDict(model)


def tree_stats_into_proto(tree_stat):
    if tree_stat:
        return ParseDict(tree_stat, _stats_proto.FertileStats()).SerializeToString()
    else:
        return tree_stat

def tree_stat_proto_into_stats(tree_stat_proto):
    stat = _stats_proto.FertileStats()
    stat.ParseFromString(tree_stat_proto)
    return MessageToDict(stat)

def tree_path_proto_into_dict(tree_path_proto):
    model = _stats_proto.TreePath()
    model.ParseFromString(tree_path_proto)
    return MessageToDict(model)

class TreeInferenceVariables(tensor_forest.TreeTrainingVariables):
    # def __init__(self, params, tree_weight, tree_stat, tree_num, training=False):
    def __init__(self, params, tree_weight, tree_num, training=False):
        if (not hasattr(params, 'params_proto') or
                not isinstance(params.params_proto,
                               _params_proto.TensorForestParams)):
            params.params_proto = tensor_forest.build_params_proto(params)

        params.serialized_params_proto = params.params_proto.SerializeToString()

        # self.stats = tensor_forest.stats_ops.fertile_stats_variable(
        #     params, tree_stats_into_proto(tree_stat), self.get_tree_name('stats', tree_num))

        self.tree = tensor_forest.model_ops.tree_variable(
            params, tree_weight_into_proto(tree_weight), None, self.get_tree_name('tree', tree_num))
            # params, tree_weight_into_proto(tree_weight), self.stats, self.get_tree_name('tree', tree_num))


class ForestInferenceVariables(tensor_forest.ForestTrainingVariables):
    # def __init__(self, params, tree_weights, tree_stats, device_assigner, training=False,
    def __init__(self, params, tree_weights, device_assigner, training=False,
                 tree_variables_class=TreeInferenceVariables):
        if tree_weights is not None:
            assert len(tree_weights) == params.num_trees
        self.variables = []
        self.device_dummies = []
        with tensor_forest.ops.device(device_assigner):
            for i in range(params.num_trees):
                self.device_dummies.append(tensor_forest.variable_scope.get_variable(
                    name='device_dummy_%d' % i, shape=0))

        # tree_weight, tree_stat = '', ''
        tree_weight = ''
        for tree_num in range(params.num_trees):
            if tree_weights is not None:
                tree_weight = tree_weights[tree_num]
            # if tree_stats is not None:
            #     tree_stat = tree_stats[tree_num]
            with tensor_forest.ops.device(self.device_dummies[tree_num].device):
                # self.variables.append(tree_variables_class(params, tree_weight, tree_stat, tree_num, training))
                self.variables.append(tree_variables_class(params, tree_weight, tree_num, training))


class RandomForestInferenceGraphs(tensor_forest.RandomForestGraphs):
    __doc__ = tensor_forest.RandomForestGraphs.__doc__

    # def __init__(self, params, tree_weights, tree_stats, device_assigner=None, tree_graphs=None, training=False):
    def __init__(self, params, tree_weights, device_assigner=None, tree_graphs=None, training=False):
        params = params.fill()
        device_assigner = (
            device_assigner or tensor_forest.framework_variables.VariableDeviceChooser())
        # variables = ForestInferenceVariables(params, tree_weights, tree_stats, device_assigner, training)
        variables = ForestInferenceVariables(params, tree_weights, device_assigner, training)
        super(RandomForestInferenceGraphs, self).__init__(params, device_assigner, variables=variables, tree_graphs=tree_graphs,
                                                 training=training)

