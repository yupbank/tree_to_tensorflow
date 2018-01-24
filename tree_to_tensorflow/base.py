import tensorflow  as tf
from tensorflow.contrib.tensor_forest import RandomForestGraphs as OriginRandomForestGraphs



class RandomForestGraphs(OriginRandomForestGraphs):
    def fill_the_trees(tree_data_protos):
        for n, tree in enumerate(self.trees):
            tree.restore(tree_data_protos[n], None)
