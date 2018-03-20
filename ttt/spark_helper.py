import ttt.weight_utils as wutil
from collections import defaultdict

# https://github.com/apache/spark/pull/20825 blocking by this

def forest_to_trees(forest):
    header = forest[0].strip()
    trees = []
    tree = []
    for line in forest[1:]:
        line = line.strip()
        if line.startswith('Tree'):
            tree = [line]
            trees.append(tree)
        else:
            tree.append(line)
    return header, trees

def create_node():
    node_id = [0]
    def _(**kwargs):
        new_node = dict(id=node_id[0], **kwargs)
        node_id[0] += 1
        return new_node
    return _


def tree_to_weight(tree):
    new_node = create_node()
    header = tree[0].strip()
    node_id = 0
    root = new_node()
    stack = [root]
    prev_line = None
    for line in tree[1:]:
        prev_node = stack.pop()
        if line.startswith('If'):
            node = new_node()
            prev_node['condition'] = [line]
            prev_node['left_child'] = node
            stack.append(prev_node)
            stack.append(node)
        elif line.startswith('Else'):
            node = new_node()
            prev_node['condition'].append(line)
            prev_node['right_child'] = node
            stack.append(node)
        elif line.startswith('Predict'):
            node = new_node(leaf=line)
            if 'left_child' not in prev_node:
                prev_node['left_child'] = node
            else:
                prev_node['right_child'] = node
    return root 

if __name__ == "__main__":
    forest = open('new_spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
    forest = open('spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
