import ttt.weight_utils as wutil
from collections import defaultdict

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


def tree_to_weight(tree):
    header = tree[0].strip()
    node = dict()
    root = node
    node_id = 0
    stack = []
    for line in tree[1:]:
        print line, '!!!'
        if line.startswith('If'):
            new_node = dict(id = node_id, left_confition=line)
            root['left'] = new_node
            node_id += 1
            stack.append(new_node)
        elif line.startswith('Else'):
            prev_node = stack.pop()
            prev_node.update(dict(right_condition=line))
            root['right'] = prev_node
            stack.append(prev_node)
        elif line.startswith('Predict'):
            prev_node = stack.pop()
            if 'right_condition' not in prev_node:
                prev_node['leaf'] = [line]
                stack.append(prev_node)
            else:
                print prev_node
                prev_node['leaf'].append(line)
                prev_prev_node = stack.pop()
                if 'right_condition' in prev_prev_node:
                    prev_prev_node['right_child'] = prev_node
                else:
                    prev_prev_node['left_child'] = prev_node
                stack.append(prev_prev_node)
    print stack
             

if __name__ == "__main__":
    forest = open('new_spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
