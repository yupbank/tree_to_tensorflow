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
    node_id = 0
    stack = []
    root = {}
    prev_line = None
    for line in tree[1:]:
        if line.startswith('If'):
            node = dict(id = node_id, left_confition=line)
            node_id += 1
            stack.append(node)
        elif line.startswith('Else'):
            prev_node = stack.pop()
            prev_node.update(dict(right_condition=line))
            stack.append(prev_node)
        elif line.startswith('Predict'):
            node = dict(id=node_id, leaf = line)
            node_id += 1
            prev_node = stack.pop()
            if prev_line.startswith('If'):
                prev_node['left_child'] = node
            elif prev_line.startswith('Else'):
                prev_node['right_child'] = node
            else:
                print 'error', '!!'
            stack.append(prev_node)
        else:
            print 'error', '++'
        prev_line = line
    print stack
    print len(stack)
             

if __name__ == "__main__":
    forest = open('new_spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
    forest = open('spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
