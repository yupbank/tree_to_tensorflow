import ttt.weight_utils as wutil

def forest_to_trees(forest):
    header = forest[0].strip()
    trees = []
    tree = []
    for line in forest[1:]:
        line = line.rstrip()
        if line.strip().startswith('Tree'):
            tree = [line]
            trees.append(tree)
        else:
            tree.append(line)
    return header, trees

def tree_to_weight(tree):
    header = tree[0].strip()
    stack = []
    nodes = []
    node['child'] = current
    current = {}
    for line in tree[1:]:
        line = line.strip()
        if line.startswith('If'):
            stack.append(line)
            current['left'] = line
        elif line.startswith('Else'):
            current['right'] = line
            current['child'] = {}
            current = {}
            key = stack.pop()
            print key, line
            print "binary_node(%s, %s)"%(key, line)
        elif line.startswith('Predict'):
            print line
        else:
            print 'error'

if __name__ == "__main__":
    forest = open('../spark_tree.txt').readlines()
    header, trees = forest_to_trees(forest)
    tree_to_weight(trees[0])
