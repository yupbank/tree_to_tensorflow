from itertools import repeat

def node_stat(node_id, depth):
    return {'depth': depth, 'nodeId': node_id}


def fertile_stat(node_stats):
    return {'nodeToSlot': node_stats}


def binary_node(node_id, feature_id, threshold, left_child_id, right_child_id):
    payload = {
                'binaryNode': {
                                'inequalityLeftChildTest':
                                {
                                    'featureId': {'id': str(feature_id)},
                                    'threshold': {'floatValue': threshold}
                                },
                                'leftChildId': left_child_id,
                                'rightChildId': right_child_id
                            }
             }
    if node_id:
        payload['nodeId'] = node_id
    return payload


def leaf_node(node_id, value):
    value_payload = [{i:j} for i, j in zip(repeat('floatValue'), value)]
    return {
                'leaf':
                    {
                        'vector':
                            {'value': value_payload}
                    },
                 'nodeId': node_id
            }


def tree_model(nodes):
    return {'decisionTree': {'nodes': sorted(nodes, key=lambda r: r.get('nodeId', 0))}}


def stat_from_weight(tree_weight):
    stats = []
    nodes = {}
    for node in tree_weight['decisionTree']['nodes']:
        node_id = node.get('nodeId', 0)
        nodes[node_id] = node
    start_node = 0
    depth = 0
    stack = [(start_node, depth)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node = nodes[node_id]
        if 'binaryNode' in node:
            node = node['binaryNode']
            depth += 1
            stack.append((node['leftChildId'], depth))
            stack.append((node['rightChildId'], depth))
        else:
            stats.append(node_stat(node_id, depth))
    return fertile_stat(stats)

def _predict(x, weight, path):
    nodes = weight['decisionTree']['nodes']
    start_node = 0
    paths = []
    while True:
        node = nodes[start_node]
        if path:
            paths.append(node)
        if 'binaryNode' in node:
            node = node['binaryNode']
            feature = node['inequalityLeftChildTest']
            feature_id = int(feature['featureId']['id'])
            threshold = feature['threshold']['floatValue']
            feature_value = x[feature_id]
            if feature_value <= threshold:
                start_node = node['leftChildId']
            else:
                start_node = node['rightChildId']
        if 'leaf' in node:
            node = node['leaf']
            prediction = node['vector']['value']
            prediction = [i['floatValue'] for i in prediction]
            if not path:
                return prediction
            else:
                return prediction, paths

def predict(X, tree_weights, path=False):
    nodes = tree_weights['decisionTree']['nodes']
    for x in X:
        yield _predict(x, tree_weights, path)
