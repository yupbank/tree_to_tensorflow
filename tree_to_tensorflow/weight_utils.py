from itertools import repeat

def fertile_stats(node_stats):
    return {'nodeToSlot': node_stats}

def node_stats(depth, node_id):
    return {'depth': depth, 'nodeId': node_id}

def base_model(nodes):
    return {'decisionTree': {'nodes': nodes}}


def binary_node(node_id, feature_id, threshold, left_child_id, right_child_id):
    payload = {
                'binaryNode': {
                                'inequalityLeftChildTest':
                                {
                                    'featureId': {'id': unicode(feature_id)},
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



def predict(x, tree_weights, path=False):
    nodes = {}
    for node in tree_weights['decisionTree']['nodes']:
        node_id = node.get('nodeId', 0)
        nodes[node_id] = node
    start_node = 0
    if path:
        paths = [nodes[start_node]]
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
            if not path:
                return prediction
            else:
                return prediction, paths
