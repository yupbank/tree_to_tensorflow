from itertools import repeat


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
