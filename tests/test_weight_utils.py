from ttt.weight_utils import stat_from_weight, predict

def test_predict(weight, postive_x, negative_x, pred_positive, pred_negative):
    actual = predict(postive_x, weight)
    assert actual == pred_positive

    actual = predict(negative_x, weight)
    assert actual == pred_negative

