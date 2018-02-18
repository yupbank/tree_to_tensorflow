from ttt.weight_utils import stat_from_weight, predict

def test_predict(weight, postive_x, negative_x, pred_positive, pred_negative):
    actual = predict(postive_x, weight)
    assert actual == pred_positive

    actual = predict(negative_x, weight)
    assert actual == pred_negative

def test_stat_from_weight(weight, stat):
    actual = stat_from_weight(weight)
    assert actual == stat
