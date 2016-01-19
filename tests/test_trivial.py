from __future__ import division

import _nn_like


def test_backprop_11():
    print('\n=== test_backprop_11 ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_11_nonunit():
    print('\n=== test_backprop_11_nonunit ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_11_neg():
    print('\n=== test_backprop_11_neg ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [-0.6], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert -0.65 < o_after < 0.55


def test_backprop_111():
    print('\n=== test_backprop_111 ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_111_nonunit():
    print('\n=== test_backprop_111_nonunit ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_111_neg():
    print('\n=== test_backprop_111_neg ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [-0.6], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert -0.65 < o_after < 0.55


def test_backprop_111_neg_input():
    print('\n=== test_backprop_111_neg_input ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([-2])[0]
    _nn_like.backprop_deterministic([o_before], [-0.6], 1.0)
    o_after = _nn_like.forward_deterministic([-2])[0]
    print(o_before, o_after)
    assert -0.65 < o_after < 0.55


def test_backprop_121():
    print('\n=== test_backprop_121 ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 2, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_1221():
    print('\n=== test_backprop_1221 ===')
    _nn_like.bias(0)
    _nn_like.nn_like([1, 2, 2, 1])
    _nn_like.fixed_weights(0.5, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_1221_wb():
    print('\n=== test_backprop_1221_wb ===')
    _nn_like.bias(1)
    _nn_like.nn_like([1, 2, 2, 1])
    _nn_like.fixed_weights(0.3, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_221_wb():
    print('\n=== test_backprop_221_wb ===')
    _nn_like.bias(1)
    _nn_like.nn_like([2, 2, 1])
    _nn_like.fixed_weights(0.5, 1.0)
    o_before = _nn_like.forward_deterministic([0, 1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([0, 1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


if __name__ == '__main__':
    test_backprop_11()
    test_backprop_11_nonunit()
    test_backprop_11_neg()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_111_neg()
    test_backprop_111_neg()
    test_backprop_111_neg()
    test_backprop_111_neg()
    test_backprop_111_neg_input()
    test_backprop_111_neg_input()
    test_backprop_111_neg_input()
    test_backprop_111_neg_input()
    test_backprop_121()
    test_backprop_121()
    test_backprop_121()
    test_backprop_121()
    test_backprop_1221()
    test_backprop_1221()
    test_backprop_1221()
    test_backprop_1221()
    test_backprop_1221_wb()
    test_backprop_1221_wb()
    test_backprop_1221_wb()
    test_backprop_1221_wb()
    test_backprop_221_wb()
    test_backprop_221_wb()
    test_backprop_221_wb()
    test_backprop_221_wb()
