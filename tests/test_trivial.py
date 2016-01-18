from __future__ import division

import _nn_like


def test_trivial_deterministic():
    _nn_like.nn_like([3, 1])
    o = _nn_like.forward_deterministic([0, 1, 2])
    print(o)
    # assert o == 1.5  # relu
    # assert 0.81 < o < 0.82  # logistic function


def test_trivial():
    _nn_like.nn_like([3, 1])
    o = sum(_nn_like.forward([0, 1, 2])[0] for _ in range(100)) / 100.0
    print(o)
    # assert 1.4 < o < 1.6  # relu
    # assert 0.6 < o < 0.7  # logistic function


def test_backprop_deterministic():
    _nn_like.nn_like([3, 1])
    o_before = _nn_like.forward_deterministic([0, 1, 2])[0]
    c = -1
    for i in range(10):
        o = _nn_like.forward_deterministic([0, 1, 2])[0]
        print(o)
        if c == -1 and 0.949 < o < 0.951:
            c = i
            print('converged at {}'.format(c))
        _nn_like.backprop_deterministic([o], [0.95], 0.2)
    o_after = _nn_like.forward_deterministic([0, 1, 2])[0]
    print(o_before, o_after, c)
    assert 0.94 < o_after < 0.96


def test_backprop_11():
    print('\n=== test_backprop_11 ===')
    _nn_like.nn_like([1, 1])
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.95], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.94 < o_after < 0.96


def test_backprop_11_nonunit():
    print('\n=== test_backprop_11_nonunit ===')
    _nn_like.nn_like([1, 1])
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [0.95], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert 0.94 < o_after < 0.96


def test_backprop_111():
    print('\n=== test_backprop_111 ===')
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_111_nonunit():
    print('\n=== test_backprop_111_nonunit ===')
    _nn_like.nn_like([1, 1, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([2])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([2])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_121():
    print('\n=== test_backprop_121 ===')
    _nn_like.nn_like([1, 2, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_1221():
    print('\n=== test_backprop_1221 ===')
    _nn_like.nn_like([1, 2, 2, 1])
    _nn_like.fixed_weights(1.2, 1.0)
    o_before = _nn_like.forward_deterministic([1])[0]
    _nn_like.backprop_deterministic([o_before], [0.6], 1.0)
    o_after = _nn_like.forward_deterministic([1])[0]
    print(o_before, o_after)
    assert 0.55 < o_after < 0.65


def test_backprop_deterministic_2layer():
    print('===========')
    _nn_like.nn_like([3, 1, 1])
    o_before = _nn_like.forward_deterministic([0, 1, 2])[0]
    c = -1
    for i in range(10):
        o = _nn_like.forward_deterministic([0, 1, 2])[0]
        print(o)
        if c == -1 and 0.949 < o < 0.951:
            c = i
            print('converged at {}'.format(c))
        _nn_like.backprop_deterministic([o], [0.6], 0.5)
    o_after = _nn_like.forward_deterministic([0, 1, 2])[0]
    print(o_before, o_after, c)
    assert 0.55 < o_after < 0.65


if __name__ == '__main__':
    # test_trivial_deterministic()
    # test_trivial()
    # test_backprop_11()
    # test_backprop_11_nonunit()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_111_nonunit()
    test_backprop_121()
    test_backprop_121()
    test_backprop_121()
    test_backprop_121()
    test_backprop_1221()
    test_backprop_1221()
    test_backprop_1221()
    test_backprop_1221()
    # test_backprop_deterministic()
    # test_backprop_deterministic_2layer()
