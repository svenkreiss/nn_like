from __future__ import division

import _nn_like


def test_trivial_deterministic():
    _nn_like.nn_like([3, 1])
    o = _nn_like.forward_deterministic([0, 1, 2])
    print(o)
    # assert o == 1.5  # relu
    assert 0.81 < o < 0.82  # logistic function


def test_trivial():
    _nn_like.nn_like([3, 1])
    o = sum(_nn_like.forward([0, 1, 2])[0] for _ in range(100)) / 100.0
    print(o)
    # assert 1.4 < o < 1.6  # relu
    assert 0.6 < o < 0.7  # logistic function


def test_backprop_deterministic():
    _nn_like.nn_like([3, 1])
    o_before = _nn_like.forward_deterministic([0, 1, 2])[0]
    for _ in range(1000):
        o = _nn_like.forward_deterministic([0, 1, 2])[0]
        print(o)
        _nn_like.backprop_deterministic([o], [0.95], 1.0)
    o_after = _nn_like.forward_deterministic([0, 1, 2])[0]
    print(o_before, o_after)
    assert 0.94 < o_after < 0.96


if __name__ == '__main__':
    test_trivial_deterministic()
    test_trivial()
    test_backprop_deterministic()
