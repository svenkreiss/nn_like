from __future__ import division

import _nn_like
import numpy as np


def test_xor(samples=10000, low=0.1, high=0.9, test_samples=1000):
    _nn_like.nn_like([2, 2, 1])
    _nn_like.print_connections()

    X_all = np.random.choice([low, high],
                             size=samples*2).reshape((samples, 2))
    y_all = np.array([[high] if a != b else [low]
                      for a, b in X_all])
    print(X_all[:20], y_all[:20])

    # test data
    X_test, y_test = (X_all[:test_samples], y_all[:test_samples])
    error = sum((y[0]-_nn_like.forward_deterministic(X)[0])**2
                for X, y in zip(X_test, y_test)) / test_samples
    print(error)

    # train
    for X, y in zip(X_all, y_all):
        o = _nn_like.forward_deterministic(X)
        _nn_like.backprop_deterministic(o, y, 0.1)
    _nn_like.forward_deterministic([low, low])
    _nn_like.print_connections()
    _nn_like.print_states()

    # test
    error = sum((y[0]-_nn_like.forward_deterministic(X)[0])**2
                for X, y in zip(X_test, y_test)) / test_samples
    print(error)

    # print the 4 cases
    print('00', _nn_like.forward_deterministic([low, low]))
    print('01', _nn_like.forward_deterministic([low, high]))
    print('10', _nn_like.forward_deterministic([high, low]))
    print('11', _nn_like.forward_deterministic([high, high]))


if __name__ == '__main__':
    test_xor()
