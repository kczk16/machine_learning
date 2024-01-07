import numpy as np
import pytest
from sklearn.linear_model import Ridge


class RidgeRegr:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, Y):
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta (approximately) minimising quadratic loss function L with Ridge penalty,
        # using an iterative method.
        c = 0.00001
        n, m = X.shape
        ones = np.ones(n)
        X2 = np.concatenate((ones[:, np.newaxis], X), axis=1)
        theta = np.zeros((m + 1))
        nabla = np.zeros(m + 1)
        nabla[0] = 1
        while sum(abs(nabla)) > 0.00001:
            wek = Y - np.dot(X2, theta)
            for i in range(m+1):
                if i == 0:
                    dL_dtheta0 = 2 * np.sum(wek) * (-1)
                    nabla[i] = dL_dtheta0
                else:
                    dL_dtheta_j = -2 * np.dot(wek, (X2[:, i]).T) + 2 * self.alpha * theta[i]
                    nabla[i] = dL_dtheta_j
            theta = theta - c * nabla
        self.theta = theta.T

        return self

    def predict(self, X):
        # input:
        #  X = np.array, shape = (k, m)
        # returns:
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, m = X.shape
        ones = np.ones(k)
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)
        Y = np.dot(X, self.theta)
        return Y


def test_RidgeRegressionInOneDim():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])
    X_test = np.array([1, 2, 10]).reshape((3, 1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)


def test_RidgeRegressionInThreeDim():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])
    X_test = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)