import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


class LinearRegr:
    def fit(self, X, Y):
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta minimising quadratic loss function L, using an explicit formula.
        # Note: before applying the formula to X one should append to X a column with ones.
        n, m = X.shape
        ones = np.ones(n)
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)
        xx = np.dot(X.T, X)
        theta = np.dot(np.linalg.inv(xx), np.dot(X.T, Y))
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
        print(Y)
        return Y


def test_RegressionInOneDim():
    X = np.array([1, 3, 2, 5]).reshape((4, 1))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 2, 10]).reshape((3, 1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))


def test_RegressionInThreeDim():
    X = np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5]).reshape((4, 3))
    Y = np.array([2, 5, 3, 8])
    a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3]).reshape((5, 3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))
