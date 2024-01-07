import numpy as np
import pandas as pd


# 784 columns + first column with labels
data = pd.read_csv(r'mnist_data.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# training dataset - first 41000 rows
data_train = data[0:41000].T
# first column - lables
Y_train = data_train[0]
X_train = data_train[1:n]
# we want to have values between 0 and 1
X_train = X_train / 255
_, m_train = X_train.shape

# test dataset - last 1000 rows
data_test = data[41000:].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255


# HE initialization
def initialize_parameters():
    # weight for the first layer
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    # bias for the first layer
    b1 = np.random.randn(10, 1) * np.sqrt(2)
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.random.randn(10, 1) * np.sqrt(2)
    return W1, b1, W2, b2

# activation function - first layer
def ReLU(Z):
    return np.maximum(Z, 0)

# normalized exponential function - second layer
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print('predicted: ', predictions)
    print('actual: ', Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initialize_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration number: ", i)
            predictions = get_predictions(A2)
            accu = get_accuracy(predictions, Y)
            print(accu)
    return W1, b1, W2, b2, accu


W1, b1, W2, b2, accu = gradient_descent(X_train, Y_train, 0.1, 1000)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    accu = get_accuracy(predictions, Y_test)
    return predictions, accu

test_predictions, accu_test = make_predictions(X_test, W1, b1, W2, b2)
