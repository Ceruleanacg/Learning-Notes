import numpy as np


def relu(x):
    return min(x, 1e2) if x > 0 else 0.0


def grad_relu(x):
    return 1.0 if x > 0 else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + np.power(np.e, min(-x, 1e2)))


def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def grad_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def linear(x):
    return x


def grad_linear(x):
    return 1.0


def softmax(x):
    x_copy = x.copy()
    a = np.exp(x_copy - np.max(x_copy, axis=1, keepdims=True))
    z = np.sum(a, axis=1, keepdims=True)
    return a / z


def mean_square_error(y, label):
    return np.mean(np.sqrt(np.sum(np.power(y - label, 2))))


def grad_mean_square_error(y, label):
    return label - y


def softmax_cross_entropy(y, label):
    return np.mean(np.sum(label * -np.log(softmax(y) + 1e-100)))


def grad_softmax_cross_entropy(y, label):
    return label - y

