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
    x[x > 1e2] = 1e2
    ep = np.power(np.e, x)
    return ep / np.sum(ep)


def mean_square_error(y, label):
    return np.mean(np.sqrt(np.sum(np.power(y - label, 2))))


def softmax_cross_entropy(y, label):
    return np.mean(-np.sum(np.multiply(label, np.log(softmax(y) + 1e-100))))
