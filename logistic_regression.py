"""
    Perform a linear regression on a test data set
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from utils.generate_points import generate_2_class_points

w: np.ndarray = None
lr = 0.1
max_iter = 15000


def fit(x: np.ndarray, y: np.ndarray, should_add_intercept: bool = True):
    if should_add_intercept:
        x = add_intercept(x)

    global w
    w = np.zeros(x.shape[1])

    for iter in range(max_iter):
        grad = gradient(x, y)

        w_old = w
        w_new = lr * grad
        w = w_old - w_new

        loss = cross_entropy_loss(x, y)

        print("loss={}, iter: {}".format(loss, iter))

        converged_comp = [1 for j in range(w.shape[1]) if math.isclose(w_old.item(j), w.item(j))]

        if len(converged_comp) == w.shape[1]:
            print("Converged at iter {}".format(iter))
            break


def add_intercept(x) -> np.ndarray:
    intercept = np.ones((x.shape[0], 1))
    return np.concatenate((intercept, x), axis=1)


def g(x):
    return np.asscalar(sigmoid(np.dot(w, np.transpose(x))))


def sigmoid(x):
    sig_val = 1 / (1 + np.exp(-x))
    return sig_val


def cross_entropy_loss(x, y):
    loss = sum([y_n * math.log(g(x_n)) + (1 - y_n) * math.log(1 - g(x_n)) for x_n, y_n in zip(x, y)])
    return - loss / x.shape[0]


def gradient(x, y):
    grad = np.matrix([sum([(g(x_n) - y_n) * x_n[j] for x_n, y_n in zip(x, y)]) for j in range(x.shape[1])])
    grad /= x.shape[0]

    return grad


def plot_points(p, c):
    x0 = [x_n[0] for x_n in p]
    x1 = [x_n[1] for x_n in p]
    plt.plot(x0, x1, '{}o'.format(c))


def predict(x: np.matrix, should_add_intercept: bool = True):
    p_x = predict_proba(x, should_add_intercept)

    return 1 if p_x[1] >= 0.5 else 0


def predict_proba(x: np.matrix, should_add_intercept: bool = True):
    if should_add_intercept:
        x = add_intercept(x)

    p_x = g(x)
    return [1 - p_x, p_x]


points = generate_2_class_points(means=[[2, 2], [7, 7]], n=50)
c1 = points[0]
c2 = points[1]

all_points = np.concatenate((c1, c2))

plot_points(c1, 'b')
plot_points(c2, 'r')

labels = np.array(([1] * len(c1)) + ([0] * len(c2)))

fit(all_points, labels)

print("w vector: {}".format(w))

print(predict_proba(np.matrix([[2, 2]])))
print(predict_proba(np.matrix([[7, 7]])))

plt.show()
