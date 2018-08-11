"""
    Simple implementation of a support vector machine (SVM)
"""

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers

COLORS = ['g', 'b', 'r', 'y', 'm']


def plot_data_with_labels():
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        plt.scatter(x_sub[:, 0], x_sub[:, 1], c=COLORS[li])

    slope = -w[0] / w[1]
    intercept = -b / w[1]

    start = min(x, key=lambda el: el[0])[0]
    end = max(x, key=lambda el: el[0])[0] + 1

    x_i = np.arange(start, end)
    plt.plot(x_i, x_i * slope + intercept, 'k-')

    plt.show()


def get_alphas():
    """
        Based on http://goelhardik.github.io/2016/11/28/svm-cvxopt/
    :return:
    """
    num = x.shape[0]

    k = y[:, None] * x
    k = np.dot(k, k.T)

    p = cvxopt.matrix(k)

    q = cvxopt.matrix(-np.ones((num, 1)))
    g = cvxopt.matrix(-np.eye(num))

    h = cvxopt.matrix(np.zeros(num))
    a = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    solvers.options['show_progress'] = False
    solv = cvxopt.solvers.qp(p, q, g, h, a, b)

    return solv['x']


def get_support_vectors():
    sv_i = [i for i, a in enumerate(alphas) if not a <= 1e-5]
    return [(alphas[i], x[i], y[i]) for i in sv_i]


def get_weight_vector():
    return np.round(sum([x_i[0] * x_i[2] * x_i[1] for x_i in sv]), 3)


def get_margin():
    norm = 1 / len(sv)
    margin = round(norm * sum([x_i[2] - w * x_i[1] for x_i in sv])[0], 3)

    return margin


x1 = np.array([[0, -3], [0, 3], [4, 3]])
x2 = np.array([[6, 0], [8, 0], [6, -3]])

x = np.concatenate((x1, x2))

y1 = np.ones((x1.shape[0],))
y2 = -np.ones((x2.shape[0],))

y = np.concatenate((y1, y2))

alphas = get_alphas()
print("Solved alphas: {}".format(alphas))

sv = get_support_vectors()
print("Support-Vectors: {}".format(sv))

w = get_weight_vector()
b = get_margin()

print("sgn(w* x + b) = sgn({} x + {})".format(w, b))

plot_data_with_labels()
