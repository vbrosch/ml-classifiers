"""
    Example of performing a ridge regression over a data set
    => Solve the equation w = b(XX^t + λI)^-1
"""
import math
import random

import numpy as np
import matplotlib.pyplot as plt

m = 8
n = 10
n_max = 1000
x_min = 0
x_max = math.pi * 2
x_full_range = np.arange(x_min, x_max, x_max / n_max)
x_range = np.arange(x_min, x_max, x_max / n)


def generator_fn(x):
    return math.sin(x)


def generate_points():
    return [generator_fn(x) + (0.5 * random.randint(-1, 1) * random.random()) for x in x_range]


def get_points_on_line():
    return [generator_fn(x) for x in x_full_range]


def plot_original_fn():
    plt.plot(x_full_range, get_points_on_line(), color='g')


def plot_point_samples():
    plt.plot(x_range, targets, 'ro')


def get_x_matrix():
    x = [[math.pow(x_n, m_pow) for x_n in x_range] for m_pow in range(m + 1)]
    return np.matrix(x)


def get_b_vec():
    b = [[sum([t_n * math.pow(x_n, m_pow) for t_n, x_n in zip(targets, x_range)])] for m_pow in range(m + 1)]
    return np.matrix(b)


def get_w_val(x_n, w_vec):
    return sum([w_vec * math.pow(x_n, m_pow) for m_pow, w_vec in enumerate(w_vec)])


def plot_w_fn(w_vec):
    y_points = [get_w_val(x_n, w_vec) for x_n in x_full_range]
    plt.plot(x_full_range, y_points, color='r')


targets = generate_points()

x_mat = get_x_matrix()
b_vec = np.transpose(get_b_vec())

x_mat_transpose = np.transpose(x_mat)

for i, lam in enumerate([0, 0.25, 0.5, 1, 10, 15]):
    plt.subplot(2, 3, i + 1)
    plt.title("$\lambda = {}$".format(lam))
    plt.axis([x_min, x_max, -1.5, 1.5])

    x_result = np.dot(x_mat, x_mat_transpose) + (np.eye(x_mat.shape[0]) * lam)

    x_pseudoinverse = np.linalg.inv(x_result)

    w = np.dot(b_vec, x_pseudoinverse).tolist()[0]

    print("lambda = {}, w = {}".format(lam, w))

    plot_original_fn()
    plot_point_samples()
    plot_w_fn(w)

plt.show()
