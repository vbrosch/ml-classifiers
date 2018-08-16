import math
import random

import matplotlib.pyplot as plt

import numpy as np


def get_range(x_min, x_max, n):
    x_min = 0 if x_min is None else x_min
    x_max = math.pi * 2 if x_max is None else x_max

    return np.arange(x_min, x_max, x_max / n)


def get_point_x(x_min=None, x_max=None, n=10):
    return get_range(x_min, x_max, n)


def get_x_range(x_min=None, x_max=None, n=1000):
    return get_range(x_min, x_max, n)


def generator_fn(x):
    return math.sin(x)


def generate_points(x: np.ndarray):
    return [generator_fn(x) + (0.5 * random.randint(-1, 1) * random.random()) for x in x]


def get_points_on_line(x):
    return [generator_fn(x_n) for x_n in x]


def plot_original_fn(x):
    plt.plot(x, get_points_on_line(x), color='g')


def plot_point_samples(x, targets):
    plt.plot(x, targets, 'ro')


def get_x_matrix(x, m):
    x = [[math.pow(x_n, m_pow) for x_n in x] for m_pow in range(m + 1)]
    return np.matrix(x)


def get_b_vec(x, t, m):
    b = [[sum([t_n * math.pow(x_n, m_pow) for t_n, x_n in zip(t, x)])] for m_pow in range(m + 1)]
    return np.matrix(b)


def get_w_val(x_n, w_vec):
    return sum([w_vec * math.pow(x_n, m_pow) for m_pow, w_vec in enumerate(w_vec)])


def plot_w_fn(x, w_vec):
    y_points = [get_w_val(x_n, w_vec) for x_n in x]
    plt.plot(x, y_points, color='r')


def get_w(x: np.ndarray, t: list, m: int = 2, lam: int = 0):
    x_mat = get_x_matrix(x, m)
    x_mat_transpose = np.transpose(x_mat)

    b_vec = np.transpose(get_b_vec(x, t, m))

    x_result = np.dot(x_mat, x_mat_transpose) + (np.eye(x_mat.shape[0]) * lam)
    x_pseudoinverse = np.linalg.inv(x_result)
    return np.dot(b_vec, x_pseudoinverse).tolist()[0]
