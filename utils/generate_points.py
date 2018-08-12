import numpy as np


def generate_points(mean, sigma, n):
    return np.random.multivariate_normal(mean, [[sigma, 0], [0, sigma]], size=n)


def generate_2_class_points(means=None, sigmas=None, n=100):
    if sigmas is None:
        sigmas = [1, 2]
    if means is None:
        means = [[5, 5], [10, 10]]
    return [
        generate_points(means[0], sigmas[0], n),
        generate_points(means[1], sigmas[1], n)
    ]
