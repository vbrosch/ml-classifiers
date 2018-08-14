"""
    Perform a principal component analysis (dimensionality reduction on a set)
"""
import numpy as np

from lina.linear_algebra import transpose, init_matrix, matrix_sub, dot, matrix_multiply, matrix_add
from lina.vectors import to_vec, vector_init, vector_multiply, vector_add, vector_dim


def get_dimensionality():
    return vector_dim(samples[0])


def get_mean_vector():
    norm = 1 / len(samples)

    mean = vector_init(dim=get_dimensionality())

    for s in samples:
        x = vector_multiply(s, norm)
        mean = vector_add(mean, x)

    return mean


def get_covariance_matrix(mean):
    n = 1 / len(samples)

    mean_t = transpose(mean)

    size_cov = get_dimensionality()
    cov = init_matrix(nrows=size_cov, ncols=size_cov)

    for y_i in samples:
        y_t = transpose(y_i)
        y1 = matrix_sub(y_t, mean_t)
        y2 = matrix_sub(y_i, mean)

        y3 = dot(y1, y2)
        y4 = matrix_multiply(y3, n)

        cov = matrix_add(cov, y4)

    return cov


def get_eigenvectors(cov):
    cov_mat = np.matrix(cov)
    _, v = np.linalg.eig(cov_mat)
    return [to_vec(x) for x in v.tolist()]


def principal_component_analysis():
    mean = get_mean_vector()
    print("Mean: {}".format(mean))

    cov = get_covariance_matrix(mean)
    print("Cov: {}".format(cov))

    eigenv = get_eigenvectors(cov)
    print("EigenV: {}".format(eigenv))

    return [dot(x, transpose(eigenv[0])) for x in samples]


samples = [[2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [3, 4], [5, 2]]
samples = [to_vec(x) for x in samples]

new_samples = principal_component_analysis()

print("Old samples: {}\n".format(samples))

print("New samples: {}".format(new_samples))