import numpy as np

from linear_algebra import transpose, init_matrix, matrix_add, matrix_sub, dot, matrix_multiply, matrix_determinant, \
    matrix_invert
from vectors import vector_add, vector_sub, vector_init, vector_multiply, to_vec, vector_dim


def calc_mean(y):
    if len(y) == 0:
        return []

    n = 1 / len(y)

    mean = vector_init(dim=vector_dim(y[0]))

    for y_i in y:
        mean = vector_add(mean, y_i)

    mean = vector_multiply(mean, n)

    return mean


def calc_cov_matrix(y, mean):
    n = 1 / len(y)

    mean_t = transpose(mean)

    size_cov = vector_dim(mean)
    cov = init_matrix(nrows=size_cov, ncols=size_cov)

    for y_i in y:
        y_t = transpose(y_i)
        y1 = matrix_sub(y_t, mean_t)
        y2 = matrix_sub(y_i, mean)

        y3 = dot(y1, y2)
        y4 = matrix_multiply(y3, n)

        cov = matrix_add(cov, y4)

    return cov


def __discriminant_function(p, mean, cov):
    return lambda x: np.log(p) - 0.5 * np.log(matrix_determinant(cov)) - 0.5 * (dot(dot(matrix_sub(x, mean),
                                                                                        matrix_invert(cov)),
                                                                                    transpose(matrix_sub(x, mean))))[0][
        0]


def discriminant_function(p1, p2, mean1, mean2, cov1, cov2):
    return lambda x: __discriminant_function(p1, mean1, cov1)(x) - __discriminant_function(p2, mean2, cov2)(x)


def classify(vec, df):
    if df(vec) >= 0:
        assignment = "C1"
    else:
        assignment = "C2"

    print("{} -> {}".format(vec, assignment))


p_c1 = 0.5
p_c2 = 0.5

c1 = [[2, 6], [3, 4], [3, 8], [4, 6]]
c2 = [[1, -2], [3, 0], [3, -4], [5, -2]]

c1_vec = [to_vec(x_i) for x_i in c1]
c2_vec = [to_vec(x_i) for x_i in c2]

mean_c1 = calc_mean(c1_vec)
mean_c2 = calc_mean(c2_vec)

print("Mean C1: {}".format(mean_c1))
print("Mean C2: {}".format(mean_c2))

cov_c1 = calc_cov_matrix(c1_vec, mean_c1)
cov_c2 = calc_cov_matrix(c2_vec, mean_c2)

print("Cov C1: {}".format(cov_c1))
print("Cov C2: {}".format(cov_c2))

dc_f = discriminant_function(p_c1, p_c2, mean_c1, mean_c2, cov_c1, cov_c2)

classify(dc_f, to_vec([2, 7]))
classify(dc_f, to_vec([2, 1]))
