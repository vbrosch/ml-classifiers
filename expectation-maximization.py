"""
    Implementation of the EM-Algorithm
"""
import math

import numpy as np
from matplotlib import pyplot

from lina.linear_algebra import matrix_sub, dot, transpose, init_matrix, matrix_add, matrix_multiply, matrix_determinant, \
    matrix_invert
from lina.vectors import to_vec, vector_dim, unwrap

# init parameters
from utils.generate_points import generate_2_class_points

max_iter = 10000
L = 2
n = 2
mixins = [0.5, 0.5]
means = [to_vec([1, 8]), to_vec([12, 12])]
covs = [[[1, 0], [0, 1]], [[2, 0], [0, 2]]]
responsibility_matrix = None


def calc_mean(l, samples, ml):
    mean = init_matrix(nrows=1, ncols=2)
    for i, y in enumerate(samples):
        mean = matrix_add(mean, matrix_multiply(y, responsibility_matrix[i][l]))
    mean = matrix_multiply(mean, 1 / ml)

    return mean


def calc_cov_matrix(samples, mean, ml, l):
    mean_t = transpose(mean)

    size_cov = vector_dim(mean)
    cov = init_matrix(nrows=size_cov, ncols=size_cov)

    for i, y_i in enumerate(samples):
        y_t = transpose(y_i)
        y1 = matrix_sub(y_t, mean_t)
        y2 = matrix_sub(y_i, mean)

        y3 = dot(y1, y2)
        y3 = matrix_multiply(y3, responsibility_matrix[i][l])
        y4 = matrix_multiply(y3, 1 / ml)

        cov = matrix_add(cov, y4)

    return cov


def init(y):
    global responsibility_matrix
    responsibility_matrix = init_matrix(nrows=len(y), ncols=L)


def normal_distribution(x, mean, cov, mix=1.0):
    exp = -0.5 * (dot(dot(matrix_sub(x, mean), matrix_invert(cov)), transpose(matrix_sub(x, mean)))[0][0])
    a = mix * (1 / ((pow(2 * math.pi, n / 2)) * pow(matrix_determinant(cov), 0.5))) * math.exp(exp)

    return a


def log_likelihood(samples, mix, m, covariances):
    likelihood = 0.0

    for y in samples:
        partial_sum = 0.0
        for i in range(L):
            partial_sum += normal_distribution(y, m[i], covariances[i], mix[i])
        likelihood += np.log(partial_sum)

    return likelihood


def e_step(samples, mixins, means, covariances):
    for i, y in enumerate(samples):
        norm = sum([normal_distribution(y, means[j], covariances[j], mixins[j]) for j in range(L)])
        responsibility_matrix[i] = [(normal_distribution(y, means[l], covariances[l], mixins[l]) / norm) for l in
                                    range(L)]


def m_step(samples, mixins, means, covariances):
    for l in range(L):
        ml = sum([responsibility_matrix[i][l] for i, _ in enumerate(samples)])
        mixins[l] = ml / len(samples)
        means[l] = calc_mean(l, samples, ml)
        covariances[l] = calc_cov_matrix(samples, means[l], ml, l)


# parameters

n_points = 200

x1 = 2
y1 = 2
sigma = 2

x2 = 5
y2 = 5

points = generate_2_class_points([[x1, y1], [x2, y2]], [sigma, sigma], n_points)

# c1 = [[2, 2], [3, 3], [3, 4], [2, 4]]
# c2 = [[12, 12], [12, 13], [13, 12], [13, 14]]

c1_points = [to_vec(x) for x in points[0]]
c2_points = [to_vec(x) for x in points[1]]

all_samples = c1_points + c2_points

init(all_samples)
last_likelihood = log_likelihood(all_samples, mixins, means, covs)
print("Initial log_likelihood: {}".format(last_likelihood))
converged = False

for i in range(max_iter):
    e_step(all_samples, mixins, means, covs)
    m_step(all_samples, mixins, means, covs)

    new_likelihood = log_likelihood(all_samples, mixins, means, covs)

    if math.isclose(last_likelihood, new_likelihood):
        print("Converged at iter {}!".format(i))
        converged = True
        break

    print("New log_likelihood: {}".format(new_likelihood))
    last_likelihood = new_likelihood
    print("Not converged yet ({}/{})".format(i, max_iter))

if converged:
    print("Final Means: M1={}, M2={}".format(means[0], means[1]))
    print("Final Covs: Cov1={}, Cov2={}".format(covs[0], covs[1]))
    print("Final Mixins: Mixin1={}, Mixin2={}".format(mixins[0], mixins[1]))

    unwrapped_c1 = np.array([unwrap(x) for x in c1_points])
    unwrapped_c2 = np.array([unwrap(x) for x in c2_points])
    unwrapped_centers = np.array([unwrap(x) for x in means])

    pyplot.plot(unwrapped_c1[:, 0], unwrapped_c1[:, 1], 'ro')
    pyplot.plot(unwrapped_c2[:, 0], unwrapped_c2[:, 1], 'bo')

    pyplot.plot(unwrapped_centers[:, 0], unwrapped_centers[:, 1], 'go')
    for i in range(L):
        pyplot.annotate("C{}".format(i), xy=(unwrapped_centers[i][0], unwrapped_centers[i][1]), size='xx-large',
                        color='g')

    pyplot.show()
else:
    print("Did not converge in {} iterations".format(max_iter))
