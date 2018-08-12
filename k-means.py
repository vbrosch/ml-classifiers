import math
import random

import numpy as np
from matplotlib import pyplot

from lina.linear_algebra import matrix_sub, dot, transpose
from lina.vectors import to_vec, vector_multiply, vector_init, vector_add, unwrap
from utils.generate_points import generate_points

max_iter = 10000
d = 2
k = 2
centers = [to_vec([0.0, 0.0]), to_vec([15.0, 15.0])]
clusters = [[], []]


def initialize(samples):
    sep = len(samples) / 2
    clusters[0] = samples[:int(sep)]
    clusters[1] = samples[int(sep):]


def decide_membership():
    samples = []

    for c in range(k):
        samples.extend((y, c) for y in clusters[c])

    for y_oc in samples:
        nc = nearest_cluster(y_oc[0])

        if y_oc[1] != nc:
            clusters[y_oc[1]].remove(y_oc[0])
            clusters[nc].append(y_oc[0])


def reestimate_centers():
    for c in range(k):
        centers[c] = cluster_mean(c)


def cluster_distance(y, i):
    y_m_mean = matrix_sub(y, centers[i])
    return math.sqrt(dot(y_m_mean, transpose(y_m_mean))[0][0])


def nearest_cluster(y):
    distances = [(i, cluster_distance(y, i)) for i in range(k)]
    # pick element with minimum distance
    el = min(distances, key=lambda x: x[1])

    return el[0]


def cluster_mean(i):
    norm = 1 / max(len(clusters[i]), 1)
    t_vec = vector_init(dim=d)

    for y in clusters[i]:
        t_vec = vector_add(t_vec, y)
    t_vec = vector_multiply(t_vec, norm)

    return t_vec


def cluster_variance(i):
    c_sum = 0.0
    for y in clusters[i]:
        y_m_mean = matrix_sub(y, centers[i])
        c_sum += dot(y_m_mean, transpose(y_m_mean))[0][0]

    return c_sum


def sum_of_cluster_variances():
    return sum(cluster_variance(i) for i in range(len(clusters)))


n = 100

x1 = 5
y1 = 5
sigma = 1

x2 = 10
y2 = 10

x3 = 7.5
y3 = 7.5

# generate class distributions using R

c1 = generate_points([x1, y1], sigma, n).tolist()
c2 = generate_points([x2, y2], sigma, n).tolist()
c3 = generate_points([x3, y3], sigma, n).tolist()

c1_vec = [to_vec(x) for x in c1]
c2_vec = [to_vec(x) for x in c2]
c3_vec = [to_vec(x) for x in c3]

all_samples = c1_vec + c2_vec + c3_vec

random.shuffle(all_samples)

initialize(all_samples)
last_means = centers.copy()
last_j = sum_of_cluster_variances()
converged = False

for i in range(max_iter):
    decide_membership()
    reestimate_centers()

    new_j = sum_of_cluster_variances()

    means_close = True

    for c in range(k):
        l_c = unwrap(last_means[c])
        c_c = unwrap(centers[c])

        if not (math.isclose(l_c[0], c_c[0]) and math.isclose(l_c[1], l_c[1])):
            means_close = False

    if means_close:
        print("Converged at iter {}!".format(i))
        converged = True
        break

    print("New J: {}".format(new_j))
    last_means = centers.copy()
    last_j = new_j
    print("Not converged yet ({}/{})".format(i, max_iter))

if converged:
    print("Center C1: {} | Center C2: {}".format(centers[0], centers[1]))
    print("Final Clusters, C1: {}".format(clusters[0]))
    print("Final Clusters, C2: {}".format(clusters[1]))

    unwrapped_c1 = np.array([unwrap(x) for x in clusters[0]])
    unwrapped_c2 = np.array([unwrap(x) for x in clusters[1]])
    unwrapped_centers = np.array([unwrap(x) for x in centers])

    pyplot.plot(unwrapped_c1[:, 0], unwrapped_c1[:, 1], 'ro')
    pyplot.plot(unwrapped_c2[:, 0], unwrapped_c2[:, 1], 'bo')

    pyplot.plot(unwrapped_centers[:, 0], unwrapped_centers[:, 1], 'go')
    for i in range(k):
        pyplot.annotate("C{}".format(i), xy=(unwrapped_centers[i][0], unwrapped_centers[i][1]), size='xx-large',
                        color='g')

    pyplot.show()

else:
    print("Did not converge in {} iterations".format(max_iter))
