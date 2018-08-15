"""
    A simple example for 'agglomerative clustering'
"""
from pprint import pprint

import numpy as np
import sys
from math import sqrt

from lina.linear_algebra import transpose, matrix_sub, to_scalar, dot
from lina.vectors import to_vec

samples = [[1, 1], [2, 1], [8, 8], [9, 9], [9, 11], [12, 11]]
partitionings = []

m = len(samples)


class Cluster:
    def __init__(self, el: list):
        self.el = el

    def all_elements(self):
        all_e = []
        for element in self.el:
            if is_point(element):
                all_e.append(element)
            else:
                all_e.extend(element.all_elements())
        return all_e

    def get_element_tree(self):
        tree = []

        for element in self.el:
            if is_point(element):
                tree.append(element)
            else:
                tree.append([element.get_element_tree()])

        return tree


def is_point(p):
    return not is_cluster(p)


def is_cluster(c):
    return isinstance(c, Cluster)


"""def minimize_cluster(c):
    if len(c) == 1 and is_point(c[0]):
        return c[0]
    return c
"""


def merge_clusters(c1, c2):
    return Cluster([c1, c2])


def euclidean_distance(x, y) -> float:
    t = matrix_sub(to_vec(x), to_vec(y))
    t = dot(t, transpose(t))
    return sqrt(to_scalar(t))


def get_cluster_distances(c1, c2):
    return [euclidean_distance(x, y) for x in c1.all_elements() for y in c2.all_elements()]


def single_linkage_distance(c1, c2):
    return min(get_cluster_distances(c1, c2))


def complete_linkage_distance(c1, c2):
    return max(get_cluster_distances(c1, c2))


def initialize_cluster():
    p1 = [Cluster([s]) for s in samples]
    partitionings.append(p1)


def perform_clustering(linkage):
    for t in range(1, m):
        p_new: list = partitionings[t - 1].copy()

        min_i = -1
        min_j = -1
        min_distance = sys.maxsize

        for i, c1 in enumerate(p_new):
            for j, c2 in enumerate(p_new):
                if i == j:
                    continue
                distance = linkage(c1, c2)

                if distance < min_distance:
                    min_i = i
                    min_j = j
                    min_distance = distance

        c1_old = p_new[min_i]
        c2_old = p_new[min_j]

        new_cluster = merge_clusters(c1_old, c2_old)

        p_new.remove(c1_old)
        p_new.remove(c2_old)

        p_new.append(new_cluster)
        linkage(new_cluster, p_new[0])

        partitionings.append(p_new)


# single linkage distance

print("Will cluster with single linkage distance now")

initialize_cluster()

perform_clustering(single_linkage_distance)
clustering = partitionings[len(partitionings) - 1][0]
pprint(clustering.get_element_tree())

# complete linkage distance

print("Will cluster with full linkage distance now")

initialize_cluster()

perform_clustering(complete_linkage_distance)
clustering = partitionings[len(partitionings) - 1][0]
pprint(clustering.get_element_tree())
