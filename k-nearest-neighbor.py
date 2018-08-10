from math import sqrt

from linear_algebra import matrix_sub, transpose, dot, to_scalar
from vectors import to_vec

k = 3
samples = []


def init(c1: list, c2: list):
    global samples

    samples = [(to_vec(x), 0) for x in c1]
    samples += [(to_vec(x), 1) for x in c2]


def euclidean_distance(x, y) -> float:
    t = matrix_sub(x, y)
    t = dot(transpose(t), t)
    return sqrt(to_scalar(t))


def k_nearest(x) -> list:
    samples_distances = [(y[0], y[1], euclidean_distance(x, y[0])) for y in samples]
    samples_distances.sort(key=lambda x1: x1[2])  # sort by euclidean distance
    return samples_distances[:k]


def classify(nearest: list) -> int:
    n_sum = sum([x[1] for x in nearest])
    return 2 if n_sum >= (k / 2) else 1


def predict(x) -> int:
    nearest = k_nearest(to_vec(x))
    return classify(nearest)


c1_p = [[1, 2], [2, 3], [2, 5], [1, 6], [3, 7]]
c2_p = [[5, 1], [2, 4], [6, 4], [7, 6], [5, 7]]

p = [3, 4]

init(c1_p, c2_p)

for p in [[3, 4], [4, 4]]:
    c = predict(p)
    print("Classified {} -> C{}".format(p, c))
