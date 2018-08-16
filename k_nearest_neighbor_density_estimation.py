import math

import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x, y):
    return math.sqrt(math.pow(x - y, 2))


def k_nearest_neighbor_distance(x, sample_set, k):
    distances = [euclidean_distance(x, y) for y in sample_set]
    distances.sort()

    return distances[k - 1]


def unit_sphere_volume(d):
    return (math.pow(math.pi, d / 2)) / math.gamma(d / 2 + 1)


def density_function(d, x, sample_set, k, n):
    return k / (n * unit_sphere_volume(d) * k_nearest_neighbor_distance(x, sample_set, k))


def plot_density_estimation(x, y):
    plt.plot(x, y, '-k', lw=2)


def main():
    d = 1
    m = 1000
    x = np.random.randn(m)
    k = 100

    for i, k in enumerate([1, 5, 10, 25, 50, 100]):
        plt.subplot(2, 3, i + 1)
        plt.axis([-4, 4, 0, 0.5])
        plt.title("$k={}$".format(k))

        # perform kernel density
        xfit = np.linspace(-5, 5, 1000)
        plt.hist(x, 50, normed=True, alpha=0.5)

        density = [density_function(d, x_i, x, k, m) for x_i in xfit]

        plot_density_estimation(xfit, density)

    plt.show()


main()
