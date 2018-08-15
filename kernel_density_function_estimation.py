"""
    Kernel density function estimation
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from utils.generate_distribution import normal_distribution

min_x = 0
max_x = 20
d = 1


def kernel_function(x):
    return normal_distribution(x, mean=0, var=1)


def to_kernel_arg(h, x_i, y):
    return (x_i - y) / h


def density_function(h, x_i):
    fact1 = sum([kernel_function(to_kernel_arg(h, x_i, y)) for y in x])
    fact2 = (1 / (m * math.pow(h, d)))
    res1 = fact1 * fact2
    return res1


def plot_histogramm():
    plt.hist(x, 50, normed=True, alpha=0.5)


def plot_kernel_estimation():
    plt.plot(xfit, density, '-k', lw=2)


m = 1000
x = np.random.randn(m)

# perform kernel density
xfit = np.linspace(-5, 5, 1000)

for i, h in enumerate([0.1, 0.2, 0.5, 0.75, 1, 2]):
    plt.subplot(2, 3, i + 1)
    plt.title("$h = {}$".format(h))

    density = [density_function(h, x_i) for x_i in xfit]

    plot_histogramm()
    plot_kernel_estimation()


plt.show()
