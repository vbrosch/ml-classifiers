"""
    Example of performing a ridge regression over a data set
    => Solve the equation w = b(XX^t + λI)^-1
"""

import matplotlib.pyplot as plt

from regression import get_x_range, get_point_x, generate_points, get_w, plot_original_fn, plot_point_samples, plot_w_fn

m = 9

x_range = get_x_range()
x_point_range = get_point_x()
targets = generate_points(x_point_range)

for i, lam in enumerate([0, 0.25, 0.5, 1, 10, 15]):
    plt.subplot(2, 3, i + 1)
    plt.title("$\lambda = {}$".format(lam))
    plt.axis([min(x_range), max(x_range), -1.5, 1.5])

    w = get_w(x_point_range, targets, m, lam)

    print("lambda = {}, w = {}".format(lam, w))

    plot_original_fn(x_range)
    plot_point_samples(x_point_range, targets)
    plot_w_fn(x_range, w)

plt.show()
