import math

import numpy as np
import matplotlib.pyplot as plt

# parameters
from utils.generate_distribution import generate_set, plot_pdf

min_x = 0
max_x = 20


def get_bins():
    return [(x, get_bin_density(x)) for x in np.arange(min_x, max_x, delta)]


def get_bin_density(x):
    m_i = sum([s[1] for s in samples if x <= s[0] < (x + delta)])
    return m_i / (sum(s[1] for s in samples) * delta)


def plot_bins():
    x_bins = [x[0] for x in bins]
    densities = [x[1] for x in bins]

    plt.bar(x_bins, densities, delta, color='#0B3861', edgecolor='#000000', linewidth=1)


def bin_probabilities():
    return [(b[1] * delta) for b in bins]


samples = generate_set(min_x, max_x)

for i, dx in enumerate([0.25, 0.5, 1, 2, 4, 5]):
    plt.subplot(2, 3, i + 1)
    plt.title("$\Delta = {}$".format(dx))

    delta = dx

    plot_pdf(samples)
    bins = get_bins()
    plot_bins()

    bin_probs = bin_probabilities()
    print("Bin probabilities for d={}: {}".format(dx, bin_probs))

    sum_probs = sum(bin_probs)
    print("Is bin probability for d={} valid probability distribution: {} ({})".format(dx, math.isclose(sum_probs, 1),
                                                                                       sum_probs))

plt.show()
