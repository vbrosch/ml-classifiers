import math
import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(x, mean, var) -> float:
    return (1 / (math.sqrt(2 * math.pi * var))) * math.exp(-(math.pow(x - mean, 2)) / (2 * var))


def get_point(x, mixins, means, variances):
    return [x,
            sum([mixins[pdf_i] * normal_distribution(x, means[pdf_i], variances[pdf_i]) for pdf_i in
                 range(len(means))])]


def generate_set(min_x=0, max_x=20, m=1000, mixins=None, means=None, variances=None):
    if variances is None:
        variances = [1, 5]
    if means is None:
        means = [5, 15]
    if mixins is None:
        mixins = [0.5, 0.5]

    return [get_point(x, mixins, means, variances) for x in np.arange(min_x, max_x, max_x / m)]


def plot_pdf(samples):
    x_axis = [x[0] for x in samples]
    y_axis = [x[1] for x in samples]

    plt.plot(x_axis, y_axis, color='#A5DF00')
