import math

from regression import generate_points, get_x_range, get_point_x, get_w, plot_w_fn, plot_original_fn, \
    plot_point_samples, get_w_val

import matplotlib.pyplot as plt

m = 3


def get_sigma(w, x, t):
    return (1 / len(x)) * sum([math.pow((get_w_val(x_n, w) - t_n), 2) for x_n, t_n in zip(x, t)])


x_range = get_x_range()
x_point_range = get_point_x()

targets = generate_points(x_point_range)

w = get_w(x_point_range, targets, m)

plt.axis([min(x_range), max(x_range), -1.5, 1.5])

plot_point_samples(x_point_range, targets)
plot_original_fn(x_range)
plot_w_fn(x_range, w)

print("Estimated w: {}".format(w))

m = get_w_val(math.pi / 2, w)
variance = get_sigma(w, x_range, targets)
deviation = math.sqrt(variance)

i_start = m - 2 * deviation
i_end = m + 2 * deviation
print("Most likely point at x=pi/2 -> y = {}".format(m))
print("x=pi/2 -> y lies in interval [{},{}] with 95% probability!".format(i_start, i_end))

plt.plot([math.pi/2], [m], 'bo', label='$\mu$')
plt.annotate("$\mu - 2\sigma$", xy=(5, i_start), xytext=(5, i_start + 0.05))
plt.annotate("$\mu + 2\sigma$", xy=(5, i_end), xytext=(5, i_end + 0.05))

plt.axhline(i_start)
plt.axhline(i_end)

plt.legend()
plt.show()
