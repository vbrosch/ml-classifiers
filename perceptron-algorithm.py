"""
Sample implementation of the perceptron algorithm
"""
import numpy
from matplotlib import pyplot

from utils.generate_points import generate_2_class_points


def get_delta(y_i: numpy.matrix, w: numpy.matrix) -> float:
    y_i_t = numpy.transpose(y_i)
    d_i = numpy.dot(w, y_i_t)

    return d_i


n = 100
learning_rate = 1
max_iter = 10000

constant_bias = numpy.array([[1]] * n)

points = generate_2_class_points(n=n)

c1 = points[0]
c1_biased = numpy.append(c1, constant_bias, axis=1)

c2 = points[1]
c2_biased = numpy.append(c2, constant_bias, axis=1)
c2_biased = numpy.multiply(-1, c2_biased)

# get linear discriminant function

w = numpy.matrix([1, 1, 1])
c1_c2 = numpy.concatenate((c1_biased, c2_biased), axis=0)

for k in range(max_iter):
    delta = numpy.matrix([0, 0, 0])

    for y_i in c1_c2:
        d_i = get_delta(y_i, w)
        if d_i <= 0:
            delta = numpy.add(delta, y_i)

    w = numpy.add(w, numpy.multiply(learning_rate, delta))

    holds_for_all = True
    for y_i in c1_c2:
        d_i = get_delta(y_i, w)

        if d_i <= 0:
            holds_for_all = False

    if holds_for_all:
        print("Converged at iteration {}/{}".format(k, max_iter))
        break
    print("No convergence at iteration {}/{}".format(k, max_iter))

print("Parameters are w=({}, {}, {})".format(w.item(0), w.item(1), w.item(2)))

x1 = w.item(0)
x2 = w.item(1)
x3 = w.item(2)

intercept = -x3 / x2
slope = -x1 / x2

print("Calculated linear discriminant function as y = {}*x + ({})".format(slope, intercept))

x_points = range(15)
y_points = list(map(lambda x: slope * x + intercept, x_points))

pyplot.plot(c1[:, 0], c1[:, 1], 'ro')
pyplot.plot(c2[:, 0], c2[:, 1], 'bo')
pyplot.plot(x_points, y_points, 'g')
pyplot.show()
