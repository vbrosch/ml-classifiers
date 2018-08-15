"""
Sample implementation of the perceptron algorithm.
"""
import numpy
from matplotlib import pyplot

# r connection
from utils.generate_points import generate_2_class_points

# parameters

init_b = 1

n = 100

x1 = 5
y1 = 5
sigma = 1

x2 = 10
y2 = 10

# generate class distributions using R

constant_bias = numpy.array([[1]] * n)

points = generate_2_class_points(means=[[x1, y1], [x2, y2]], sigmas=[sigma, sigma])
c1 = points[0]
c1_biased = numpy.append(c1, constant_bias, axis=1)
c1_biased = numpy.transpose(c1_biased)

c2 = points[1]
c2_biased = numpy.append(c2, constant_bias, axis=1)
c2_biased = numpy.multiply(-1, c2_biased)
c2_biased = numpy.transpose(c2_biased)

# get linear discriminant function

b = numpy.transpose(numpy.matrix([[init_b]] * 2*n))

x_matrix = numpy.concatenate((c1_biased, c2_biased), axis=1)

x_transposed = numpy.transpose(x_matrix)

x_matrix_transposed = numpy.dot(x_matrix, x_transposed)

x_pseudoinverse = numpy.linalg.inv(x_matrix_transposed)

x_transposed_pseudoinverse = numpy.dot(x_transposed, x_pseudoinverse)
w = numpy.dot(b, x_transposed_pseudoinverse)

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
