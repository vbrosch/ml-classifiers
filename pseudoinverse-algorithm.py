"""
Sample implementation of the perceptron algorithm.
"""
import numpy
import rpy2.robjects as robjects
from matplotlib import pyplot
from rpy2.robjects.packages import importr

# r connection

r = robjects.r
MASS = importr('MASS')

# parameters

init_b = 1

n = 100

x1 = 5
y1 = 5
sigma = 1
cov_mat1 = r.matrix(r.c(sigma, 0, 0, sigma), nrow=2, ncol=2)

x2 = 10
y2 = 10
cov_mat2 = r.matrix(r.c(sigma, 0, 0, sigma), nrow=2, ncol=2)

# generate class distributions using R

constant_bias = numpy.array([[1]] * n)

c1 = numpy.matrix(MASS.mvrnorm(n=n, mu=r.c(x1, y1), Sigma=cov_mat1))
c1_biased = numpy.append(c1, constant_bias, axis=1)
c1_biased = numpy.transpose(c1_biased)

c2 = numpy.matrix(MASS.mvrnorm(n=n, mu=r.c(x2, y2), Sigma=cov_mat2))
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
