"""
Sample implementation of the perceptron algorithm
"""
import numpy
import rpy2.robjects as robjects
from matplotlib import pyplot
from rpy2.robjects.packages import importr


def get_delta(y_i: numpy.matrix, w: numpy.matrix) -> float:
    y_i_t = numpy.transpose(y_i)
    d_i = numpy.dot(w, y_i_t)

    return d_i


# r connection

r = robjects.r
MASS = importr('MASS')

# parameters

learning_rate = 1
max_iter = 10000

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

c2 = numpy.matrix(MASS.mvrnorm(n=n, mu=r.c(x2, y2), Sigma=cov_mat2))
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
