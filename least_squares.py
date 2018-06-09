"""
    Calculate a regression function
    The polynomial degree can be set with the argument polynomial degree.
"""

import random
from functools import reduce

import numpy
from matplotlib import pyplot
from numpy.linalg import inv

print("Before a basic least-squares line fitting")

number_of_points = 20

slope = 2
intercept = 1

polynomial_degree = 1
length_w = polynomial_degree + 1


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def get_point_on_line(l_slope, l_intercept, x):
    return l_slope * x + l_intercept


def get_point_on_generator_line(x):
    return get_point_on_line(slope, intercept, x) + (random.randint(-3, 3) * random.random())


def generate_points(n: int) -> list:
    points = []
    for x in range(n):
        y = get_point_on_generator_line(x)
        points.append(Point(x, y))
    return points


def get_sum(arr: []):
    return reduce(lambda x_1, x_2: x_1 + x_2, arr)


def x_sum_with_power(elements: [], exponent: int):
    return reduce(lambda x, y: x + y, map(lambda x: pow(x.x, exponent), elements))


def get_matrix(points: list) -> numpy.matrix:
    matrix = []
    for i in range(length_w):
        row = []
        for j in range(length_w):
            exp = i + j
            row.append(x_sum_with_power(points, exp))

        matrix.append(row)
    return numpy.matrix(matrix)


def get_target_vector_element(points: list, exponent: int):
    return get_sum(map(lambda p: p.y * pow(p.x, exponent), points))


def get_target_vector(points: list) -> []:
    target_vector = []
    for i in range(length_w):
        target_vector.append(get_target_vector_element(points, i))
    return target_vector


def calculate_point_with_w(w: numpy.ndarray, x: float) -> float:
    y = 0
    for i in range(length_w):
        y += w.item(i) * pow(x, i)

    return y


# generate points
points_g = generate_points(number_of_points)

target_vector = get_target_vector(points_g)
print("Target vector: {}".format(target_vector))

a_matrix = get_matrix(points_g)
print("Matrix A={}".format(a_matrix))

a_inverse = inv(a_matrix)
print("Matrix A^-1={}".format(a_inverse))

w = numpy.dot(a_inverse, target_vector)

print("Solve weights by calculation of A^-1 . t = {}".format(w))

x = list(map(lambda p: p.x, points_g))
y = list(map(lambda p: p.y, points_g))
y_generated = list(map(lambda x_1: calculate_point_with_w(w, x_1), x))

pyplot.plot(x, y, 'ro')
pyplot.plot(x, y_generated)
pyplot.show()
