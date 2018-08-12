import numpy as np


def init_matrix(val: float = 0.0, nrows=0, ncols=0):
    return [[val] * ncols for _ in range(nrows)]


def get_cols(x):
    if len(x) == 0:
        return 0
    return len(x[0])


def get_rows(x):
    return len(x)


def row_val(x, i):
    return x[i]


def col_val(x, i):
    return [x_i[i] for x_i in x]


def matrix_item(x, i, j):
    return x[i][j]


def shape(x):
    return get_rows(x), get_cols(x)


def transpose(x):
    x_shape = shape(x)

    t = init_matrix(nrows=x_shape[1], ncols=x_shape[0])

    for row in range(0, x_shape[0]):
        for col in range(0, x_shape[1]):
            t[col][row] = x[row][col]

    return t


def dot(x, y):
    x_shape = shape(x)
    y_shape = shape(y)

    if x_shape[1] != y_shape[0]:
        raise ArithmeticError("Invalid shapes of matrices/vectors!")

    size = x_shape[1]

    result = init_matrix(nrows=x_shape[0], ncols=y_shape[1])

    for i in range(0, x_shape[0]):
        for j in range(0, y_shape[1]):
            s = 0

            for k in range(0, size):
                s += x[i][k] * y[k][j]

            result[i][j] = s

    return result


def __matrix_check_dim_equality(x, y):
    if shape(x) != shape(y):
        raise ArithmeticError("Shapes of matrices that should be added must match! shape(x) = {}, shape(y) = {}"
                              .format(shape(x), shape(y)))


def __matrix_binary_operation(x, y, op):
    __matrix_check_dim_equality(x, y)
    x_shape = shape(x)

    return [[op(x[row_i][col_i], y[row_i][col_i]) for col_i in range(x_shape[1])] for row_i in range(x_shape[0])]


def __matrix_unary_operation(x, op):
    return [[op(col) for col in row] for row in x]


def matrix_add(x, y):
    return __matrix_binary_operation(x, y, lambda el1, el2: el1 + el2)


def matrix_sub(x, y):
    return __matrix_binary_operation(x, y, lambda el1, el2: el1 - el2)


def matrix_multiply(x, y: float):
    return __matrix_unary_operation(x, lambda el: el * y)


def __as_np(x):
    return np.matrix(x)


def matrix_invert(x):
    return np.linalg.inv(__as_np(x)).tolist()


def matrix_determinant(x):
    return np.linalg.det(x)


def to_scalar(x):
    return x[0][0]


def matrix_diag(number: int, shape):
    m = init_matrix(nrows=shape[0], ncols=shape[1])
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                m[i][j] = number

    return m
