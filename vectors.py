def vector_init(number: float = 0.0, dim: int = 0):
    return to_vec([number] * dim)


def __vector_dim(x):
    return len(x)


def vector_dim(x):
    return __vector_dim(unwrap(x))


def check_dim(x, y):
    if __vector_dim(x) != __vector_dim(y):
        raise ArithmeticError("Dimensions of vectors that should be added must match! dim(x) = {}, dim(y) = {}"
                              .format(len(x), len(y)))


def vector_binary_operation(x, y, op):
    x = unwrap(x)
    y = unwrap(y)

    check_dim(x, y)

    return to_vec([op(x[i], y[i]) for i in range(len(x))])


def vector_unary_operation(x, op):
    x = unwrap(x)

    return to_vec([op(x[i]) for i in range(len(x))])


def vector_add(x, y):
    return vector_binary_operation(x, y, lambda l, r: l + r)


def vector_sub(x, y):
    return vector_binary_operation(x, y, lambda l, r: l - r)


def vector_multiply(x, y: float):
    return vector_unary_operation(x, lambda x1: x1 * y)


def to_vec(x: object) -> object:
    return [x]


def unwrap(x):
    if len(x) > 0:
        return x[0]
    return []
