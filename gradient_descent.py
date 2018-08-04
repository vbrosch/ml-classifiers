import math


def fx(x: float):
    """
    The function for which gradient descent will be performed.
    fx(x) returns the function value of f(x)=(x+3)^2+1 at position x.
    :param x: the value of x
    :return: the function value of f(x)
    """
    return pow((x - 3), 2) + 1


def dx_fx(x: float):
    """
    Return the derivative function value for f(x) = x^2.
    :param x:
    :return:
    """
    return 2 * (x - 3)


def gradient_descent(x: float, dx_fx_f, learning_rate: float = 0.1, tol: float = 1e-9, max_iter=10000) -> float:
    """
    Perform a gradient descent to find an extrema of f(x) = x^2
    :param dx_fx_f: the derivative of the function for which gradient descent should be performed
    :param x: the current value of x (or starting value)
    :param learning_rate: the learning rate which lies between [0,1] (default 0.1)
    :param tol: the tolerance which is used to check for convergence (default 1e-9)
    :param max_iter: the maximum number of iterations
    :return:
    """

    new_x = x
    i = 0

    for i in range(max_iter):
        new_x = x - learning_rate * dx_fx_f(x)

        # check convergence
        if math.isclose(x, new_x, rel_tol=tol):
            print("Converged at x={} in iteration: {}".format(new_x, i))
            return new_x

        x = new_x
        print("Setting x={}. Not converged yet in iteration {}/{}".format(new_x, i, max_iter))

    print("Maximum number of iterations reached! Current x={} in iteration {}".format(new_x, i))
    return new_x


x_start = 300

print("Searching for extrema of f(x) = x^2 with x=10")
x_ex = gradient_descent(x_start, dx_fx)

print("Gradient descent returned extrema at x={}, y={}".format(x_ex, fx(x_ex)))
