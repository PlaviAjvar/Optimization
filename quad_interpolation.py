import decimal
import numpy

# finds parabola through 3 points
def find_parabola(x_1, x_2, x_3, y_1, y_2, y_3, epsilon):
    # numpy inverse doesnt work directly with Decimal :(
    den1 = x_1*x_2 + x_1*x_3 - x_2*x_3 - x_1**2
    if den1.copy_abs() < epsilon:
        return 0, 0, 0, False

    a_11 = -1 / den1
    a_21 = (x_2 + x_3) / den1
    a_31 = -x_2*x_3 / den1

    den2 = x_1 * x_2 - x_1 * x_3 + x_2 * x_3 - x_2 ** 2
    if den2.copy_abs() < epsilon:
        return 0, 0, 0, False

    a_12 = -1 / den2
    a_22 = (x_1 + x_3) / den2
    a_32 = -x_1*x_3 / den2

    den3 = x_1*x_2 - x_1*x_3 - x_2*x_3 + x_3**2
    if den3.copy_abs() < epsilon:
        return 0, 0, 0, False

    a_13 = 1 / den3
    a_23 = -(x_1 + x_2) / den3
    a_33 = x_1*x_2 / den3

    a = a_11 * y_1 + a_12 * y_2 + a_13 * y_3
    b = a_21 * y_1 + a_22 * y_2 + a_23 * y_3
    c = a_31 * y_1 + a_32 * y_2 + a_33 * y_3

    return a, b, c, True

# function implementing quadratic interpolation method for minimizing functions

# Parameters of function are:
# 1. function to be optimized
# 2. Boundaries of the search interval

# Optional parameters are:
# 1. Sensitivity epsilon
# 2. Maximum iteration count

def quadratic_interpolation(f, x_low, x_high, epsilon = decimal.Decimal(1e-20), max_iter = 10000):
    # initial guess middle of interval
    x_c = (x_low + x_high) / 2
    for iter in range(max_iter):
        # find parabola passing through the 3 points
        (a, b, c, flag) = find_parabola(x_low, x_c, x_high, f(x_low), f(x_c), f(x_high), epsilon)

        # calculate new guess for x
        # if matrix is singular resort to bisection
        if not flag:
            x_new = (x_low + x_high) / 2
        else:
            x_new = -b / (2*a)

        # compare function value at new guess compared to current one
        # and compare the positions themselves
        x_next = x_new
        if x_new < x_c:
            if f(x_new) <= f(x_c) + epsilon:
                x_high = x_c
            if f(x_new) + epsilon >= f(x_c):
                x_low = x_new
                x_next = x_c
        else:
            if f(x_c) <= f(x_new) + epsilon:
                x_high = x_new
                x_next = x_c
            if f(x_c) + epsilon >= f(x_new):
                x_low = x_c

        # if convergence condition is satisfied return solution
        if (f(x_new) - f(x_c)).copy_abs() < epsilon:
            return (x_new, f(x_new), "Solution found in " + str(iter) + " iterations.")

        x_c = x_next

    # maximum iterations reached without finding solution
    return (x_new, f(x_new), "Solution not found. Maximum iterations (" + str(max_iter) + ") exceeded.")