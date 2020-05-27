import bisection
import newton_rhapson
import quad_interpolation
import decimal
from operator import add

# function implementing exact line search using one of the search methods
# Takes goal function, a directional vector and the initial point, a sensitivity epsilon and max_iter count
# Also takes parameter string which determines which algorithm should be used. The choices are:

# 1. "Newton" (Newton-Rhapson, the default parameter)
# 2. "Bisection" (Bisection method / binary search)
# 3. "Quadint" (Quadratic interpolation method)

# Returns the parameter value s which minimizes the goal function in that direction
# Also returns the value of the function and a runtime message


def exact_line_search(goal_function, directional_vector, x_init, epsilon=1e-20, max_iter=10000, algorithm="Newton"):
    # constant for search bounds
    x_symm = decimal.Decimal(5)
    # define function in direction of the directional vector
    gamma = lambda s: goal_function(list(map(add, x_init, [s*x_i for x_i in directional_vector])))
    if algorithm == "Bisection":
        (s_0, f_0, msg) = bisection.bisection_minimum(gamma, -x_symm, x_symm, True, epsilon, max_iter)
    elif algorithm == "Quadint":
        (s_0, f_0, msg) = quad_interpolation.quadratic_interpolation(gamma, -x_symm, x_symm, epsilon, max_iter)
    else:
        # need to calculate derivative numerically
        dg_dx = newton_rhapson.numeric_derivative(gamma)
        d2g_dx2 = newton_rhapson.numeric_derivative(dg_dx)
        # start from some initial point in [0, 1]
        (s_0, f_0, msg) = newton_rhapson.newton_rhapson(dg_dx, d2g_dx2, decimal.Decimal(0.5), epsilon, max_iter)
    return (s_0, f_0, msg)


# helper function for calculating directional derivative given gradient and dx
def directional_derivative(gradient, directional_vector):
    derivative = decimal.Decimal(0)
    for i in range(len(gradient)):
        derivative += gradient[i] * directional_vector[0]
    return derivative


# Function imeplementing Armijo backtracking line search
# The function takes the following parameters:

# 1. The goal function
# 2. The direction vector of the line search
# 3. The current initial point
# 4. The gradient at the initial point (needed for the gradient descent method anyway)
# 5. Parameter alpha controls the relative upwards pitch of line compared to tangent (default value 0.4)
# 6. Parameter beta controls the change of parameter s while iterating (default value 0.7)


def basic_armijo(goal_function, directional_vector, x_init, gradient, alpha=decimal.Decimal(0.2), beta=decimal.Decimal(0.8), max_iter=10000):
    # goal function restricted to given direction
    gamma = lambda s: goal_function(list(map(add, x_init, [s * x_i for x_i in directional_vector])))

    # first we need to find a small enough step s_0, for which the value of gamma drops
    # we'll do this by repeated halving, which shouldnt influence the algorithm runtime that much
    # most of the time the initial guess should be sufficient
    s_0 = decimal.Decimal(1e-5)
    while gamma(s_0) > gamma(0):
        s_0 /= 2

    # now we iterate to find s
    s = decimal.Decimal(1)
    epsilon = decimal.Decimal(1e-10)  # for numerical stability

    for iter in range(max_iter):
        f_s = gamma(s)
        f_0 = gamma(decimal.Decimal(0))
        d_0 = directional_derivative(gradient, directional_vector)

        # if the uplifted tangent is above the function value
        if f_s <= f_0 + alpha * s * d_0 - epsilon:
            # we are in the permissible region
            return (s, f_s, "Solution found in " + str(iter + 1) + " iterations.")
        else:
            # the solution isn't in the permissible region, adjust s
            s *= beta

        # if we have dropped bellow s_0, which guarantees a drop in function value
        if s < s_0:
            return (s_0, gamma(s_0), "Solution found in " + str(iter + 1) + " iterations.")


    # If this is reached, the maximum number of iterations has been exceeded
    # in which case we return s_0
    return (s_0, gamma(s_0), "Maximum number of iterations (" + str(max_iter) + ") exceeded.")


# helper saturation function for cubic armijo algorithm

def saturation(s_imp, s_low, s_high):
    # if it goes outside the limits, round it up or down
    if s_imp < s_low:
        return s_low
    if s_imp > s_high:
        return s_high
    # otherwise return the proper value
    return s_imp


# helper function for interpolating a cubic polynomial
# we've been given 3 (s, gamma(s)) pairs, one of which is (0, gamma(0))
# we've also been given the derivative at zero (0, gamma'(0))

# this gives us sufficient information to extrapolate the coefficients of the cubic
# in fact, because we've been given gamma'(0) and gamma(0), this reduces to a 2 x 2 system of linear equations

# obtaining the minimum of the cubic gives us the next guess for s
# we will give the solutions to the system implicitly, rather than calculating it on the go

def cubic(f_k, d_k, f_s, s, f_p, s_p):
    # if s = s_p we dont have enough info
    # this shouldnt occur but to make sure there are no divisions by zero
    if s == s_p:
        return s / 2

    # find determinant and right side of equations
    det = s**2 * s_p**2 * (s - s_p)
    b_1 = f_s - f_k - s*d_k
    b_2 = f_p - f_k - s_p*d_k

    # calculate coeficients of cubic
    a = (s_p**2 * b_1 - s**2 * b_2) / det
    b = (-s_p**3 * b_1 + s**3 * b_2) / det
    c = d_k
    d = f_k

    # if a = 0 -> quadratic instead of cubic
    epsilon = decimal.Decimal(1e-10)  # numeric stability
    if a.copy_abs() < epsilon:
        return -c / (2*b)

    # otherwise it's proper cubic
    # we have two solutions, we take the rightmost one
    D = b*b - 3*a*c
    return (-b + D.sqrt()) / (3*a)


def cubic_armijo(goal_function, directional_vector, x_init, gradient, alpha=decimal.Decimal(0.2), max_iter=10000):
    # goal function restricted to given direction
    gamma = lambda s: goal_function(list(map(add, x_init, [s * x_i for x_i in directional_vector])))

    # first we need to find a small enough step s_0, for which the value of gamma drops
    # we'll do this by repeated halving, which shouldnt influence the algorithm runtime that much
    # most of the time the initial guess should be sufficient
    s_0 = decimal.Decimal(1e-5)
    while gamma(s_0) > gamma(0):
        s_0 /= 2

    f_k = gamma(0)
    d_k = directional_derivative(gradient, directional_vector)
    s = decimal.Decimal(1)
    s_p = decimal.Decimal(1)  # s in previous iteration (makes no sense for first iteration)
    epsilon = decimal.Decimal(1e-10)  # numeric stability

    # iterativelly improve parameter s
    for iter in range(max_iter):
        # parameter has dropped bellow the minimum value, break iteration
        if s < s_0:
            return (s_0, gamma(s_0), "Solution found in " + str(iter + 1) + " iterations.")

        f_s = gamma(s)
        # if the function is bellow the uplifted tangent, we found an appropriate s
        if f_s <= f_k + s*alpha*d_k - epsilon:
            return (s, f_s, "Solution found in " + str(iter + 1) + " iterations.")

        # otherwise we need to improve parameter s
        # if the parameter has the initial value (s = 1) set it to the initial guess
        if (s - 1).copy_abs() < epsilon:
            s_imp = d_k / (2*(f_k + d_k - f_s)).copy_abs()

        # otherwise it's not the first iteration, use cubic interpolation to improve s
        else:
            s_imp = cubic(f_k, d_k, f_s, s, f_p, s_p)

        # current s and f become the previous in next iteration
        s_p = s
        f_p = f_s

        # to ensure that s doesn't drop too much and that it doesn't increase too much
        # add saturation limits of [s / 10, s / 2] and round to the nearest limit if they are exceeded
        s = saturation(s_imp, s / 10, s / 2)

    # if we make it here the maximum iteration count has been exceeded, still return s_0 and gamma(s_0)
    return (s_0, gamma(s_0), "Maximum number of iterations (" + str(max_iter) + ") exceeded.")


# helper function which calls basic or cubic armijo depending on algorithm parameter

def armijo_line_search(goal_function, directional_vector, x_init, gradient, algorithm="BasicArm",
                       alpha = decimal.Decimal(0.2), beta = decimal.Decimal(0.8), max_iter = 10000):
    if algorithm == "BasicArm":
        return basic_armijo(goal_function, directional_vector, x_init, gradient, alpha, beta, max_iter)
    return cubic_armijo(goal_function, directional_vector, x_init, gradient, alpha, max_iter)


# master function which calls all other line search methods
# used in gradient descent
# algorithm can be any of the standard labels of the algorithms used in the other functions
# 1. "BasicArm" for basic Armijo backtracking line search.
# 2. "CubicArm" for cubic Armijo backtracking line search.
# 3. "Newton" for Newton-Rhapson line search.
# 4. "Bisection" for Bisection line search.
# 5. "Quadint" for Quadratic interpolation line search.

def line_search(goal_function, directional_vector, x_init, gradient, algorithm="BasicArm"):
    if algorithm == "CubicArm" or algorithm == "BasicArm":
        return armijo_line_search(goal_function, directional_vector, x_init, gradient, algorithm)
    return exact_line_search(goal_function, directional_vector, x_init, decimal.Decimal(1e-5),
                                                10000, algorithm)