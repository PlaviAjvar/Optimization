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

def exact_line_search(goal_function, directional_vector, x_init, epsilon = 1e-20, max_iter = 10000, algorithm = "Newton"):
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