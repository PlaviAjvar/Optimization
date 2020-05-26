import decimal
import copy

# implements Newton-Rhapson method for finding zero of function
# returns x_0 and f(x_0) of zero of function

# Pass following arguments
# 1. function
# 2. derivative of function
# 2. initial point

# Optionally pass:
# 1. sensitivity epsilon
# 2. maximum number of iterations

def newton_rhapson(f, df_dx, x_init, epsilon = decimal.Decimal(1e-20), max_iter = 10000):
    x = x_init
    for iter in range(max_iter):
        # find new candidate x
        f_val = f(x)
        x = x - f_val / df_dx(x)
        # if the function is close enough to zero return current candidate
        if f_val.copy_abs() < epsilon:
            return (x, f_val, "Solution found in " + str(iter) + " iterations")
    # if we reach here max iteration count has been exceeded
    return (x, f_val, "Solution not found. Maximum number of iterations ("  + str(max_iter) + ") has been exceeded.")


# if we have no closed form for df/dx, this function obtains the derivative numerically
def numeric_derivative(f, epsilon = decimal.Decimal(1e-10)):
    df_dx = lambda x : (f(x + epsilon) - f(x - epsilon)) / (2*epsilon)
    return df_dx