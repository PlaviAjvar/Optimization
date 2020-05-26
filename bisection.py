import decimal

# implements bisection method (binary search) for finding zero of monotone increasing function
# returns x_0 and f(x_0) of root of function

# Pass following arguments
# 1. function
# 2. bounds of search interval

# Optionally pass:
# 1. sensitivity epsilon
# 2. maximum number of iterations

def bisection_zero(f, lower_bound, upper_bound, epsilon = decimal.Decimal(1e-20), max_iter = 10000):
    for iter in range(max_iter):
        # calculate midpoint of interval
        x_mid = (lower_bound + upper_bound) / 2
        f_val = f(x_mid)
        # if the function is close enough to zero return midpoint
        if f_val.copy_abs() < epsilon:
            return (x_mid, f_val, "Solution found in " + str(iter + 1) + " iterations.")

        # adjust bounds based on sign of f(x_mid)
        if f_val > 0:
            upper_bound = x_mid
        else:
            lower_bound = x_mid

    # if this is reached, the number of iterations exceeds the maximum
    return (x_mid, f_val, "Solution not found. Maximum number of iterations (" + str(max_iter) + ") exceeded.")

# implements bisection method (binary search) for finding minimum of convex function
# returns x_0 and f(x_0) or f'(x_0) of minimum
# for maximum of concave function, pass -f(x) as argument

# Pass following arguments
# 1. function or derivative (function_flag = True if function, False if derivative)
# 2. bounds of search interval

# Optionally pass:
# 1. flag which represents if function or derivative was passed (default is True)
# 2. sensitivity epsilon (when function is passed bounds x_upper - x_lower, when derivative is passed bounds derivative value)
# 3. maximum number of iterations

def bisection_minimum(f, lower_bound, upper_bound, function_flag = True, epsilon = decimal.Decimal(1e-20), max_iter = 10000):
    for iter in range(max_iter):
        # calculate midpoint of interval
        x_mid = (lower_bound + upper_bound) / 2

        # if the function itself was passed
        if function_flag:
            # if epsilon is too small the function values won't differ
            # set larger epsilon and adjust it depending on size of search interval
            dif_eps = max(1, upper_bound - lower_bound) * epsilon
            # symmetric difference for derivative approximation
            f_low = f(x_mid - dif_eps)
            f_high = f(x_mid + dif_eps)

            # adjust interval bounds
            if f_low < f_high: # function rising
                upper_bound = x_mid
            else: # function falling
                lower_bound = x_mid

            # search interval sufficiently bounded
            if upper_bound - lower_bound < epsilon:
                return (x_mid, f(x_mid), "Minimum found in " + str(iter + 1) + " iterations.")

        # if the derivative was passed explicitly
        else:
            f_val = f(x_mid)
            # if the derivative is almost zero
            if f_val.copy_abs() < epsilon:
                return (x_mid, f_val, "Minimum found in " + str(iter + 1) + " iterations.")

            # adjust bounds based on sign of df(x_mid) / dx
            if f_val > 0:
                upper_bound = x_mid
            else:
                lower_bound = x_mid

    # if this is reached, the number of iterations exceeds the maximum
    return (x_mid, f(x_mid), "Minimum not found. Maximum number of iterations (" + str(max_iter) + ") exceeded.")