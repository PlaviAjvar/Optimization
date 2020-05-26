import decimal
import line_search
import copy

# helper function for finding partial derivative of function

def partial_derivative(f, x_cur, index, epsilon=decimal.Decimal(1e-10)):
    x_up = copy.deepcopy(x_cur)
    x_down = copy.deepcopy(x_cur)
    # add and remove epsilon variation to x
    x_up[index] += epsilon
    x_down[index] -= epsilon

    # divide by 2*epsilon to obtain symmetric Newton coefficient
    return (f(x_up) - f(x_down)) / (2 * epsilon)

# helper function which calculates the gradient numerically

def gradient(f, x_cur, epsilon=decimal.Decimal(1e-10)):
    n = len(x_cur)
    grad = [0 for _ in range(n)]

    # i-th component is the partial derivative over x_i
    for i in range(n):
        grad[i] = partial_derivative(f, x_cur, i)

    return grad

# function which finds a local minimum of the goal function
# Input parameters are:
# 1. Goal function: R^n -> R
# 2. Initial solution x_init
# 3. Algorithm in use (optional)
# 3. Sensitivity epsilon which determines convergence (optional)
# 4. Maximum number of iterations (optional)

def gradient_descent(goal_function, x_init, algorithm="BasicArm", epsilon=decimal.Decimal(1e-10), max_iter=10000):
    x_cur = x_init

    for iter in range(max_iter):
        # calculate gradient and set dx = -grad(x_cur)
        grad = gradient(goal_function, x_cur)
        dx = [-g_i for g_i in grad]

        # if the maximum norm among the dx components is bellow some epsilon, convergence is achieved
        if max([dx_i.copy_abs() for dx_i in dx]) < epsilon:
            return (x_cur, goal_function(x_cur), "Solution found in (" + str(iter+1) + ") iterations.")

        # solve line search potproblem in direction of -grad(x_cur)
        # use algorithm depending on algorithm parameter
        (s, f_s, msg) = line_search.line_search(goal_function, dx, x_cur, grad, algorithm)

        # update the current x-value based on the value of s we obtained
        x_cur = [x_cur[i] + s * dx[i] for i in range(len(x_cur))]

    # If we're here we have exceeded the maximum number of iterations
    return (x_cur, goal_function(x_cur), "Maximum number of iterations (" + str(max_iter) + ") exceeded.")