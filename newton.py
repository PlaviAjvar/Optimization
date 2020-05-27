import decimal
import numpy
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


# helper function which finds the value of the hessian at some point numerically
# How to do this?
# In every outer loop iteration calculate the value of the partial derivative over the appropriate variable
# Then find the gradient of that function. This becomes a column of the matrix.
# Under some assumptions, the partial derivatives will be symmetric.

def hessian(f, x_cur, epsilon=decimal.Decimal(1e-10)):
    n = len(x_cur)
    H = [[0] * n for i in range(n)]   # n x n matrix

    for j in range(n):
        # find the partial derivative over x_j
        f_pj = lambda x: partial_derivative(f, x, j)

        for i in range(n):
            # now we need to find the derivative of the partial derivative over x_i
            H[i][j] = partial_derivative(f_pj, x_cur, i)

    return H


# implements Newton's method (Newton-Euler) for finding the minimum of a multivariable function
# Combined with line search to find even better candidate solutions

def newton(goal_function, x_init, algorithm="BasicArm", epsilon=decimal.Decimal(1e-10), max_iter=10000):
    x = [x_init]
    y = [goal_function(x_init)]
    n = len(x_init)

    for iter in range(max_iter):
        # find the hessian numerically
        # unfortunatelly, because numpy doesn't work with decimal
        # the type has to be changed to something like long double
        H_temp = hessian(goal_function, x[iter])
        H = numpy.array(H_temp, dtype=numpy.float64)

        # calculate the gradient as well numerically, same applies
        grad_temp = gradient(goal_function, x[iter])
        grad = numpy.array([[entry] for entry in grad_temp], dtype=numpy.float64)

        # calculate change in x based on Newton-Rhapson step
        # dx = H^-1 * grad
        dx = -1 * numpy.dot(numpy.linalg.inv(H), grad)

        # test definiteness of Hessian, in min it should be positive definite
        q_arr = numpy.dot(numpy.dot(numpy.transpose(dx), H), dx)
        q = decimal.Decimal(q_arr[0][0])
        # if q is possitive and small enough (2 is because we haven't halved the quadratic form) we have converged
        if q > 0 and q < 2*epsilon:
            return (x[iter], goal_function(x[iter]), "Solution found in (" + str(iter + 1) + ") iterations.", x, y)

        # now that we have the Newton-Rhapson diferential dx, we apply backtracking line search in that direction
        dx = [decimal.Decimal(dx[i][0]) for i in range(n)]
        grad = [decimal.Decimal(grad[i][0]) for i in range(n)]
        (s, f_s, msg) = line_search.line_search(goal_function, dx, x[iter], grad, algorithm)

        # compute next candidate x_k+1 = x_k + sdx
        x.append([x[iter][i] + dx[i] * s for i in range(n)])
        y.append(goal_function(x[-1]))

    # if we're here the maximum number of iterations has been exceeded
    return (x[-1], goal_function(x[-1]), "Maximum number of iterations (" + str(max_iter) + ") exceeded.", x, y)