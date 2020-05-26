import line_search
import decimal
import gradient_descent
import matplotlib.pyplot as plt
from operator import add

############################
# Goal functions for testing
############################

# dimensionality two
def fun(x):
    return 3*x[0]**2 + x[1]**2 + 4

# dimensionality two
def gun(x):
    return x[0]**2 + x[1]**2 - 2*x[0]*x[1]

# dimensionality 2
def hun(x):
    return 10*x[0]**4 - 20*x[0]**2 * x[1] + 10*x[1]**2 + x[0]**2 - 2*x[0] + 5

# dimensionality 3
def pun(x):
    return x[0]**2 + 2*x[1]**2 + 2*x[2]**2 + 2*x[0]*x[1] + 2*x[1]*x[2]

########################
# Testing
########################

def test_exact(algorithm):
    if algorithm == "Newton":
        print("****************")
        print("Newton-Rhapson")
        print("****************\n")
    elif algorithm == "Bisection":
        print("****************")
        print("Bisection method")
        print("****************\n")
    else:
        print("***********************")
        print("Quadratic interpolation")
        print("***********************\n")

    # default parameters for algorithm
    epsilon = decimal.Decimal(1e-20)
    max_iter = 10000

    print("Minimum of function f(x, y) = 3*x^2 + y^2 + 4 in the direction (dx, dy) = (1, 1), starting at (x, y) = (1, 1):\n")
    (s_0, f_0, msg) = line_search.exact_line_search(fun, [decimal.Decimal(1), decimal.Decimal(1)], [decimal.Decimal(1),
                                                    decimal.Decimal(1)], epsilon, max_iter, algorithm)
    print("f(x + " + str(s_0) + ", y + " + str(s_0) + ") = " + str(fun([decimal.Decimal(1) + s_0, decimal.Decimal(1) + s_0])))
    print("f'(" + str(s_0) + ") =", f_0)
    print("s =", s_0)
    print(msg)
    print("\n")

    print("Minimum of function f(x, y) = x^2 + y^2 - 2xy in the direction (dx, dy) = (0.3, 0.4), starting at (x, y) = (1, 1):\n")
    (s_0, f_0, msg) = line_search.exact_line_search(gun, [decimal.Decimal(0.3), decimal.Decimal(0.4)],
                                                    [decimal.Decimal(1), decimal.Decimal(1)], epsilon, max_iter, algorithm)
    print("f(x + " + str(s_0 * decimal.Decimal(0.3)) + ", y + " + str(s_0 * decimal.Decimal(0.4)) + ") = " +
          str(gun([decimal.Decimal(1) + s_0 * decimal.Decimal(0.3), decimal.Decimal(1) + s_0 * decimal.Decimal(0.4)])))
    print("f'(" + str(s_0) + ") =", f_0)
    print("s =", s_0)
    print(msg)
    print("\n")

    print(
        "Minimum of function f(x, y) = 10x^4 - 20x^2 y + 10y^2 + x^2 - 2x + 5 in the direction (dx, dy) = (0.3, 0.4), starting at (x, y) = (2, 2):\n")
    (s_0, f_0, msg) = line_search.exact_line_search(hun, [decimal.Decimal(0.3), decimal.Decimal(0.4)],
                                                    [decimal.Decimal(2), decimal.Decimal(2)], epsilon, max_iter, algorithm)
    print("f(x + " + str(s_0 * decimal.Decimal(0.3)) + ", y + " + str(s_0 * decimal.Decimal(0.4)) + ") = " +
          str(hun([decimal.Decimal(2) + s_0 * decimal.Decimal(0.3), decimal.Decimal(2) + s_0 * decimal.Decimal(0.4)])))
    print("f'(" + str(s_0) + ") =", f_0)
    print("s =", s_0)
    print(msg)
    print("\n")

    print(
        "Minimum of function f(x, y, z) = x^2 + 2y^2 + 2z^2 + 2xy + 2yz in the direction (dx, dy, dz) = (0.3, 0.4, 0.5), starting at (x, y, z) = (2, 2, 2):\n")
    (s_0, f_0, msg) = line_search.exact_line_search(pun, [decimal.Decimal(0.3), decimal.Decimal(0.4), decimal.Decimal(0.5)],
                                                    [decimal.Decimal(2), decimal.Decimal(2), decimal.Decimal(2)], epsilon, max_iter, algorithm)
    print("f(x + " + str(s_0 * decimal.Decimal(0.3)) + ", y + " + str(s_0 * decimal.Decimal(0.4)) + ", z + " +
          str(s_0 * decimal.Decimal(0.5)) + ") = " +
          str(pun([decimal.Decimal(2) + s_0 * decimal.Decimal(0.3), decimal.Decimal(2) + s_0 * decimal.Decimal(0.4),
                   decimal.Decimal(2) + s_0 * decimal.Decimal(0.5)])))
    print("f'(" + str(s_0) + ") =", f_0)
    print("s =", s_0)
    print(msg)
    print("\n")

    t = [-10 + decimal.Decimal(time / 10) for time in range(200)]
    y = [hun(list(map(add, [decimal.Decimal(2), decimal.Decimal(2)], [time*x_i for x_i in [decimal.Decimal(0.3), decimal.Decimal(0.4)]]))) for time in t]
    plt.plot(t, y)
    plt.title("hun(2 + 0.3s, 2 + 0.4s)")
    plt.show()

def test_armijo(algorithm):
    if algorithm == "Basic":
        print("******************")
        print("Basic Armijo")
        print("******************\n")
    else:
        print("******************")
        print("Cubic Armijo")
        print("******************\n")

    #############################################
    # Important
    #############################################

    # In the case of armijo search, s > 0 is a requirement
    # To simulate real cases, the direction is reversed compared to the examples before if the previous direction goes away from the minimum

    print("Next value of f(x, y) = 3*x^2 + y^2 + 4 in the direction (dx, dy) = (-1, -1), starting at (x, y) = (1, 1):")
    grad = gradient_descent.gradient(fun, [decimal.Decimal(1), decimal.Decimal(1)])
    (s_0, f_0, msg) = line_search.armijo_line_search(fun, [-decimal.Decimal(1), -decimal.Decimal(1)], [decimal.Decimal(1),
                                                    decimal.Decimal(1)], grad, algorithm)
    print("f(x - " + str(s_0) + ", y - " + str(s_0) + ") = " + str(fun([decimal.Decimal(1) - s_0, decimal.Decimal(1) - s_0])))
    print("s =", s_0)
    print(msg)
    print("\n")

    print("Next value of f(x, y) = x^2 + y^2 - 2xy in the direction (dx, dy) = (-0.3, -0.4), starting at (x, y) = (1, 1):")
    grad = gradient_descent.gradient(gun, [decimal.Decimal(1), decimal.Decimal(1)])
    (s_0, f_0, msg) = line_search.armijo_line_search(gun, [-decimal.Decimal(0.3), -decimal.Decimal(0.4)],
                                                    [decimal.Decimal(1), decimal.Decimal(1)], grad, algorithm)
    print("f(x + " + str(-s_0 * decimal.Decimal(0.3)) + ", y + " + str(-s_0 * decimal.Decimal(0.4)) + ") = " +
          str(gun([decimal.Decimal(1) - s_0 * decimal.Decimal(0.3), decimal.Decimal(1) - s_0 * decimal.Decimal(0.4)])))
    print("s =", s_0)
    print(msg)
    print("\n")

    print(
        "Next value of f(x, y) = 10x^4 - 20x^2 y + 10y^2 + x^2 - 2x + 5 in the direction (dx, dy) = (-0.3, -0.4), starting at (x, y) = (2, 2):")
    grad = gradient_descent.gradient(hun, [decimal.Decimal(2), decimal.Decimal(2)])
    (s_0, f_0, msg) = line_search.armijo_line_search(hun, [-decimal.Decimal(0.3), -decimal.Decimal(0.4)],
                                                    [decimal.Decimal(2), decimal.Decimal(2)], grad, algorithm)
    print("f(x + " + str(-s_0 * decimal.Decimal(0.3)) + ", y + " + str(-s_0 * decimal.Decimal(0.4)) + ") = " +
          str(hun([decimal.Decimal(2) - s_0 * decimal.Decimal(0.3), decimal.Decimal(2) - s_0 * decimal.Decimal(0.4)])))
    print("s =", s_0)
    print(msg)
    print("\n")

    print(
        "Next value of f(x, y, z) = x^2 + 2y^2 + 2z^2 + 2xy + 2yz in the direction (dx, dy, dz) = (-0.3, -0.4, -0.5), starting at (x, y, z) = (2, 2, 2):")
    grad = gradient_descent.gradient(pun, [decimal.Decimal(2), decimal.Decimal(2), decimal.Decimal(2)])
    (s_0, f_0, msg) = line_search.armijo_line_search(pun, [-decimal.Decimal(0.3), -decimal.Decimal(0.4), -decimal.Decimal(0.5)],
                                                    [decimal.Decimal(2), decimal.Decimal(2), decimal.Decimal(2)], grad, algorithm)
    print("f(x + " + str(-s_0 * decimal.Decimal(0.3)) + ", y + " + str(-s_0 * decimal.Decimal(0.4)) + ", z + " +
          str(-s_0 * decimal.Decimal(0.5)) + ") = " +
          str(pun([decimal.Decimal(2) - s_0 * decimal.Decimal(0.3), decimal.Decimal(2) - s_0 * decimal.Decimal(0.4),
                   decimal.Decimal(2) - s_0 * decimal.Decimal(0.5)])))
    print("s =", s_0)
    print(msg)
    print("\n")

    t = [-10 + decimal.Decimal(time / 10) for time in range(200)]
    y = [hun(list(map(add, [decimal.Decimal(2), decimal.Decimal(2)], [time*x_i for x_i in [decimal.Decimal(0.3), decimal.Decimal(0.4)]]))) for time in t]
    plt.plot(t, y)
    plt.title("hun(2 + 0.3s, 2 + 0.4s)")
    plt.show()


if __name__ == "__main__":
    # set precision of decimal to maximum
    decimal.getcontext().prec = 28

    # choose testing set
    print("Choose testing set:\n1. Exact line search Newton-Rhapson\n2. Exact line search Bisection\n"
          + "3. Exact line search Quadratic interpolation\n4. Armijo line search basic\n5. Armijo line search cubic\n")
    test_mode = int(input())
    print("")

    if test_mode == 1:
        test_exact("Newton")
    elif test_mode == 2:
        test_exact("Bisection")
    elif test_mode == 3:
        test_exact("Quadint")
    elif test_mode == 4:
        test_armijo("BasicArm")
    elif test_mode == 5:
        test_armijo("CubicArm")