import line_search
import decimal
import gradient_descent
import newton
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

def test(algorithm, method="gd"):
    print("*********************")
    print(algorithm)
    print("*********************\n")

    print("1. Minimum of f(x, y) = 3*x^2 + y^2 + 4, starting at (x, y) = (3, 3):\n")

    if method == "gd":
        (x, f, msg) = gradient_descent.gradient_descent(fun, [decimal.Decimal(3), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg) = newton.newton(fun, [decimal.Decimal(3), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y) = (" + str(x[0]) + ", " + str(x[1]) + ")")
    print("f(x, y) =", f)
    print(msg)
    print("\n")

    print("2. Minimum of f(x, y) = x^2 + y^2 - 2xy, starting at (x, y) = (1, 0):\n")

    if method == "gd":
        (x, f, msg) = gradient_descent.gradient_descent(gun, [decimal.Decimal(1), decimal.Decimal(0)], algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg) = newton.newton(gun, [decimal.Decimal(1), decimal.Decimal(0)], algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y) = (" + str(x[0]) + ", " + str(x[1]) + ")")
    print("f(x, y) =", f)
    print(msg)
    print("\n")

    print("3. Minimum of f(x, y) = 10x^4 - 20x^2 y + 10y^2 + x^2 - 2x + 5, starting at (x, y) = (-1, 3):\n")

    if method == "gd":
        (x, f, msg) = gradient_descent.gradient_descent(hun, [decimal.Decimal(-1), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg) = newton.newton(hun, [decimal.Decimal(-1), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y) = (" + str(x[0]) + ", " + str(x[1]) + ")")
    print("f(x, y) =", f)
    print(msg)
    print("\n")

    print("4. Minimum of f(x, y, z) = x^2 + 2y^2 + 2z^2 + 2xy + 2yz, starting at (x, y, z) = (2, 4, 10):\n")

    if method == "gd":
        (x, f, msg) = gradient_descent.gradient_descent(pun, [decimal.Decimal(2), decimal.Decimal(4), decimal.Decimal(10)],
                                                        algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg) = newton.newton(pun, [decimal.Decimal(2), decimal.Decimal(4), decimal.Decimal(10)],
                                                        algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y, z) = (" + str(x[0]) + ", " + str(x[1]) + "," + str(x[2]) + ")")
    print("f(x, y, z) =", f)
    print(msg)
    print("\n")

if __name__ == "__main__":
    # set precision of decimal to maximum
    decimal.getcontext().prec = 28

    # choose testing set
    print("Choose testing set:\n1. Gradient descent\n2. Newton's method\n")
    test_mode = int(input())
    print("")

    print("Input line search algorithm of choice:\n")
    algorithm = input()

    if test_mode == 1:
        if algorithm != "0":
            test(algorithm, "gd")
        else:
            test("BasicArm", "gd")
    else:
        if algorithm != "0":
            test(algorithm, "newton")
        else:
            test("BasicArm", "newton")