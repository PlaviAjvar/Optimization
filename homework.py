import decimal
import gradient_descent
import newton
import matplotlib.pyplot as plt
import numpy as np

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
    plot_title = ""

    print("*********************")

    if method == "gd":
        print("Gradient descent")
        plot_title += "Gradient descent"
    else:
        print("Newton's method")
        plot_title += "Newton's method"

    print("*********************")
    plot_title += " / "

    if algorithm == "Newton":
        print("Newton-Rhapson")
        plot_title += "Newton-Rhapson"
    elif algorithm == "Quadint":
        print("Quadratic interpolation")
        plot_title += "Quadratic interpolation"
    elif algorithm == "Bisection":
        print("Bisection method")
        plot_title += "Bisection method"
    elif algorithm == "BasicArm":
        print("Basic Armijo")
        plot_title += "Basic Armijo"
    else:
        print("Cubic Armijo")
        plot_title += "Cubic Armijo"

    print("*********************\n")

    print("1. Minimum of f(x, y) = x^2 + y^2 - 2xy, starting at (x, y) = (1, 0):\n")

    if method == "gd":
        (x, f, msg, x_v, f_v) = gradient_descent.gradient_descent(gun, [decimal.Decimal(1), decimal.Decimal(0)], algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg, x_v, f_v) = newton.newton(gun, [decimal.Decimal(1), decimal.Decimal(0)], algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y) = (" + str(x[0]) + ", " + str(x[1]) + ")")
    print("f(x, y) =", f)
    print(msg)
    print("\n")

    # contour_plot(x_v, gun)

    print("2. Minimum of f(x, y) = 10x^4 - 20x^2 y + 10y^2 + x^2 - 2x + 5, starting at (x, y) = (-1, 3):\n")

    if method == "gd":
        (x, f, msg, x_v, f_v) = gradient_descent.gradient_descent(hun, [decimal.Decimal(-1), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg, x_v, f_v) = newton.newton(hun, [decimal.Decimal(-1), decimal.Decimal(3)], algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y) = (" + str(x[0]) + ", " + str(x[1]) + ")")
    print("f(x, y) =", f)
    print(msg)
    print("\n")

    contour_plot(x_v, hun, plot_title)
    plot_energy(x_v, f_v, plot_title)

    print("3. Minimum of f(x, y, z) = x^2 + 2y^2 + 2z^2 + 2xy + 2yz, starting at (x, y, z) = (2, 4, 10):\n")

    if method == "gd":
        (x, f, msg, x_v, f_v) = gradient_descent.gradient_descent(pun, [decimal.Decimal(2), decimal.Decimal(4), decimal.Decimal(10)],
                                                        algorithm, decimal.Decimal(0.005))
    else:
        (x, f, msg, x_v, f_v) = newton.newton(pun, [decimal.Decimal(2), decimal.Decimal(4), decimal.Decimal(10)],
                                                        algorithm, decimal.Decimal(0.005))

    print("The minimum is at (x, y, z) = (" + str(x[0]) + ", " + str(x[1]) + "," + str(x[2]) + ")")
    print("f(x, y, z) =", f)
    print(msg)
    print("\n")


# helper function to plot energy over time

def plot_energy(x, y, title):
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.show()


# helper function for drawing contour plot
def contour_plot(x_v, f, title):
    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    start, stop, n_values = -4, 4, 800

    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = [[0] * n_values for _ in range(n_values)]

    for i in range(n_values):
        for j in range(n_values):
            Z[i][j] = f([X[i][j], Y[i][j]])

    plt.plot([x[0] for x in x_v], [x[1] for x in x_v], "-ro")
    m = 50
    levels = np.linspace(-100, 100, m)

    for i in range(len(x_v)):
        np.append(levels, f(x_v[i]))

    plt.contour(X, Y, Z, colors='black', levels=levels)
    plt.title(title)
    #plt.colorbar(cp)

    plt.show()

if __name__ == "__main__":
    # set precision of decimal to maximum
    decimal.getcontext().prec = 28

    # 1. Solve all 3 functions using exact and backtracking line search (ok)
    # 2. Draw solutions on contour graphs.
    # 3. Compare solutions and iteration counts for Gradient descent and Newton's method (ok)

    iter_gd_newton = test("Newton", "gd")
    iter_gd_armijo = test("CubicArm", "gd")

    iter_new_newton = test("Newton", "newton")
    iter_new_armijo = test("CubicArm", "newton")