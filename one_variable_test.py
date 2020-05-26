import decimal
import bisection
import newton_rhapson
import quad_interpolation

##############################
# Custom functions for testing
##############################

# Monotone function example

def monotonic_function(x):
    return decimal.Decimal.ln(x)

def derivative_monotonic_function(x):
    return decimal.Decimal(1) / x

# Convex function example

def convex_function(x):
    a = decimal.Decimal(3)
    c = decimal.Decimal(-2)
    x_0 = decimal.Decimal(4)
    return a*(x-x_0)*(x-x_0) + c

def explicit_derivative(x):
    a = decimal.Decimal(3)
    x_0 = decimal.Decimal(4)
    return 2*a*(x-x_0)

def explicit_2nd_derivative(x):
    return decimal.Decimal(6)

# Primjer iz predavanja

def lecture_function(x):
    return x*x*x - decimal.Decimal(2) * x - decimal.Decimal(5)

def lecture_derivative(x):
    return decimal.Decimal(3) * x*x -decimal.Decimal(2)

def lecture_2nd_derivative(x):
    return decimal.Decimal(6) * x

###################
# Testing functions
###################

def test_bisection():
    # find root of function
    (x_0, f_0, message) = bisection.bisection_zero(monotonic_function, decimal.Decimal(0.5), decimal.Decimal(1000))
    print("Root of function f(x) = ln(x):")
    print("f(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

    # test max_iter parameter
    print("Root of function f(x) = ln(x):")
    (x_0, f_0, message) = bisection.bisection_zero(monotonic_function, decimal.Decimal(0.5), decimal.Decimal(1000),
                                                   decimal.Decimal(1e-15), 20)
    print("f(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

    # find minimum of function passing function itself as argument
    print("Minimum of function f(x) = 3*(x-4)^2 - 2 (passing function itself):")
    (x_0, f_0, message) = bisection.bisection_minimum(convex_function, decimal.Decimal(-50), decimal.Decimal(50))
    print("f(" + str(x_0) + ") = " + str(f_0))
    print("f'(" + str(x_0) + ") = " + str(explicit_derivative(x_0)))
    print(message)
    print("")

    # find minimum of function passing derivative as argument
    print("Minimum of function f(x) = 3*(x-4)^2 - 2 (passing explicit derivative):")
    (x_0, f_0, message) = bisection.bisection_minimum(explicit_derivative, decimal.Decimal(-50),
                                                      decimal.Decimal(50), False)
    print("f(" + str(x_0) + ") = " + str(convex_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

    ##############################
    # Testni primjer iz predavanja
    ##############################

    print("Minimum of function f(x) = x^3 - 2x - 5 (passing function itself):")
    (x_0, f_0, message) = bisection.bisection_minimum(lecture_function, decimal.Decimal(0), decimal.Decimal(2),
                                                      True, decimal.Decimal(1e-5), 50)
    print("f(" + str(x_0) + ") = " + str(f_0))
    print("f'(" + str(x_0) + ") = " + str(lecture_derivative(x_0)))
    print(message)
    print("")

    print("Korespondira tacno primjeru iz predavanja:\n")
    print("Minimum of function f(x) = x^3 - 2x - 5 (passing explicit derivative):")
    (x_0, f_0, message) = bisection.bisection_minimum(lecture_derivative, decimal.Decimal(0), decimal.Decimal(2),
                                                      False,
                                                      decimal.Decimal(1e-5), 50)
    print("f(" + str(x_0) + ") = " + str(lecture_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

def test_newton():
    # find root of function
    (x_0, f_0, message) = newton_rhapson.newton_rhapson(monotonic_function, derivative_monotonic_function, decimal.Decimal(2))
    print("Root of function f(x) = ln(x) (closed form):")
    print("f(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

    # test numeric derivative
    df_dx = newton_rhapson.numeric_derivative(monotonic_function)
    (x_0, f_0, message) = newton_rhapson.newton_rhapson(monotonic_function, df_dx, decimal.Decimal(2))
    print("Root of function f(x) = ln(x) (numeric derivative):")
    print("f(" + str(x_0) + ") = " + str(f_0))
    print(message)
    print("")

    # find minimum of function passing closed form for derivatives
    print("Minimum of function f(x) = 3*(x-4)^2 - 2 (passing closed form):")
    (x_0, f_0, message) = newton_rhapson.newton_rhapson(explicit_derivative, explicit_2nd_derivative, decimal.Decimal(10))
    print("f'(" + str(x_0) + ") = " + str(convex_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(f_0))
    print("f''(" + str(x_0) + ") = " + str(explicit_2nd_derivative(x_0)))
    print(message)
    print("")

    # find minimum of function passing numeric derivatives
    df_dx = newton_rhapson.numeric_derivative(convex_function)
    d2f_dx2 = newton_rhapson.numeric_derivative(df_dx)

    print("Minimum of function f(x) = 3*(x-4)^2 - 2 (passing numeric derivative):")
    (x_0, f_0, message) = newton_rhapson.newton_rhapson(df_dx, d2f_dx2, decimal.Decimal(10))
    print("f'(" + str(x_0) + ") = " + str(convex_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(explicit_derivative(x_0)))
    print("f''(" + str(x_0) + ") = " + str(explicit_2nd_derivative(x_0)))
    print(message)
    print("")

    ##############################
    # Testni primjer iz predavanja
    ##############################

    print("Minimum of function f(x) = x^3 - 2x - 5 (passing numeric derivative):")
    df_dx = newton_rhapson.numeric_derivative(lecture_function)
    d2f_dx2 = newton_rhapson.numeric_derivative(df_dx)

    (x_0, f_0, message) = newton_rhapson.newton_rhapson(df_dx, d2f_dx2, decimal.Decimal(1.5), decimal.Decimal(1e-15), 20)
    print("f(" + str(x_0) + ") = " + str(lecture_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(lecture_derivative(x_0)))
    print("f''(" + str(x_0) + ") = " + str(lecture_2nd_derivative(x_0)))
    print(message)
    print("")

    print("Korespondira tacno primjeru iz predavanja:\n")

    print("Minimum of function f(x) = x^3 - 2x - 5 (passing closed form):")
    (x_0, f_0, message) = newton_rhapson.newton_rhapson(lecture_derivative, lecture_2nd_derivative, decimal.Decimal(1.5), decimal.Decimal(1e-15), 20)
    print("f(" + str(x_0) + ") = " + str(lecture_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(lecture_derivative(x_0)))
    print("f''(" + str(x_0) + ") = " + str(lecture_2nd_derivative(x_0)))
    print(message)
    print("")

def test_quad():
    print("Minimum of function f(x) = 3*(x-4)^2 - 2:")
    (x_0, f_0, message) = quad_interpolation.quadratic_interpolation(convex_function, decimal.Decimal(-10), decimal.Decimal(10))
    print("f'(" + str(x_0) + ") = " + str(convex_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(f_0))
    print("f''(" + str(x_0) + ") = " + str(explicit_2nd_derivative(x_0)))
    print(message)
    print("")

    print("Korespondira tacno primjeru iz predavanja:\n")

    print("Minimum of function f(x) = x^3 - 2x - 5:")
    (x_0, f_0, message) = quad_interpolation.quadratic_interpolation(lecture_function, decimal.Decimal(0), decimal.Decimal(2), decimal.Decimal(1e-15), 50)
    print("f(" + str(x_0) + ") = " + str(lecture_function(x_0)))
    print("f'(" + str(x_0) + ") = " + str(lecture_derivative(x_0)))
    print("f''(" + str(x_0) + ") = " + str(lecture_2nd_derivative(x_0)))
    print(message)
    print("")

if __name__ == "__main__":
    # set precision of decimal to maximum
    decimal.getcontext().prec = 28

    # choose testing set
    print("Choose testing set:\n1. bisection\n2. Newton-Rhapson\n3. Quadratic interpolation\n")
    test_mode = int(input())
    print("")

    if test_mode == 1:
        test_bisection()
    elif test_mode == 2:
        test_newton()
    elif test_mode == 3:
        test_quad()