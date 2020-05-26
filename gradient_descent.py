import decimal

# helper function which calculates the gradient numerically
def gradient(f, x_init, epsilon = decimal.Decimal(1e-10)):
    n = len(x_init)
    grad = [0 for _ in range(n)]

    for i in range(n):
        # add epsilon variation to x_i
        x_up = x_init
        x_up[i] += epsilon
        x_down = x_init
        x_down[i] -= epsilon

        # calculate derivative numerically using symmetric Newton difference
        grad[i] = (f(x_up) - f(x_down)) / (2 * epsilon)

    return grad