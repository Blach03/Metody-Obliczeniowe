import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import trapz, simps
from scipy.special import roots_legendre

def f(x):
    return 4 / (1 + x ** 2)

def midpoint_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(n):
        x_mid = (a + i * h) + h / 2
        integral += f(x_mid)
    integral *= h
    return integral

def calculate_integral_error(method, f, a, b, m):
    errors = []
    for i in range(1, m + 1):
        n = 2 ** i + 1
        if method == 'midpoint':
            integral = midpoint_rule(f, a, b, n)
        elif method == 'trapezoidal':
            x = np.linspace(a, b, n)
            y = f(x)
            integral = trapz(y, x)
        elif method == 'simpson':
            x = np.linspace(a, b, n)
            y = f(x)
            integral = simps(y, x)
        elif method == 'gauss_legendre':
            if i < 8:
                x, w = roots_legendre(n)
                integral = np.sum(w * np.abs(f(x)))
        true_value, _ = quad(f, a, b)
        error = np.abs(integral - true_value)
        relative_error = error / np.abs(true_value)
        if method == 'gauss_legendre':
            relative_error -= 1
        errors.append(relative_error)
    return errors

def plot_errors(errors, method):
    n_values = 2 ** np.arange(1, len(errors) + 1)
    plt.plot(n_values, errors, label=method)

# Zadanie 1
a = 0
b = 1
m = 25
methods = ['midpoint', 'trapezoidal', 'simpson', 'gauss_legendre']
plt.figure(figsize=(10, 6))

for method in methods:
    errors = calculate_integral_error(method, f, a, b, m)
    plot_errors(errors, method)

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of function evaluations')
plt.ylabel('Relative error')
plt.title('Convergence of numerical integration methods')
plt.legend()
plt.show()

# Zadanie 2
n_values = np.arange(1, 100)
errors_gauss_legendre = []

for n in n_values:
    x, w = roots_legendre(n)
    integral_approx = np.sum(w * f(x))
    true_value, _ = quad(f, a, b)
    error = np.abs(integral_approx - true_value)
    relative_error = error / np.abs(true_value)
    errors_gauss_legendre.append(relative_error)

plt.figure(figsize=(10, 6))
plot_errors(errors_gauss_legendre, 'Gauss-Legendre')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of function evaluations')
plt.ylabel('Relative error')
plt.title('Convergence of Gauss-Legendre quadrature')
plt.legend()
plt.show()

