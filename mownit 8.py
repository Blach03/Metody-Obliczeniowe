from scipy.optimize import root_scalar

def func_a(x):
    return x ** 3 - 5 * x

def func_b(x):
    return x ** 3 - 3 * x + 1

def func_c(x):
    return 2 - x ** 5

def func_d(x):
    return x ** 4 - 4.29 * x ** 2 - 5.29

methods = ['bisect', 'secant', 'brentq']
funcs = {'a': func_a, 'b': func_b, 'c': func_c, 'd': func_d}
initial_guesses = {'a': 1, 'b': 1, 'c': 0.01, 'd': 0.8}
brackets = {'a': [-5, 5], 'b': [-5, 5], 'c': [-5, 5], 'd': [-3, 3]}

for label, func in funcs.items():
    print(f"Results for function ({label}):")
    for method in methods:
        try:
            if method in ['bisect', 'brentq']:
                result = root_scalar(func, method=method, bracket=brackets[label])
            else:
                result = root_scalar(func, method=method, x0=initial_guesses[label])
            print(f"Method: {method}, root: {result.root}, iterations: {result.iterations}")
        except ValueError as e:
            print(f"Method: {method}, Error: {e}")
    print()


import numpy as np
import matplotlib.pyplot as plt

def g1(x):
    return (x**2 + 2) / 3

def g2(x):
    return np.sqrt(3 * x - 2)

def g3(x):
    return 3 - 2 / x

def g4(x):
    return (x**2 - 2) / (2 * x - 3)


def dg1(x):
    return 2 * x / 3

def dg2(x):
    return 1.5 / np.sqrt(3 * x - 2)

def dg3(x):
    return 2 / x**2

def dg4(x):
    return (2 * x * (2 * x - 3) - (x**2 - 2) * 2) / (2 * x - 3)**2




x_range = np.linspace(1, 2, 400) 
plt.figure(figsize=(12, 8))
plt.plot(x_range, np.abs(dg1(x_range)), label='|dg1(x)|')
plt.plot(x_range, np.abs(dg2(x_range)), label='|dg2(x)|', linestyle='--')
plt.plot(x_range, np.abs(dg3(x_range)), label='|dg3(x)|', linestyle='-.')
plt.plot(x_range, np.abs(dg4(x_range)), label='|dg4(x)|', linestyle=':')
plt.axhline(1, color='red', linestyle='--')
plt.title('Derivative Magnitudes and Convergence Criterion')
plt.xlabel('x')
plt.ylabel('|g\'(x)|')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

import random
eps = random.uniform(-0.01, 0.01)

x0 = 1.5 + eps


x_exact = 2


num_iterations = 10

def iterate(g, x0, num_iterations):
    x_vals = [x0]
    for _ in range(num_iterations):
        x_next = g(x_vals[-1])
        x_vals.append(x_next)
    return x_vals

def plot_errors(g, x0, g_name):
    x_vals = iterate(g, x0, num_iterations)
    errors = [abs(x - x_exact) for x in x_vals]
    plt.semilogy(range(len(errors)), errors, label=g_name)


plot_errors(g1, 1.5, 'g1')
plot_errors(g2, 1.5, 'g2')
plot_errors(g3, 1.5, 'g3')
plot_errors(g4, 1.6, 'g4')
plt.xlabel('Iteration')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.show()


import math


def f_a(x):
    return x**3 - 2*x - 5

def f_prime_a(x):
    return 3*x**2 - 2


def f_b(x):
    return math.exp(-x) - x

def f_prime_b(x):
    return -math.exp(-x) - 1


def f_c(x):
    return x * math.sin(x) - 1

def f_prime_c(x):
    return math.sin(x) + x * math.cos(x)


def newton_method(f, f_prime, x0, tolerance=1e-15, max_iterations=100):
    x = x0
    for iteration in range(max_iterations):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_next = x - fx / fpx
        if abs(x_next - x) < tolerance:
            return x_next, iteration + 1
        x = x_next
    raise ValueError("Exceeded maximum iterations")


x0 = 1.5 + eps 


target_precisions = {'24-bit': 1e-7, '53-bit': 1e-15}

for label, tol in target_precisions.items():
    print(f"\nResults for {label} precision:")
    try:
        root_a, iterations_a = newton_method(f_a, f_prime_a, x0, tol)
        print(f"Equation (a): Root = {root_a}, Iterations = {iterations_a}")
    except ValueError as e:
        print(f"Equation (a): {e}")

    try:
        root_b, iterations_b = newton_method(f_b, f_prime_b, x0, tol)
        print(f"Equation (b): Root = {root_b}, Iterations = {iterations_b}")
    except ValueError as e:
        print(f"Equation (b): {e}")

    try:
        root_c, iterations_c = newton_method(f_c, f_prime_c, x0, tol)
        print(f"Equation (c): Root = {root_c}, Iterations = {iterations_c}")
    except ValueError as e:
        print(f"Equation (c): {e}")



print()
print("zad4")


def f1(x1, x2):
    return x1**2 + x2**2 - 1

def f2(x1, x2):
    return x1**2 - x2


def jacobian(x1, x2):
    return np.array([[2 * x1, 2 * x2], [2 * x1, -1]])


def newton_system(F, J, x0, tolerance=1e-10, max_iterations=100):
    x = np.array(x0)
    for iteration in range(max_iterations):
        Fx = np.array(F(x[0], x[1]))
        Jx = J(x[0], x[1])
        delta = np.linalg.solve(Jx, Fx)
        x_next = x - delta
        if np.linalg.norm(x_next - x) < tolerance:
            return x_next, iteration + 1
        x = x_next
    raise ValueError("Exceeded maximum iterations")


def system(x1, x2):
    return np.array([f1(x1, x2), f2(x1, x2)])


x1_exact = np.sqrt((np.sqrt(5) - 1) / 2)
x2_exact = (np.sqrt(5) - 1) / 2
exact_solution = np.array([x1_exact, x2_exact])

x0 = [0.7, 0.7]


try:
    solution, iterations = newton_system(system, jacobian, x0)
    print(f"Solution: {solution}, Iterations: {iterations}")

    relative_error = np.linalg.norm(solution - exact_solution) / np.linalg.norm(exact_solution)
    print(f"Relative error: {relative_error:.10f}")

except ValueError as e:
    print(e)
