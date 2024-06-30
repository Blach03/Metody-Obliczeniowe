import sympy as sp
import pandas as pd

#Zadanie 1
x, y = sp.symbols('x y')

f1 = x**2 - 4*x*y + y**2
f2 = x**4 - 4*x*y + y**4
f3 = 2*x**3 - 3*x**2 - 6*x*y*(x - y - 1)
f4 = (x - y)**4 + x**2 - y**2 - 2*x + 2*y + 1

functions = [f1, f2, f3, f4]


def find_critical_points(f):
    grad = sp.Matrix([sp.diff(f, var) for var in (x, y)])
    stationary_points = sp.solve(grad, (x, y), dict=True)
    return stationary_points


def classify_critical_points(f, points):
    hessian = sp.Matrix([[sp.diff(f, var1, var2) for var1 in (x, y)] for var2 in (x, y)])
    classifications = []
    for point in points:
        hessian_at_point = hessian.subs(point)
        det_hessian = hessian_at_point.det()
        eigenvals = hessian_at_point.eigenvals()
        real_eigenvals = [sp.re(val) for val in eigenvals]
        if det_hessian == 0:
            classification = 'indeterminate due to zero determinant'
        elif all(val > 0 for val in real_eigenvals):
            classification = 'minimum'
        elif all(val < 0 for val in real_eigenvals):
            classification = 'maximum'
        else:
            classification = 'saddle point'
        classifications.append((point, classification))
    return classifications

critical_points_and_classifications = {}
for i, f in enumerate(functions):
    points = find_critical_points(f)
    classifications = classify_critical_points(f, points)
    critical_points_and_classifications[f'Function {i+1}'] = classifications


flat_results = []
for func_name, points in critical_points_and_classifications.items():
    for point, classification in points:
        flat_results.append({
            'Function': func_name,
            'Point': point,
            'Classification': classification
        })

df_results = pd.DataFrame(flat_results)


print(df_results)

#Zadanie 2
import numpy as np
import matplotlib.pyplot as plt

n = 20
k = 50
x_start = np.array([0, 0])
x_end = np.array([20, 20])
r = np.random.uniform(0, 20, (k, 2))
lambda_1 = 1
lambda_2 = 1
epsilon = 1e-13
iterations = 400

def F(x, r, lambda_1, lambda_2, epsilon):
    term1 = lambda_1 * np.sum([1 / (epsilon + np.linalg.norm(x[i] - r[j])**2) for i in range(n+1) for j in range(k)])
    term2 = lambda_2 * np.sum([np.linalg.norm(x[i+1] - x[i])**2 for i in range(n)])
    return term1 + term2

def grad_F(x, r, lambda_1, lambda_2, epsilon):
    grad = np.zeros_like(x)
    for i in range(1, n):
        grad_term1 = lambda_1 * np.sum([-2 * (x[i] - r[j]) / (epsilon + np.linalg.norm(x[i] - r[j])**2)**2 for j in range(k)], axis=0)
        grad_term2 = lambda_2 * (2 * (x[i] - x[i-1]) - 2 * (x[i+1] - x[i]))
        grad[i] = grad_term1 + grad_term2
    return grad

def golden_section_search(f, a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

def gradient_descent(x, r, lambda_1, lambda_2, epsilon, iterations):
    alpha = 1
    values = []
    for _ in range(iterations):
        grad = grad_F(x, r, lambda_1, lambda_2, epsilon)
        f = lambda a: F(x - a * grad, r, lambda_1, lambda_2, epsilon)
        alpha = golden_section_search(f, 0, alpha)
        x -= alpha * grad
        values.append(F(x, r, lambda_1, lambda_2, epsilon))
    return x, values


results = []

for i in range(5):
    x = np.linspace(x_start, x_end, n+1)
    x[1:n] = np.random.uniform(0, 20, (n-1, 2))
    x[0] = x_start
    x[-1] = x_end
    
    x_opt, values = gradient_descent(x, r, lambda_1, lambda_2, epsilon, iterations)
    results.append((x_opt, values))


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i, (x_opt, values) in enumerate(results):
    plt.plot(values, label=f'Initialization {i+1}')
plt.xlabel('Iteracje')
plt.ylabel('Wartość funkcji celu')
plt.title('Wartość funkcji celu w zależności od iteracji')
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(r[:, 0], r[:, 1], c='r', label='Przeszkody')
for i, (x_opt, _) in enumerate(results):
    plt.plot(x_opt[:, 0], x_opt[:, 1], label=f'Path {i+1}')
plt.scatter([x_start[0], x_end[0]], [x_start[1], x_end[1]], c='g', label='Punkty startowe i końcowe')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Najkrótsza ścieżka robota')

plt.tight_layout()
plt.show()

