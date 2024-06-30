import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

def lagrange_interpolation_equidistant_nodes(f, n, a, b):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    poly = lagrange(x, y)
    return poly

def cubic_spline_interpolation_equidistant_nodes(f, n, a, b):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    spline = CubicSpline(x, y)
    return spline

def chebyshev_nodes(n, a, b):
    j = np.arange(n + 1)
    xj = np.cos((2*j + 1) * np.pi / (2*(n + 1)))
    return xj

def lagrange_interpolation_chebyshev_nodes(f, n, a, b):
    x = chebyshev_nodes(n, a, b)
    y = f(x)
    poly = lagrange(x, y)
    return poly

f1 = lambda x: 1 / (1 + 25 * x**2)
f2 = lambda x: np.exp(np.cos(x))


interval_f1 = (-1, 1)
interval_f2 = (0, 2 * np.pi)

n = 12

poly_f1_equidistant = lagrange_interpolation_equidistant_nodes(f1, n, *interval_f1)
poly_f2_equidistant = lagrange_interpolation_equidistant_nodes(f2, n, *interval_f2)


spline_f1_equidistant = cubic_spline_interpolation_equidistant_nodes(f1, n, *interval_f1)
spline_f2_equidistant = cubic_spline_interpolation_equidistant_nodes(f2, n, *interval_f2)


poly_f1_chebyshev = lagrange_interpolation_chebyshev_nodes(f1, n, *interval_f1)
poly_f2_chebyshev = lagrange_interpolation_chebyshev_nodes(f2, n, *interval_f2)


sample_points = np.linspace(*interval_f1, 1000)
interpolation_points_equidistant = np.linspace(*interval_f1, 10 * len(poly_f1_equidistant.c))
interpolation_points_chebyshev = chebyshev_nodes(n, *interval_f1)

f1_values = f1(sample_points)
interpolation_values_equidistant = poly_f1_equidistant(interpolation_points_equidistant)
spline_values_equidistant = spline_f1_equidistant(interpolation_points_equidistant)
chenyshev_values = poly_f1_chebyshev(interpolation_points_chebyshev)


plt.figure(figsize=(10, 6))
plt.plot(sample_points, f1_values, label='f1(x)', color='blue')
plt.plot(interpolation_points_equidistant, interpolation_values_equidistant, label='Lagrange Interpolation', linestyle='--', color='green')
plt.plot(interpolation_points_equidistant, spline_values_equidistant, label='Cubic Spline Interpolation', linestyle='-.', color='orange')
plt.plot(interpolation_points_chebyshev, chenyshev_values, label='Chebyshev Interpolation', linestyle='-.', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja funkcji Rungego (f1(x))')
plt.legend()
plt.grid(True)
plt.gca().set_ylim(bottom=-0.5)
plt.show()



n_values = range(4, 51)


sample_points1 = np.linspace(*interval_f1, 500)
sample_points2 = np.linspace(*interval_f2, 500)


error_norms_f1_equidistant = []
error_norms_f1_chebyshev = []
error_norms_f1_spline = []
error_norms_f2_equidistant = []
error_norms_f2_chebyshev = []
error_norms_f2_spline = []


for n in n_values:

    poly_f1_equidistant = lagrange_interpolation_equidistant_nodes(f1, n, *interval_f1)
    error_norm_f1_equidistant = np.linalg.norm(f1(sample_points1) - poly_f1_equidistant(sample_points1))
    error_norms_f1_equidistant.append(error_norm_f1_equidistant)


    poly_f1_chebyshev = lagrange_interpolation_chebyshev_nodes(f1, n, *interval_f1)
    error_norm_f1_chebyshev = np.linalg.norm(f1(sample_points1) - poly_f1_chebyshev(sample_points1))
    error_norms_f1_chebyshev.append(error_norm_f1_chebyshev)


    spline_f1_equidistant = cubic_spline_interpolation_equidistant_nodes(f1, n, *interval_f1)
    error_norm_f1_spline = np.linalg.norm(f1(sample_points1) - spline_f1_equidistant(sample_points1))
    error_norms_f1_spline.append(error_norm_f1_spline)


    poly_f2_equidistant = lagrange_interpolation_equidistant_nodes(f2, n, *interval_f2)
    error_norm_f2_equidistant = np.linalg.norm(f2(sample_points2) - poly_f2_equidistant(sample_points2))
    error_norms_f2_equidistant.append(error_norm_f2_equidistant)


    poly_f2_chebyshev = lagrange_interpolation_chebyshev_nodes(f2, n, *interval_f2)
    error_norm_f2_chebyshev = np.linalg.norm(f2(sample_points2) - poly_f2_chebyshev(sample_points2))
    error_norms_f2_chebyshev.append(error_norm_f2_chebyshev)


    spline_f2_equidistant = cubic_spline_interpolation_equidistant_nodes(f2, n, *interval_f2)
    error_norm_f2_spline = np.linalg.norm(f2(sample_points2) - spline_f2_equidistant(sample_points2))
    error_norms_f2_spline.append(error_norm_f2_spline)

plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
plt.plot(n_values, error_norms_f1_equidistant, label='Lagrange Equidistant Nodes', marker='o')
plt.plot(n_values, error_norms_f1_chebyshev, label='Lagrange Chebyshev Nodes', marker='o')
plt.plot(n_values, error_norms_f1_spline, label='Cubic Spline Equidistant Nodes', marker='o')
plt.title('Norm of Error vs. Number of Interpolation Points for f1(x)')
plt.xlabel('Number of Interpolation Points (n)')
plt.ylabel('Norm of Error')
plt.yscale('log')
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(n_values, error_norms_f2_equidistant, label='Lagrange Equidistant Nodes', marker='o')
plt.plot(n_values, error_norms_f2_chebyshev, label='Lagrange Chebyshev Nodes', marker='o')
plt.plot(n_values, error_norms_f2_spline, label='Cubic Spline Equidistant Nodes', marker='o')
plt.title('Norm of Error vs. Number of Interpolation Points for f2(x)')
plt.xlabel('Number of Interpolation Points (n)')
plt.ylabel('Norm of Error')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
