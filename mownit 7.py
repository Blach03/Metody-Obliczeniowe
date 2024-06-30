import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec


def f1(x):
    return 4.0 / (1 + x**2)
exact_value1 = np.pi


def f2(x):
    return np.where(x > 0, np.sqrt(x) * np.log(x), 0)
exact_value2 = -4/9


a = 0.001
b = 0.004

def f3(x):
    term1 = 1 / ((x - 0.3)**2 + a)
    term2 = 1 / ((x - 0.9)**2 + b)
    return term1 + term2 - 6

def exact_integral(x0, a):
    return 1 / np.sqrt(a) * (np.arctan((1 - x0) / np.sqrt(a)) + np.arctan(x0 / np.sqrt(a)))

exact_value_1 = exact_integral(0.3, a)
exact_value_2 = exact_integral(0.9, b)

exact_value3 = exact_value_1 + exact_value_2 - 6



errors_trapezoidal = []
errors_gauss_kronrod = []
evaluations_trapezoidal = []
evaluations_gauss_kronrod = []
result_trapezoidal = []
result_gauss_kronrod = []


tolerances = np.logspace(0, -14, 25)
for tolerance in tolerances:
    trapez_result, trapez_error, trapez_info = quad_vec(f1, 0, 1, epsabs=tolerance, full_output=True, quadrature='trapezoid')
    errors_trapezoidal.append(abs(abs(trapez_result - exact_value1) / exact_value1))
    evaluations_trapezoidal.append(trapez_info.neval)
    result_trapezoidal.append(trapez_result)

    gk_result, gk_error, gk_info = quad_vec(f1, 0, 1, epsabs=tolerance, full_output=True, quadrature='gk15')
    errors_gauss_kronrod.append(abs(abs(gk_result - exact_value1) / exact_value1))
    evaluations_gauss_kronrod.append(gk_info.neval)
    result_gauss_kronrod.append(gk_result)

print("Errors gauss-kronrod", errors_gauss_kronrod)
print("Errors trapezoidal", errors_trapezoidal)
print("Evaluations gauss-kronrod", evaluations_gauss_kronrod)
print("Evaluations trapezoidal", evaluations_trapezoidal)
print("Result gauss-kronrod", result_gauss_kronrod)
print("Result trapezoidal", result_trapezoidal)



plt.figure(figsize=(12, 6))
plt.plot(evaluations_trapezoidal, errors_trapezoidal, '--', label='Metoda trapezów')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, label='Metoda Gaussa-Kronroda')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.xscale('log')
plt.yscale('log')
plt.title('Błąd względny w zależności od liczby ewaluacji dla różnych metod całkowania dla f1')
plt.legend()
plt.grid(True)
plt.show()

print('1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')

errors_trapezoidal = []
errors_gauss_kronrod = []
evaluations_trapezoidal = []
evaluations_gauss_kronrod = []
result_trapezoidal = []
result_gauss_kronrod = []


tolerances = np.logspace(0, -14, 25)
for tolerance in tolerances:
    trapez_result, trapez_error, trapez_info = quad_vec(f2, 0, 1, epsabs=tolerance, full_output=True, quadrature='trapezoid')
    errors_trapezoidal.append(abs(abs(trapez_result - exact_value2) / exact_value2))
    evaluations_trapezoidal.append(trapez_info.neval)
    result_trapezoidal.append(trapez_result)

    gk_result, gk_error, gk_info = quad_vec(f2, 0, 1, epsabs=tolerance, full_output=True, quadrature='gk15')
    errors_gauss_kronrod.append(abs(abs(gk_result - exact_value2) / exact_value2))
    evaluations_gauss_kronrod.append(gk_info.neval)
    result_gauss_kronrod.append(gk_result)

print("Errors gauss-kronrod", errors_gauss_kronrod)
print("Errors trapezoidal", errors_trapezoidal)
print("Evaluations gauss-kronrod", evaluations_gauss_kronrod)
print("Evaluations trapezoidal", evaluations_trapezoidal)
print("Result gauss-kronrod", result_gauss_kronrod)
print("Result trapezoidal", result_trapezoidal)



plt.figure(figsize=(12, 6))
plt.plot(evaluations_trapezoidal, errors_trapezoidal, '--', label='Metoda trapezów')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, label='Metoda Gaussa-Kronroda')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.xscale('log')
plt.yscale('log')
plt.title('Błąd względny w zależności od liczby ewaluacji dla różnych metod całkowania dla f2')
plt.legend()
plt.grid(True)
plt.show()

print('22222222222222222222222222222222222222222222222222222222222222222222222222222222222222')


errors_trapezoidal = []
errors_gauss_kronrod = []
evaluations_trapezoidal = []
evaluations_gauss_kronrod = []
result_trapezoidal = []
result_gauss_kronrod = []


tolerances = np.logspace(0, -14, 25)
for tolerance in tolerances:
    trapez_result, trapez_error, trapez_info = quad_vec(f3, 0, 1, epsabs=tolerance, full_output=True, quadrature='trapezoid')
    errors_trapezoidal.append(abs(abs(trapez_result - exact_value3) / exact_value3))
    evaluations_trapezoidal.append(trapez_info.neval)
    result_trapezoidal.append(trapez_result)

    gk_result, gk_error, gk_info = quad_vec(f3, 0, 1, epsabs=tolerance, full_output=True, quadrature='gk15')
    errors_gauss_kronrod.append(abs(abs(gk_result - exact_value3) / exact_value3))
    evaluations_gauss_kronrod.append(gk_info.neval)
    result_gauss_kronrod.append(gk_result)

print("Errors gauss-kronrod", errors_gauss_kronrod)
print("Errors trapezoidal", errors_trapezoidal)
print("Evaluations gauss-kronrod", evaluations_gauss_kronrod)
print("Evaluations trapezoidal", evaluations_trapezoidal)
print("Result gauss-kronrod", result_gauss_kronrod)
print("Result trapezoidal", result_trapezoidal)



plt.figure(figsize=(12, 6))
plt.plot(evaluations_trapezoidal, errors_trapezoidal, '--', label='Metoda trapezów')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, label='Metoda Gaussa-Kronroda')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.xscale('log')
plt.yscale('log')
plt.title('Błąd względny w zależności od liczby ewaluacji dla różnych metod całkowania dla f3')
plt.legend()
plt.grid(True)
plt.show()

print('3333333333333333333333333333333333333333333333333333333333333333333333333333')

from scipy import integrate

pi_true = np.pi
M_values = range(1, 26)
errors_midpoint = []
errors_trapez = []
errors_simpson = []
error_gauss_legendre = []

for m in M_values:
    n = 2**m
    x = np.linspace(0, 1, n+1)
    y = f1(x)
    
    x_mid = (x[:-1] + x[1:]) / 2  
    midpoint_approx = (x[1] - x[0]) * np.sum(f1(x_mid))
    errors_midpoint.append(abs(abs(midpoint_approx - exact_value1)/exact_value1))

    trapezoidal_approx = integrate.trapezoid(y=y, x=x)
    errors_trapez.append(abs(abs(trapezoidal_approx - exact_value1)/exact_value1))
    
    simpson_approx = integrate.simpson(y=y, x=x)
    errors_simpson.append(abs(abs(simpson_approx - exact_value1)/exact_value1))



plt.figure(figsize=(10, 6))
plt.plot(2**np.array(M_values), errors_midpoint, '--', label='Midpoint')
plt.plot(2**np.array(M_values), errors_trapez, '-', label='Trapezoidal')
plt.plot(2**np.array(M_values), errors_simpson , '--', label='Simpson')


plt.plot(evaluations_trapezoidal, errors_trapezoidal, 'o-', label='Adaptacyjne trapezy')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, 'o-', label='Gauss-Kronrod')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.title('Błąd względny metod całkowania numerycznego dla f1')
plt.legend()

plt.show()



errors_midpoint = []
errors_trapez = []
errors_simpson = []
error_gauss_legendre = []

for m in M_values:
    n = 2**m
    x = np.linspace(0, 1, n+1)
    y = f2(x)
    
    x_mid = (x[:-1] + x[1:]) / 2  
    midpoint_approx = (x[1] - x[0]) * np.sum(f2(x_mid))
    errors_midpoint.append(abs(abs(midpoint_approx - exact_value2)/exact_value2))

    trapezoidal_approx = integrate.trapezoid(y=y, x=x)
    errors_trapez.append(abs(abs(trapezoidal_approx - exact_value2)/exact_value2))
    
    simpson_approx = integrate.simpson(y=y, x=x)
    errors_simpson.append(abs(abs(simpson_approx - exact_value2)/exact_value2))



plt.figure(figsize=(10, 6))
plt.plot(2**np.array(M_values), errors_midpoint, '--', label='Midpoint')
plt.plot(2**np.array(M_values), errors_trapez, '-', label='Trapezoidal')
plt.plot(2**np.array(M_values), errors_simpson , '--', label='Simpson')


plt.plot(evaluations_trapezoidal, errors_trapezoidal, 'o-', label='Adaptacyjne trapezy')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, 'o-', label='Gauss-Kronrod')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.title('Błąd względny metod całkowania numerycznego dla f2')
plt.legend()

plt.show()


errors_midpoint = []
errors_trapez = []
errors_simpson = []
error_gauss_legendre = []

for m in M_values:
    n = 2**m
    x = np.linspace(0, 1, n+1)
    y = f3(x)
    
    x_mid = (x[:-1] + x[1:]) / 2  
    midpoint_approx = (x[1] - x[0]) * np.sum(f3(x_mid))
    errors_midpoint.append(abs(abs(midpoint_approx - exact_value3)/exact_value3))

    trapezoidal_approx = integrate.trapezoid(y=y, x=x)
    errors_trapez.append(abs(abs(trapezoidal_approx - exact_value3)/exact_value3))
    
    simpson_approx = integrate.simpson(y=y, x=x)
    errors_simpson.append(abs(abs(simpson_approx - exact_value3)/exact_value3))



plt.figure(figsize=(10, 6))
plt.plot(2**np.array(M_values), errors_midpoint, '--', label='Midpoint')
plt.plot(2**np.array(M_values), errors_trapez, '-', label='Trapezoidal')
plt.plot(2**np.array(M_values), errors_simpson , '--', label='Simpson')


plt.plot(evaluations_trapezoidal, errors_trapezoidal, 'o-', label='Adaptacyjne trapezy')
plt.plot(evaluations_gauss_kronrod, errors_gauss_kronrod, 'o-', label='Gauss-Kronrod')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Liczba ewaluacji funkcji podcałkowej')
plt.ylabel('Wartość bezwzględna błędu względnego')
plt.title('Błąd względny metod całkowania numerycznego dla f3')
plt.legend()

plt.show()


