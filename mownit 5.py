import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt


# Dane z lat
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])

degrees = np.arange(7)

year_1990 = 1990

predicted_populations = np.zeros((len(degrees), len(years)))
extrapolated_values = np.zeros(len(degrees))

for i, degree in enumerate(degrees):
    coeffs = np.polyfit(years, population, degree)
    predicted_populations[i] = np.polyval(coeffs, years)
    extrapolated_values[i] = np.polyval(coeffs, year_1990)

true_value_1990 = 248709873
relative_errors = np.abs(extrapolated_values - true_value_1990) / true_value_1990

for i, degree in enumerate(degrees):
    print(f"Stopień wielomianu m = {degree}:")
    print(f"Wartość ekstrapolowana do 1990: {extrapolated_values[i]:.2f}")
    print(f"Błąd względny: {relative_errors[i] * 100:.2f}%\n")


n = len(years)
AICc_values = np.zeros(len(degrees))

for i, degree in enumerate(degrees):
    k = degree + 1
    residual = population - predicted_populations[i]
    RSS = np.sum(residual ** 2)
    # Obliczenie AICc
    AICc_values[i] = 2 * k + n * np.log(RSS / n) + 2 * k * (k + 1) / (n - k - 1)

for i, degree in enumerate(degrees):
    print(f"Stopień wielomianu m = {degree}: AICc = {AICc_values[i]}")

best_degree_index = np.argmin(AICc_values)
best_degree = degrees[best_degree_index]

print(f"Najlepszy stopień wielomianu według AICc: m = {best_degree}")


def f(x):
    return np.sqrt(x)

a = 0
b = 2
n = 3

x = np.linspace(-1, 1, 100)
x_mapped = 2 * (x - a) / (b - a) 

cheb_nodes = np.polynomial.chebyshev.chebpts1(n)

y = f((b - a) * (cheb_nodes + 1) / 2 + a)

coeffs = chebfit(cheb_nodes, y, deg=n-1)

approximated_values = chebval(x_mapped, coeffs)

x_original = (x_mapped + 1) * (b - a) / 2 + a

plt.plot(x_original, approximated_values, label='Wielomian Czebyszewa')
plt.plot(x_original, f(x_original), label='f(x)')
plt.title('Aproksymacja funkcji √x wielomianem Czebyszewa')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
