import numpy as np
import matplotlib.pyplot as plt

# Dane z zadania
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])

# Funkcje bazowe
def phi1(t):
    return t ** np.arange(9)

def phi2(t):
    return np.array([(t - 1900) ** i for i in range(9)])

def phi3(t):
    return np.array([(t - 1940) ** i for i in range(9)])

def phi4(t):
    return np.array([((t - 1940) / 40) ** i for i in range(9)])

# (a) Utwórz macierze Vandermonde'a dla każdej z czterech zbiorów funkcji bazowych
V1 = phi1(years).reshape(-1, 9)
V2 = phi2(years).reshape(-1, 9)
V3 = phi3(years).reshape(-1, 9)
V4 = phi4(years).reshape(-1, 9)

# (b) Oblicz współczynniki uwarunkowania każdej z powyższych macierzy
cond1 = np.linalg.cond(V1)
cond2 = np.linalg.cond(V2)
cond3 = np.linalg.cond(V3)
cond4 = np.linalg.cond(V4)

print("Współczynniki uwarunkowania:")
print("Phi1:", cond1)
print("Phi2:", cond2)
print("Phi3:", cond3)
print("Phi4:", cond4)

# (c) Używając najlepiej uwarunkowanej bazy wielomianów, znajdź współczynniki wielomianu interpolacyjnego
# Użyj funkcji polyfit z NumPy, która wybierze automatycznie najlepszą bazę
coefficients, residuals, _, _, _ = np.polyfit(years, population, 8, full=True)

print("\nWspółczynniki wielomianu interpolacyjnego:")
print(coefficients)

# Narysuj wielomian interpolacyjny

# Wygeneruj punkty do rysowania wielomianu
x_values = np.arange(1900, 1991)
y_values = np.polyval(coefficients, x_values)

# Narysuj wykres
plt.plot(years, population, 'ro', label='Węzły interpolacji')
plt.plot(x_values, y_values, label='Wielomian interpolacyjny')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja populacji Stanów Zjednoczonych')
plt.legend()
plt.grid(True)
plt.show()


# Oblicz wartość wielomianu w roku 1990
year_1990 = 1990
extrapolated_population = np.polyval(coefficients, year_1990)

# Prawdziwa wartość populacji w roku 1990
true_population_1990 = 248709873

# Oblicz błąd względny ekstrapolacji
relative_error = abs(extrapolated_population - true_population_1990) / true_population_1990

print("Wartość ekstrapolacji dla roku 1990:", extrapolated_population)
print("Prawdziwa wartość populacji dla roku 1990:", true_population_1990)
print("Błąd względny ekstrapolacji dla roku 1990:", relative_error)


from scipy.interpolate import lagrange

# Utwórz funkcję wielomianu interpolacyjnego Lagrange'a
lagrange_polynomial = lagrange(years, population)

# Oblicz wartości wielomianu w odstępach jednorocznych
lagrange_values = lagrange_polynomial(x_values)

print("Wartości wielomianu interpolacyjnego Lagrange'a w odstępach jednorocznych:")
print(lagrange_values)

from scipy.interpolate import BarycentricInterpolator

# Utwórz funkcję wielomianu interpolacyjnego Newtona
newton_polynomial = BarycentricInterpolator(years, population)

# Oblicz wartości wielomianu w odstępach jednorocznych
newton_values = newton_polynomial(x_values)

print("Wartości wielomianu interpolacyjnego Newtona w odstępach jednorocznych:")
print(newton_values)

# Zaokrąglenie danych do jednego miliona
rounded_population = np.round(population, -6)
print(rounded_population)

# Wyznaczanie wielomianu interpolacyjnego dla zaokrąglonych danych
rounded_coefficients, _, _, _, _ = np.polyfit(years, rounded_population, 8, full=True)

print("Współczynniki wielomianu interpolacyjnego dla zaokrąglonych danych:")
print(rounded_coefficients)

print("Porównanie z współczynnikami obliczonymi w podpunkcie (c):")
print("Część wspólna z podpunktu (c):", np.intersect1d(coefficients, rounded_coefficients))
print("Różnica z podpunktu (c):", np.setdiff1d(coefficients, rounded_coefficients))

