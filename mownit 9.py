import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


h = 0.5
y = 1  # warunek początkowy

# Metoda Eulera
y = y + h * (-5 * y)
print("Metoda Eulera", y)  # Wartość y przy t = 0.5


h = 0.5
y = 1  # warunek początkowy

# Niejawna metoda Eulera
y = y / (1 + 5 * h)
print("Niejawna metoda Eulera", y)  # Wartość y przy t = 0.5



# Parametry
beta = 1
gamma = 1 / 7
h = 0.01  # zmniejszony krok czasowy
t_max = 14
t = np.arange(0, t_max + h, h)

# Wartości początkowe
S = np.zeros(len(t))
I = np.zeros(len(t))
R = np.zeros(len(t))
S[0] = 762
I[0] = 1
R[0] = 0

# Metoda Eulera z ograniczeniami
for k in range(len(t) - 1):
    S[k + 1] = S[k] - h * beta * I[k] * S[k]
    I[k + 1] = I[k] + h * (beta * I[k] * S[k] - gamma * I[k])
    R[k + 1] = R[k] + h * gamma * I[k]

    # Ograniczenia dla wartości
    S[k + 1] = max(S[k + 1], 0)
    I[k + 1] = max(I[k + 1], 0)
    R[k + 1] = max(R[k + 1], 0)

# Wykres
plt.plot(t, S, label='S(t)')
plt.plot(t, I, label='I(t)')
plt.plot(t, R, label='R(t)')
plt.xlabel('t')
plt.ylabel('Liczba osób')
plt.legend()
plt.show()

# Prawdziwe dane zakażonych
I_true = np.array([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4])

# Funkcja kosztu
def cost_function(theta):
    beta, gamma = theta
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = 762
    I[0] = 1
    R[0] = 0
    
    for k in range(len(t) - 1):
        S[k + 1] = S[k] - h * beta * I[k] * S[k]
        I[k + 1] = I[k] + h * (beta * I[k] * S[k] - gamma * I[k])
        R[k + 1] = R[k] + h * gamma * I[k]
        
        # Ograniczenia dla wartości
        S[k + 1] = max(S[k + 1], 0)
        I[k + 1] = max(I[k + 1], 0)
        R[k + 1] = max(R[k + 1], 0)
    
    return np.sum((I[:len(I_true)] - I_true) ** 2)

# Początkowe zgadywanie wartości beta i gamma
initial_guess = [1, 1/7]

# Minimalizacja funkcji kosztu
result = minimize(cost_function, initial_guess, method='Nelder-Mead')
beta_est, gamma_est = result.x
R0 = beta_est / gamma_est

print(f"Estymowane beta: {beta_est}, gamma: {gamma_est}, R0: {R0}")


