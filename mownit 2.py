import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
labels=pd.io.parsers.read_csv("breast-cancer.labels")
data1=pd.io.parsers.read_csv("breast-cancer-train.dat", usecols={1,2,4,5,10})
data2=pd.io.parsers.read_csv("breast-cancer-validate.dat", usecols={1,2,4,5,10})

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
print(df1)
print(df2)

sorted_column1 = np.sort(df1.iloc[:, 1])

plt.figure(figsize=(10, 6))
plt.hist(df1.iloc[:, 1], bins=20, color='blue')
plt.title('Histogram of Column1')
plt.xlabel('Column1 Values')
plt.ylabel('Frequency')
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(sorted_column1)
plt.title('Line Plot of Column1')
plt.xlabel('Index')
plt.ylabel('Column1 Values')
plt.grid(True)
plt.show()

df1.iloc[:, 0] = np.where(df1.iloc[:, 0] == 'M', 1, -1)
df2.iloc[:, 0] = np.where(df2.iloc[:, 0] == 'M', 1, -1)

b1 = df1.iloc[:, 0].to_numpy()
b2 = df2.iloc[:, 0].to_numpy()


A1_linear = np.vstack([df1.iloc[:, 1:].values.T]).T
A1_quadratic = np.vstack([df1.iloc[:, 1:].values.T, (df1.iloc[:, 1:].values**2).T]).T

A2_linear = np.vstack([df2.iloc[:, 1:].values.T]).T
A2_quadratic = np.vstack([df2.iloc[:, 1:].values.T, (df2.iloc[:, 1:].values**2).T]).T

print(A1_linear)
print(A1_quadratic)
ATA = A1_linear.T @ A1_linear
ATb1 = A1_linear.T @ b1

ATA2 = A2_linear.T @ A2_linear
ATb2 = A2_linear.T @ b2

BTB = A1_quadratic.T @ A1_quadratic
BTb1 = A1_quadratic.T @ b1

BTB2 = A2_quadratic.T @ A2_quadratic
BTb2 = A2_quadratic.T @ b2


tab1 = [[] for i in range(4)]
tab2 = []
tab3 = [[] for i in range(4)]
tab4 = []

tab5 = [[] for i in range(8)]
tab6 = []
tab7 = [[] for i in range(8)]
tab8 = []

for i in range(4):
    for j in range(4):
        tab1[i].append(ATA[i][j])

for i in range(4):
    tab2.append(ATb1[i])

for i in range(4):
    for j in range(4):
        tab3[i].append(ATA2[i][j])

for i in range(4):
    tab4.append(ATb2[i])

for i in range(8):
    for j in range(8):
        tab5[i].append(BTB[i][j])

for i in range(8):
    tab6.append(BTb1[i])

for i in range(8):
    for j in range(8):
        tab7[i].append(BTB2[i][j])

for i in range(8):
    tab8.append(BTb2[i])


weights_linear = solve(tab1, tab2)

weights_quadratic = solve(tab5, tab6)

print("Linear Model Weights:", weights_linear)
print("Quadratic Model Weights:", weights_quadratic)

weights_linear_validate = solve(tab3,tab4)

weights_quadratic_validate = solve(tab7,tab8)

print("Linear Model Weights for Validation:", weights_linear_validate)
print("Quadratic Model Weights for Validation:", weights_quadratic_validate)

cond_A1_linear = np.linalg.cond(np.dot(A1_linear.T, A1_linear))
cond_A1_quadratic = np.linalg.cond(np.dot(A1_quadratic.T, A1_quadratic))

cond_A2_linear = np.linalg.cond(np.dot(A2_linear.T, A2_linear))
cond_A2_quadratic = np.linalg.cond(np.dot(A2_quadratic.T, A2_quadratic))

print("Współczynniki uwarunkowania dla df1:")
print("Liniowa metoda najmniejszych kwadratów:", cond_A1_linear)
print("Kwadratowa metoda najmniejszych kwadratów:", cond_A1_quadratic)

print("\nWspółczynniki uwarunkowania dla df2:")
print("Liniowa metoda najmniejszych kwadratów:", cond_A2_linear)
print("Kwadratowa metoda najmniejszych kwadratów:", cond_A2_quadratic)

predictions_linear = np.dot(A2_linear, weights_linear)

predictions_quadratic = np.dot(A2_quadratic, weights_quadratic)

predictions_linear_labels = np.where(predictions_linear > 0, 1, -1)
predictions_quadratic_labels = np.where(predictions_quadratic > 0, 1, -1)

false_positives_linear = np.sum((predictions_linear_labels > 0) & (b2 == -1))
false_negatives_linear = np.sum((predictions_linear_labels <= 0) & (b2 == 1))

false_positives_quadratic = np.sum((predictions_quadratic_labels > 0) & (b2 == -1))
false_negatives_quadratic = np.sum((predictions_quadratic_labels <= 0) & (b2 == 1))

print("Reprezentacja liniowa:")
print("Liczba fałszywie dodatnich przypadków:", false_positives_linear)
print("Liczba fałszywie ujemnych przypadków:", false_negatives_linear)

print("\nReprezentacja kwadratowa:")
print("Liczba fałszywie dodatnich przypadków:", false_positives_quadratic)
print("Liczba fałszywie ujemnych przypadków:", false_negatives_quadratic)


