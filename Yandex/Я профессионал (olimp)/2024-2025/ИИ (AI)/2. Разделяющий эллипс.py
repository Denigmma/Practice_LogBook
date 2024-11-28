### answer = 3

import numpy as np

mu1 = np.array([-2, 3])
mu2 = np.array([1, 0])
Sigma1 = np.array([[2, 0], [0, 1]])
Sigma2 = np.array([[3, -1], [-1, 4]])

# Разница обратных ковариаций
inv_Sigma1 = np.linalg.inv(Sigma1)
inv_Sigma2 = np.linalg.inv(Sigma2)
A = inv_Sigma1 - inv_Sigma2

# Линейный член
b = 2 * (np.dot(inv_Sigma2, mu2) - np.dot(inv_Sigma1, mu1))

# Константный член (можно игнорировать для формы эллипса)
c = (
    np.dot(mu1.T, np.dot(inv_Sigma1, mu1))
    - np.dot(mu2.T, np.dot(inv_Sigma2, mu2))
    + np.log(np.linalg.det(Sigma2) / np.linalg.det(Sigma1))
)

# Собственные значения матрицы A
eigvals = np.linalg.eigvalsh(A)
print(eigvals)

# Длина большой полуоси (соответствует меньшему собственному значению)
semi_major_axis_length = 1 / np.sqrt(np.min(np.abs(eigvals)))  # Игнорируем отрицательные знаки

# Округляем до ближайшего целого
# rounded_length = round(semi_major_axis_length)
print(semi_major_axis_length)

# Матрица квадратичных членов
A = np.array([
    [3/22, -2/11 / 2],
    [-2/11 / 2, 8/11]
])

print(np.linalg.eigvals(A))




import matplotlib.pyplot as plt
def f(x1, x2):
    return (
        (3 / 22) * x1**2 +
        (8 / 11) * x2**2 -
        (2 / 11) * x1 * x2 +
        (30 / 11) * x1 -
        (64 / 11) * x2 +
        (117 / 11) +
        np.log(2 / 11)
    )

# Создаём сетку значений для x1 и x2
x1 = np.linspace(-100, 10, 500)  # Область для x1
x2 = np.linspace(-10, 10, 500)  # Область для x2
X1, X2 = np.meshgrid(x1, x2)    # Создаём сетку

Z = f(X1, X2)

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=[0], colors='red')  # Уровень f(x1, x2) = 0
plt.title("График уравнения")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.show()


