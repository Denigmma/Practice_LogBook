import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Параметры
L = 1.0  # Длина области
T = 0.08  # Время
Nx = 10  # Число пространственных шагов
Nt = 1000  # Число временных шагов
dx = L / (Nx - 1)  # Шаг по пространству
dt = T / Nt  # Шаг по времени

# Создаем сетку
x = np.linspace(0, L, Nx)
u_explicit = np.zeros((Nx, Nt + 1))

# Начальные условия
u_explicit[:, 0] = 0.5 * x ** 2

# Явная схема
for n in range(0, Nt):
    for i in range(1, Nx - 1):
        u_explicit[i, n + 1] = u_explicit[i, n] + (dt / dx ** 2) * (
                    u_explicit[i + 1, n] - 2 * u_explicit[i, n] + u_explicit[i - 1, n])
    # Граничные условия
    u_explicit[0, n + 1] = n * dt
    u_explicit[-1, n + 1] = n * dt + 0.5

# Неявная схема
u_implicit = np.zeros((Nx, Nt + 1))
u_implicit[:, 0] = 0.5 * x ** 2

# Коэффициенты для трёхдиагональной матрицы
A = np.zeros((3, Nx - 2))
for i in range(1, Nx - 1):
    A[0, i - 1] = -dt / dx ** 2
    A[1, i - 1] = 1 + 2 * dt / dx ** 2
    A[2, i - 1] = -dt / dx ** 2

# Граничные условия для неявной схемы
for n in range(0, Nt):
    b = u_implicit[1:Nx - 1, n]
    b[0] += dt / dx ** 2 * u_implicit[0, n]
    b[-1] += dt / dx ** 2 * u_implicit[-1, n]

    # Решение системы уравнений
    u_implicit[1:Nx - 1, n + 1] = solve_banded((1, 1), A, b)
    u_implicit[0, n + 1] = n * dt
    u_implicit[-1, n + 1] = n * dt + 0.5

# Визуализация результатов
plt.figure(figsize=(12, 5))

# График для явной схемы
plt.subplot(1, 2, 1)
for n in range(0, Nt + 1, Nt // 5):
    plt.plot(x, u_explicit[:, n], label=f"t={n * dt:.2f}")
plt.title('Явная схема')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()

# График для неявной схемы
plt.subplot(1, 2, 2)
for n in range(0, Nt + 1, Nt // 5):
    plt.plot(x, u_implicit[:, n], label=f"t={n * dt:.2f}")
plt.title('Неявная схема')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()

plt.tight_layout()
plt.show()
