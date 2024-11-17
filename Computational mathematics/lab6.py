import numpy as np

# Исходная матрица
A = np.array([[4, 1, 2],
              [1, 6, 1],
              [2, 1, 9]], dtype=float)


# Функция для нахождения индексов максимального недиагонального элемента
def max_offdiag_element(A):
    n = A.shape[0]
    max_val = 0
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return p, q


# Функция для проведения вращения Якоби
def jacobi_rotation(A, tol=1e-10):
    n = A.shape[0]
    V = np.eye(n)  # Матрица для собственных векторов
    while True:
        p, q = max_offdiag_element(A)
        if abs(A[p, q]) < tol:
            break

        # Вычисляем угол вращения
        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Обновление матрицы A
        App = A[p, p] * cos_theta ** 2 + A[q, q] * sin_theta ** 2 + 2 * A[p, q] * cos_theta * sin_theta
        Aqq = A[q, q] * cos_theta ** 2 + A[p, p] * sin_theta ** 2 - 2 * A[p, q] * cos_theta * sin_theta
        Apq = 0  # Недиагональный элемент после вращения

        # Обновление остальных элементов
        for i in range(n):
            if i != p and i != q:
                Aip = A[i, p]
                Aiq = A[i, q]
                A[i, p] = Aip * cos_theta + Aiq * sin_theta
                A[i, q] = Aiq * cos_theta - Aip * sin_theta
                A[p, i] = A[i, p]
                A[q, i] = A[i, q]

        A[p, p] = App
        A[q, q] = Aqq
        A[p, q] = A[q, p] = Apq

    return np.diag(A), A

# Решение
eigenvalues,A = jacobi_rotation(A)
print("Собственные числа матрицы:", eigenvalues)
print("Матрица после преобразований:\n", A)
