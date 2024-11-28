### right solution

import numpy as np
from sympy import Matrix, symbols, solve, SparseMatrix


def solve_convolution_exact(m, n, image, kernel_size, result):
    h, w = kernel_size
    pad_h = (h - 1) // 2
    pad_w = (w - 1) // 2

    padded_image = np.zeros((m + 2 * pad_h, n + 2 * pad_w), dtype=int)
    padded_image[pad_h:pad_h + m, pad_w:pad_w + n] = image

    # Формируем систему уравнений
    A = []  # Матрица коэффициентов
    b = []  # Результаты свёртки

    for i in range(m):
        for j in range(n):
            submatrix = padded_image[i:i + h, j:j + w].flatten()
            A.append(submatrix)
            b.append(result[i, j])

    # Преобразуем в матрицы sympy
    A = SparseMatrix(np.array(A))
    b = Matrix(b)

    # Создаём переменные для ядра
    kernel_vars = symbols(f'k0:{h * w}')

    # Решаем систему линейных уравнений
    solution = solve(A * Matrix(kernel_vars) - b, kernel_vars)

    # Преобразуем результат в матрицу h x w
    kernel = np.array([solution[var] for var in kernel_vars]).reshape(h, w)

    return kernel.astype(int)

m, n = map(int, input().replace(',', ' ').split())
image = np.array([list(map(int, input().replace(',', ' ').split())) for _ in range(m)])
h, w = map(int, input().replace(',', ' ').split())
result = np.array([list(map(int, input().replace(',', ' ').split())) for _ in range(m)])

kernel = solve_convolution_exact(m, n, image, (h, w), result)

kernel = [row[::-1] for row in kernel][::-1]

for row in kernel:
    print(",".join(map(str, row)))

