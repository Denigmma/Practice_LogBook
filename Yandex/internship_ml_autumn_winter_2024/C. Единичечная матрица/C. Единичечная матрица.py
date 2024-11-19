# n=int(input())
# first_row = list(map(float, input().strip().split()))

# import numpy as np
# def restore_matrix_and_sum(n, first_row):
#     # Нормализация первой строки (если не нормализована)
#     first_row = np.array(first_row)
#     norm = np.sqrt(np.sum(first_row ** 2))
#     if abs(norm - 1) > 1e-6:
#         print(f"Normalizing the first row: original norm = {norm}")
#         first_row = first_row / norm
#
#     # Восстанавливаем матрицу ранга 1
#     matrix = np.outer(first_row, first_row)
#
#     # Проверяем условия:
#     trace = np.trace(matrix)
#     sum_of_squares = np.sum(matrix ** 2)
#     print(f"Trace: {trace}, Sum of squares: {sum_of_squares}")
#
#     assert abs(trace - 1) < 1e-6, "Trace condition failed"
#     assert abs(sum_of_squares - 1) < 1e-6, "Sum of squares condition failed"
#
#     # Вычисляем сумму всех элементов
#     total_sum = np.sum(matrix)
#     return round(total_sum, 3)
#
#
# # Ввод
# n = int(input("Enter n: "))  # Размер матрицы
# first_row = list(map(float, input("Enter first row: ").split()))  # Первая строка матрицы
#
# # Вывод результата
# result = restore_matrix_and_sum(n, first_row)
# print(result)

import math
def restore_matrix_and_sum(n, first_row):
    norm = math.sqrt(sum(x ** 2 for x in first_row))
    if abs(norm - 1) > 1e-6:
        first_row = [x / norm for x in first_row]

    matrix = [[first_row[i] * first_row[j] for j in range(n)] for i in range(n)]

    trace = sum(matrix[i][i] for i in range(n))
    sum_of_squares = sum(sum(matrix[i][j] ** 2 for j in range(n)) for i in range(n))

    assert abs(trace - 1) < 1e-6, "Trace condition failed"
    assert abs(sum_of_squares - 1) < 1e-6, "Sum of squares condition failed"

    total_sum = sum(sum(row) for row in matrix)
    return round(total_sum, 3)


n = int(input())
first_row = list(map(float, input().split()))

result = restore_matrix_and_sum(n, first_row)
print(f"{result:.3f}")

