import sys
import numpy as np
from scipy.optimize import linprog


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))

    # Читаем матрицу первого слоя (n x n)
    A = []
    for _ in range(n):
        row = [float(next(it)) for _ in range(n)]
        A.append(row)
    A = np.array(A, dtype=float)

    # Читаем матрицу второго слоя (m x n)
    B = []
    for _ in range(m):
        row = [float(next(it)) for _ in range(n)]
        B.append(row)
    B = np.array(B, dtype=float)

    tol = 1e-9  # порог для нулевых значений

    # Перебираем все пары классов j, k (j < k)
    for j in range(m):
        for k in range(j + 1, m):
            # Определяем d = B[j] - B[k]
            d = B[j] - B[k]
            # ЛП: ищем y >= 0, d^T y = 0, и для всех l: (B[j]-B[l])^T y >= 0.
            # Чтобы использовать linprog, запишем неравенство в виде: - (B[j]-B[l])^T y <= 0.
            A_eq = np.array([d])
            b_eq = np.array([0.0])
            A_ub = []
            b_ub = []
            for l in range(m):
                A_ub.append(- (B[j] - B[l]))
                b_ub.append(0.0)
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            # Целевая функция: максимизировать сумму координат y, что эквивалентно минимизации -sum(y)
            c = -np.ones(n)
            res_y = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                            bounds=[(0, None)] * n, method='highs')
            # Если найдено ненулевое y (сумма > tol)
            if res_y.success and (-res_y.fun) > tol:
                y = res_y.x
                # Разобьём индексы: S - те, где y[i] > tol, T - остальные
                S = [i for i in range(n) if y[i] > tol]
                T = [i for i in range(n) if y[i] <= tol]
                if len(S) == 0:
                    continue  # y = 0, пропускаем
                # Теперь ищем x, чтобы ReLU(A*x) = y:
                # Для i in S требуем: (A[i] dot x) = y[i]
                # Для i in T требуем: (A[i] dot x) <= 0.
                A_eq_x = A[S, :] if S else None
                b_eq_x = y[S] if S else None
                A_ub_x = A[T, :] if T else None
                b_ub_x = np.zeros(len(T)) if T else None

                # Решаем ЛП для x (целевой функции можно задать нулевую)
                res_x = linprog(np.zeros(n), A_ub=A_ub_x, b_ub=b_ub_x,
                                A_eq=A_eq_x, b_eq=b_eq_x, bounds=[(None, None)] * n,
                                method='highs')
                if res_x.success:
                    x = res_x.x
                    # Проверяем, что x не тривиально нулевой (хотя если y не нулевой, x уже не должно быть нулевым)
                    if np.all(np.abs(x) < 1e-8):
                        continue
                    # Вывод результата
                    sys.stdout.write("YES\n")
                    sys.stdout.write(" ".join("{:.10f}".format(xi) for xi in x) + "\n")
                    return
    # Если ни для одной пары не найдено подходящего x
    sys.stdout.write("NO\n")


if __name__ == '__main__':
    main()
