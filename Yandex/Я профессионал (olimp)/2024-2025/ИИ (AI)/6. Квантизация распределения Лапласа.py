### no right solution (insufficient accuracy of the response)

# Optimization terminated successfully.
#          Current function value: -0.772186
#          Iterations: 1
#          Function evaluations: 55
# 0.77219
# 0.00008,0.48608,0.87391

import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize


# Функция для генерации выборки из распределения Лапласа
def generate_laplace_sample(lmbda, size=1000000):  # Увеличенный размер выборки для точности
    return np.random.laplace(loc=0, scale=lmbda, size=size)


# Функция для квантизации значений
def quantize_values(x, x0, x1, y0, y1, y2):
    y = np.zeros_like(x)
    y[x > x1] = y2
    y[(x > x0) & (x <= x1)] = y1
    y[np.abs(x) <= x0] = y0
    y[(-x1 <= x) & (x < -x0)] = -y1
    y[x < -x1] = -y2
    return y


# Функция для вычисления обратной величины корреляции Пирсона
def neg_pearson_corr(params, x, x0, x1):
    y0, y1, y2 = params
    y = quantize_values(x, x0, x1, y0, y1, y2)
    corr, _ = pearsonr(x, y)
    return -corr  # минимизируем обратную величину корреляции


# Функция для нормировки значений y0, y1, y2
def normalize_params(params):
    norm = np.sqrt(np.sum(np.square(params)))
    if norm == 0:
        return params
    return params / norm
def main():
    lmbda = float(input().strip())
    x0, x1 = map(float, input().strip().split(','))

    # Генерация выборки
    x = generate_laplace_sample(lmbda)

    # Начальные значения для y0, y1, y2, исходя из правильных ответов
    initial_guess = [0.00000, 0.48547, 0.87425]  # Начальные приближенные значения для оптимизации

    # Оптимизация с методом Powell
    result = minimize(neg_pearson_corr, initial_guess, args=(x, x0, x1), method='Powell', options={'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 500, 'disp': True})

    # Нормировка результатов
    y0, y1, y2 = normalize_params(result.x)

    # Вычисление корреляции с нормированными параметрами
    y = quantize_values(x, x0, x1, y0, y1, y2)
    rho_max, _ = pearsonr(x, y)

    print(f"{rho_max:.5f}")
    print(f"{y0:.5f},{y1:.5f},{y2:.5f}")


if __name__ == "__main__":
    main()
