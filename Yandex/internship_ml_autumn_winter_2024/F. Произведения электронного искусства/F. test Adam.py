import numpy as np
from scipy.optimize import minimize


# Функция для применения свертки с ядрами
def apply_convolution(I, kernels_order, K_unknown=None):
    temp = I.copy()
    for idx in kernels_order:
        if isinstance(idx, int):  # Применяем фиксированное ядро
            temp = convolve2d(temp, kernels[idx], mode='same')
        elif idx == 'unknown' and K_unknown is not None:  # Применяем неизвестное ядро
            temp = convolve2d(temp, K_unknown, mode='same')
    return temp


# Обновленная функция для вычисления потерь с учетом обучения неизвестного ядра
def calculate_mse_with_gradients(I, O, kernels_order, K_unknown=None):
    temp = apply_convolution(I, kernels_order, K_unknown)
    if temp.shape != O.shape:
        temp = temp[:O.shape[0], :O.shape[1]]  # Обрезаем, если размеры различаются
    mse = np.mean((temp - O) ** 2)
    return mse


# Функция для оптимизации ядра с использованием градиентного спуска
def optimize_kernel_with_gradients(kernels_order):
    def objective(K_unknown_flat):
        K_unknown = K_unknown_flat.reshape((3, 3))
        total_mse = 0
        for I, O in zip(inputs, outputs):
            total_mse += calculate_mse_with_gradients(I, O, kernels_order, K_unknown)
        return total_mse / len(inputs)

    # Инициализация случайным значением для неизвестного ядра
    K_unknown_init = np.random.randn(3, 3)

    # Минимизация функции потерь с градиентным спуском
    result = minimize(objective, K_unknown_init.flatten(), method='L-BFGS-B', jac=True)
    K_unknown_optimized = result.x.reshape((3, 3))

    # Вычисление улучшенного MSE
    best_mse = result.fun
    return best_mse, K_unknown_optimized


# Основной цикл оптимизации
best_mse = float('inf')
best_kernel = None
best_kernels_order = None
best_kernel_result = None

kernel_indices = list(range(len(kernels)))  # Изначальный порядок
for perm in permutations(kernel_indices):
    mse, K_unknown = optimize_kernel_with_gradients(list(perm) + ['unknown'])
    if mse < best_mse:
        best_mse = mse
        best_kernels_order = perm
        best_kernel_result = K_unknown

# Сохраняем лучший результат
ordered_kernels = []
for idx in best_kernels_order:
    ordered_kernels.append(kernels[idx])
ordered_kernels.append(best_kernel_result)

# Сохраняем ядра в файл
with open('restored_algos.csv', 'w') as f:
    for k in ordered_kernels:
        flat_k = k.flatten()
        line = ','.join(map(str, flat_k))
        f.write(line + '\n')

print(f"Best MSE: {best_mse}")
print(f"Best kernel: \n{best_kernel_result}")
