from itertools import permutations
import numpy as np
import pandas as pd
import glob
from PIL import Image
from scipy.signal import convolve2d
from numpy.linalg import lstsq
from scipy.optimize import minimize


# Загрузка ядер
def load_kernels():
    try:
        kernels_df = pd.read_csv('best_pictures/algos.csv', header=None)
        kernels = [np.array(kernels_df.iloc[i]).reshape((3, 3)) for i in range(len(kernels_df))]
        return kernels
    except FileNotFoundError:
        print("Error: algos.csv file not found.")
        exit()


# Загрузка изображений и выходных данных
def load_images_and_outputs():
    input_files = sorted(glob.glob('best_pictures/*.png'))
    output_files = sorted(glob.glob('best_pictures/*.txt'))

    inputs = []
    outputs = []

    for img_file, out_file in zip(input_files, output_files):
        try:
            img = Image.open(img_file).convert('L')
            img = np.array(img, dtype=np.float64)
            inputs.append(img)
        except FileNotFoundError:
            continue

        try:
            out = np.loadtxt(out_file)
            outputs.append(out)
        except FileNotFoundError:
            continue

    return inputs, outputs


# Применение известных ядер с весами
def apply_known_kernels(I, kernels_order, weights=None):
    if weights is None:
        weights = [1] * len(kernels_order)  # Присваиваем вес 1 для каждого ядра

    temp = I.copy()
    for idx, weight in zip(kernels_order, weights):
        if isinstance(idx, int):  # Если это целочисленный индекс, применяем ядро
            temp = convolve2d(temp, kernels[idx], mode='same')
        elif idx == 'unknown':  # Если это 'unknown', пропускаем это ядро
            continue
    return temp


# Функция для вычисления MSE
def calculate_mse(I, O, kernels_order, weights, K_unknown):
    temp = apply_known_kernels(I, kernels_order, weights)
    if K_unknown is not None:  # Если неизвестное ядро задано, применяем его
        temp = convolve2d(temp, K_unknown, mode='same')
    if temp.shape != O.shape:
        temp = temp[:O.shape[0], :O.shape[1]]  # Обрезаем, если размеры различаются
    mse = np.mean((temp - O) ** 2)
    return mse


# Функция для вычисления общей MSE для всех изображений
def total_mse(kernels_order, weights, K_unknown):
    total_mse = 0
    for I, O in zip(inputs, outputs):
        temp = apply_known_kernels(I, kernels_order, weights)
        print(f"Applying kernels with weights: {weights}")  # Для отладки
        if K_unknown is not None:
            temp = convolve2d(temp, K_unknown, mode='same')
        if temp.shape != O.shape:
            temp = temp[:O.shape[0], :O.shape[1]]
        mse = np.mean((temp - O) ** 2)
        print(f"Current MSE: {mse}")  # Для отладки
        total_mse += mse
    return total_mse / len(inputs)



# Оптимизация ядер и весов
def optimize_kernel_and_weights(kernels_order):
    A = []
    b = []
    for I, O in zip(inputs, outputs):
        temp = apply_known_kernels(I, kernels_order)

        # Подготовка для сбора уравнений
        h, w = O.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                patch = temp[i - 1:i + 2, j - 1:j + 2]
                if patch.shape == (3, 3):
                    A.append(patch.flatten())
                    b.append(O[i, j])

    A = np.array(A)
    b = np.array(b)

    # Линейное решение с добавлением L2-регуляризации
    lambda_reg = 0.01  # Попробуйте уменьшить регуляризацию
    I_reg = np.eye(A.shape[1])
    A_reg = np.vstack([A, np.sqrt(lambda_reg) * I_reg])
    b_reg = np.concatenate([b, np.zeros(A.shape[1])])

    K_unknown_flat, residuals, rank, s = lstsq(A_reg, b_reg, rcond=None)
    K_unknown = K_unknown_flat.reshape((3, 3))

    # Инициализация весов случайным образом
    initial_weights = np.random.uniform(0, 1, len(kernels_order))

    def objective(weights):
        mse_val = total_mse(kernels_order, weights, K_unknown)
        print(f"Current weights: {weights}, Current MSE: {mse_val}")  # Для отладки
        return mse_val

    result = minimize(objective, initial_weights, method='Nelder-Mead')  # Изменил метод оптимизации

    return result.x, K_unknown


# Основной цикл оптимизации
kernels = load_kernels()
inputs, outputs = load_images_and_outputs()

best_mse = float('inf')
best_kernel = None
best_kernels_order = None
best_kernel_result = None
best_weights = None

# Перебор всех перестановок ядер с оптимизацией неизвестного ядра и весов
kernel_indices = list(range(len(kernels)))  # Изначальный порядок
for perm in permutations(kernel_indices):
    weights, K_unknown = optimize_kernel_and_weights(list(perm) + ['unknown'])
    mse = total_mse(list(perm) + ['unknown'], weights, K_unknown)
    if mse < best_mse:
        best_mse = mse
        best_kernels_order = perm
        best_kernel_result = K_unknown
        best_weights = weights

# Сохранение лучшего ядра и оптимальных весов
ordered_kernels = []
for idx in best_kernels_order:
    ordered_kernels.append(kernels[idx])
ordered_kernels.append(best_kernel_result)

# Сохраняем ядра в файл
with open('answers.csv', 'w') as f:
    for k in ordered_kernels:
        flat_k = k.flatten()
        line = ','.join(map(str, flat_k))
        f.write(line + '\n')

# Сохраняем веса в файл
with open('optimized_weights.txt', 'w') as f:
    f.write(','.join(map(str, best_weights)) + '\n')

print(f"Best MSE: {best_mse}")
print(f"Best kernel: \n{best_kernel_result}")