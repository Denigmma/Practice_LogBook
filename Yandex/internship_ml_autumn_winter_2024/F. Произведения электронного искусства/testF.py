import numpy as np
import pandas as pd
import glob
from PIL import Image
from scipy.signal import convolve2d
from numpy.linalg import lstsq
from itertools import permutations

# 1. Проверяем, что файл с ядрами существует и загружаем его
try:
    kernels_df = pd.read_csv('best_pictures/algos.csv', header=None)
    kernels = [np.array(kernels_df.iloc[i]).reshape((3, 3)) for i in range(2)]
    print("Kernels loaded successfully.")
except FileNotFoundError:
    print("Error: algos.csv file not found.")
    exit()

# 2. Проверяем, что файлы изображений и выходных данных существуют
input_files = sorted(glob.glob('best_pictures/*.png'))
output_files = sorted(glob.glob('best_pictures/*.txt'))

inputs = []
outputs = []

for img_file, out_file in zip(input_files, output_files):
    # Чтение изображения
    try:
        img = Image.open(img_file).convert('L')
        img = np.array(img, dtype=np.float64)
        inputs.append(img)
        print(f"Loaded image: {img_file}, shape: {img.shape}")
    except FileNotFoundError:
        print(f"Error: {img_file} not found.")
        continue

    # Чтение выходного массива
    try:
        out = np.loadtxt(out_file)
        outputs.append(out)
        print(f"Loaded output for {img_file}: shape {out.shape}")
    except FileNotFoundError:
        print(f"Error: {out_file} not found.")
        continue

# Возможные перестановки ядер
kernel_indices = [0, 1, 'unknown']
kernel_permutations = list(permutations(kernel_indices))

best_mse = float('inf')
best_order = None
best_kernel = None

# Перебор всех возможных перестановок ядер
for perm in kernel_permutations:
    # Для каждого изображения создаём уравнения для нахождения неизвестного ядра
    A = []
    b = []
    for I, O in zip(inputs, outputs):
        # Применение известных ядер в заданном порядке
        temp = I.copy()
        for idx in perm:
            if idx == 'unknown':
                break
            temp = convolve2d(temp, kernels[idx], mode='same')

        # Теперь temp - это результат после применения известных ядер
        # O - финальный результат после применения всех ядер
        # Мы можем составить уравнение O = temp * K_unknown (свертка)

        # Подготовка для сбора уравнений
        h, w = O.shape
        for i in range(h):
            for j in range(w):
                # Получаем 3x3 патч из temp
                patch = temp[i:i + 3, j:j + 3]
                if patch.shape != (3, 3):
                    continue  # Пропускаем, если патч неполный (границы)
                A.append(patch.flatten())
                b.append(O[i + 1, j + 1])  # Центральный пиксель соответствует (i+1, j+1)

    A = np.array(A)
    b = np.array(b)

    # Решение для неизвестного ядра с помощью метода наименьших квадратов
    K_unknown_flat, residuals, rank, s = lstsq(A, b, rcond=None)
    K_unknown = K_unknown_flat.reshape((3, 3))

    # Теперь вычисляем ошибку реконструкции
    total_mse = 0
    for I, O in zip(inputs, outputs):
        temp = I.copy()
        for idx in perm:
            if idx == 'unknown':
                temp = convolve2d(temp, K_unknown, mode='same')  # Используем 'same' для сохранения размеров
            else:
                temp = convolve2d(temp, kernels[idx], mode='same')  # Также 'same' для известных ядер

        # Вычисляем MSE
        if temp.shape != O.shape:
            temp = temp[:O.shape[0], :O.shape[1]]  # Обрезаем, если размеры различаются
        mse = np.mean((temp - O) ** 2)
        total_mse += mse
    avg_mse = total_mse / len(inputs)

    # Проверка, если это лучший порядок
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_order = perm
        best_kernel = K_unknown

# Сохраняем ядра в лучшем порядке
ordered_kernels = []
for idx in best_order:
    if idx == 'unknown':
        ordered_kernels.append(best_kernel)
    else:
        ordered_kernels.append(kernels[idx])

# Сохраняем ядра в файл с округлением до 6 знаков после запятой
with open('restored_algos.csv', 'w') as f:
    for k in ordered_kernels:
        flat_k = np.round(k.flatten(), 6)  # Округляем значения до 6 знаков после запятой
        line = ','.join(map(str, flat_k))
        f.write(line + '\n')

print(f"Best MSE: {best_mse}")
print(f"Best kernel: \n{np.round(best_kernel, 6)}")  # Округляем финальное ядро до 6 знаков
