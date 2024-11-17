import csv
import numpy as np
import random
from degann.test_gru_model.equation_solve.simple_equation import equation_solve, str_eq_to_params

# Уравнение дифференциала, которое будем решать
equation = "(-y)"

# Параметры осей для уравнения
params = {'x': "0, 5, 0.01", 'y': "1, 1, 0.01"}  # Параметры для x и y (диапазон, шаг)

# Размеры сгенерированных выборок
sizes_of_samples = [400, 50]
sequence_length = 10  # Длина последовательности для RNN/GRU


def generate_rnn_data(equation, axes, seq_length):
    """
    Генерирует последовательные данные на основе решения дифференциального уравнения.

    equation: str - Уравнение в строковом формате.
    axes: list - Параметры осей (x, y и другие переменные).
    seq_length: int - Длина последовательности для RNN/GRU.

    Возвращает:
    sequences: np.ndarray - Список последовательностей для обучения.
    targets: np.ndarray - Соответствующие целевые значения для каждой последовательности.
    """
    # Решение уравнения, результатом является таблица точек [x, y]
    data = equation_solve(equation, axes)

    sequences = []
    targets = []

    # Генерация последовательностей и соответствующих целевых значений
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length, 0])  # Последовательности значений x
        targets.append(data[i + seq_length, 1])  # Соответствующие значения y

    return np.array(sequences), np.array(targets)


def save_to_csv(file_path, sequences, targets):
    """
    Сохраняет последовательности и целевые значения в CSV-файл.

    file_path: str - Путь к файлу.
    sequences: np.ndarray - Список последовательностей.
    targets: np.ndarray - Соответствующие целевые значения.
    """
    with open(file_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        for seq, target in zip(sequences, targets):
            row = list(seq) + [target]
            csv_writer.writerow(row)


if __name__ == "__main__":
    # Преобразуем строковые параметры в формат, который ожидает функция equation_solve
    axes = str_eq_to_params(params)

    # Генерируем данные для каждой выборки
    for size in sizes_of_samples:
        # Генерация данных с последовательностью для RNN
        sequences, targets = generate_rnn_data(equation, axes, sequence_length)

        # Сохраняем тренировочные данные
        save_to_csv(f"data/ode_train_{size}.csv", sequences, targets)

        # Сохраняем валидационные данные (в два раза меньшее количество точек)
        val_sequences, val_targets = generate_rnn_data(equation, axes, sequence_length // 2)
        save_to_csv(f"data/ode_validate_{size}.csv", val_sequences, val_targets)
