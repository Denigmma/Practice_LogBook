import numpy as np

train_data_x = [1, 2, 3, 4, 5]
train_data_y=['a','b','c','d','e']

# Функция для создания временных последовательностей
def create_sequences(data_x, data_y, time_steps):
    x, y = [], []
    for i in range(len(data_x) - time_steps):
        x.append(data_x[i:i + time_steps])
        y.append(data_y[i + time_steps])
    return np.array(x), np.array(y)

# Задаем количество временных шагов
time_steps = 2

# Преобразование данных в последовательности
train_data_x, train_data_y = create_sequences(train_data_x, train_data_y, time_steps)

print(train_data_x, train_data_y)