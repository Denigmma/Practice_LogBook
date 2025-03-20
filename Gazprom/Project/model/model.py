import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Функция для вычисления метки отказа на основе комплексных условий
def failure_condition(row):
    """
    Определяет отказ установки по комплексным условиям:
    - Утечка: низкое давление (<6.0) и высокий расход (>280.0)
    - Механическая проблема: высокая вибрация (>0.09) и давление (<6.5)
    - Блокировка/перегруз: высокое давление (>9.5) и низкий расход (<120.0)
    - Комбинированное отклонение: давление вне нормы (менее 6.2 или более 9.3) и аномалии по вибрации (>0.08) или расходу (>270.0)
    """
    cond1 = (row['pressure'] < 6.0) and (row['flow_rate'] > 280.0)
    cond2 = (row['vibration'] > 0.09) and (row['pressure'] < 6.5)
    cond3 = (row['pressure'] > 9.5) and (row['flow_rate'] < 120.0)
    cond4 = ((row['pressure'] < 6.2) or (row['pressure'] > 9.3)) and ((row['vibration'] > 0.08) or (row['flow_rate'] > 270.0))
    return int(cond1 or cond2 or cond3 or cond4)

# 2. Функция для формирования последовательностей
def create_sequences(data, seq_len=20):
    """
    Формирует последовательности данных и метки для каждой последовательности.
    Используем признаки: segment_code, pressure, temperature, vibration, flow_rate.
    Нормализация производится согласно диапазонам.
    """
    X, y = [], []
    # Группируем данные по установкам (segment_code)
    for segment, group in data.groupby('segment_code'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        for i in range(len(group) - seq_len + 1):
            seq = group.iloc[i:i+seq_len]
            label = failure_condition(seq.iloc[-1])
            features = seq[['segment_code', 'pressure', 'temperature', 'vibration', 'flow_rate']].values.astype(np.float32)
            # Нормализация:
            # segment_code: [1, 10] → [0, 1]
            features[:, 0] = (features[:, 0] - 1) / 9.0
            # pressure: [5.0, 10.0] → [0, 1]
            features[:, 1] = (features[:, 1] - 5.0) / 5.0
            # temperature: [15.0, 30.0] → [0, 1]
            features[:, 2] = (features[:, 2] - 15.0) / 15.0
            # vibration: [0.02, 0.1] → [0, 1]
            features[:, 3] = (features[:, 3] - 0.02) / (0.1 - 0.02)
            # flow_rate: [100, 300] → [0, 1]
            features[:, 4] = (features[:, 4] - 100.0) / 200.0
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# 3. Загрузка и подготовка данных
# Загрузка обучающего набора
train_df = pd.read_csv("../data/sensor_train_data.csv")
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
train_df['segment_code'] = train_df['segment_id'].apply(lambda x: int(x[1:]))
X_train, y_train = create_sequences(train_df, seq_len=20)

# Загрузка валидационного набора
val_df = pd.read_csv("../data/sensor_val_data.csv")
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
val_df['segment_code'] = val_df['segment_id'].apply(lambda x: int(x[1:]))
X_val, y_val = create_sequences(val_df, seq_len=20)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)

# 4. Функция для построения модели с возможностью настройки гиперпараметров
def build_model(seq_length, num_features, num_layers=2, neurons=[64, 32], dropout_rate=0.2):
    model = models.Sequential()
    # Первый слой GRU с указанием input_shape
    model.add(layers.GRU(neurons[0], return_sequences=(num_layers > 1), input_shape=(seq_length, num_features)))
    model.add(layers.Dropout(dropout_rate))
    # Добавляем дополнительные слои GRU (если заданы)
    for i in range(1, num_layers):
        # Если не последний слой, возвращаем последовательность
        return_seq = (i < num_layers - 1)
        model.add(layers.GRU(neurons[i], return_sequences=return_seq))
        model.add(layers.Dropout(dropout_rate))
    # Выходной слой
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Гиперпараметры модели
seq_length = 20      # Длина последовательности
num_features = 5     # segment_code, pressure, temperature, vibration, flow_rate
num_layers = 2
neurons = [64, 32]
dropout_rate = 0.2

# Создаем модель
model = build_model(seq_length, num_features, num_layers=num_layers, neurons=neurons, dropout_rate=dropout_rate)

# Компиляция модели: функция потерь, оптимизатор, метрики
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# 5. Обучение модели без ранней остановки
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)


# Сохранение обученной модели
model.save('gru_model.h5')
print("\nМодель сохранена в файл 'gru_model.h5'")

# Оценка модели после загрузки
loaded_model = tf.keras.models.load_model('gru_model.h5')
print("\nЗагруженная модель успешно восстановлена.")

# Проверяем её на валидационном наборе
val_eval = loaded_model.evaluate(X_val, y_val, verbose=0)
print("\nОценка загруженной модели на валидации:")
print(f"loss = {val_eval[0]:.4f}, accuracy = {val_eval[1]:.4f}, auc = {val_eval[2]:.4f}")



# 6. Вывод результатов обучения: метрики и лосс функции
print("\nИстория обучения:")
for epoch in range(len(history.history['loss'])):
    print(f"Эпоха {epoch+1:2d}: loss = {history.history['loss'][epoch]:.4f}, "
          f"accuracy = {history.history['accuracy'][epoch]:.4f}, auc = {history.history['auc'][epoch]:.4f} | "
          f"val_loss = {history.history['val_loss'][epoch]:.4f}, "
          f"val_accuracy = {history.history['val_accuracy'][epoch]:.4f}, val_auc = {history.history['val_auc'][epoch]:.4f}")

# 7. Оценка модели на обучающем и валидационном наборе
train_eval = model.evaluate(X_train, y_train, verbose=0)
val_eval = model.evaluate(X_val, y_val, verbose=0)
print("\nОценка на обучающем наборе:")
print(f"loss = {train_eval[0]:.4f}, accuracy = {train_eval[1]:.4f}, auc = {train_eval[2]:.4f}")
print("\nОценка на валидационном наборе:")
print(f"loss = {val_eval[0]:.4f}, accuracy = {val_eval[1]:.4f}, auc = {val_eval[2]:.4f}")





# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from datetime import datetime
#
#
# # 1. Функция для вычисления метки отказа на основе комплексных условий
# def failure_condition(row):
#     # row – строка с необработанными (сырыми) значениями показателей
#     # Условие 1: утечка (низкое давление и высокий расход)
#     cond1 = (row['pressure'] < 6.0) and (row['flow_rate'] > 280.0)
#     # Условие 2: механическая проблема (высокая вибрация и снижение давления)
#     cond2 = (row['vibration'] > 0.09) and (row['pressure'] < 6.5)
#     # Условие 3: блокировка или перегруз (высокое давление и низкий расход)
#     cond3 = (row['pressure'] > 9.5) and (row['flow_rate'] < 120.0)
#     # Условие 4: комбинированное отклонение: давление вне нормы и аномалии по вибрации или расходу
#     cond4 = ((row['pressure'] < 6.2) or (row['pressure'] > 9.3)) and (
#                 (row['vibration'] > 0.08) or (row['flow_rate'] > 270.0))
#     return int(cond1 or cond2 or cond3 or cond4)
#
#
# # 2. Определение кастомного датасета для формирования последовательностей
# class SensorDataset(Dataset):
#     def __init__(self, csv_file, seq_len=20):
#         """
#         csv_file: путь к CSV-файлу (например, data/sensor_train.csv)
#         seq_len: длина последовательности (например, 20 секунд)
#         """
#         self.seq_len = seq_len
#         self.data = pd.read_csv(csv_file)
#
#         # Преобразуем timestamp в datetime для сортировки
#         self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
#         # Преобразуем segment_id в числовой код: A1 -> 1, A2 -> 2, ... A10 -> 10
#         self.data['segment_code'] = self.data['segment_id'].apply(lambda x: int(x[1:]))
#
#         # Сортируем по установке и времени
#         self.data.sort_values(['segment_code', 'timestamp'], inplace=True)
#
#         # Формируем последовательности и вычисляем метки (на основе последнего измерения последовательности)
#         self.sequences = []
#         self.labels = []
#         # Группируем по установкам
#         grouped = self.data.groupby('segment_code')
#         for seg, group in grouped:
#             group = group.reset_index(drop=True)
#             for i in range(len(group) - seq_len + 1):
#                 seq = group.iloc[i:i + seq_len]
#                 # Вычисляем метку отказа для последней строки последовательности
#                 label = failure_condition(seq.iloc[-1])
#                 self.sequences.append(seq)
#                 self.labels.append(label)
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
#         # Извлекаем нужные признаки: segment_code, pressure, temperature, vibration, flow_rate
#         # Приводим значения к float32
#         features = seq[['segment_code', 'pressure', 'temperature', 'vibration', 'flow_rate']].values.astype(np.float32)
#
#         # Нормализация признаков по известным диапазонам:
#         # segment_code: [1, 10] → [0, 1]
#         features[:, 0] = (features[:, 0] - 1) / 9.0
#         # pressure: [5.0, 10.0] → [0, 1]
#         features[:, 1] = (features[:, 1] - 5.0) / 5.0
#         # temperature: [15.0, 30.0] → [0, 1]
#         features[:, 2] = (features[:, 2] - 15.0) / 15.0
#         # vibration: [0.02, 0.1] → [0, 1]
#         features[:, 3] = (features[:, 3] - 0.02) / (0.1 - 0.02)
#         # flow_rate: [100, 300] → [0, 1]
#         features[:, 4] = (features[:, 4] - 100.0) / 200.0
#
#         # Преобразуем в тензор
#         features = torch.tensor(features)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return features, label
#
#
# # 3. Определение GRU-модели (на основе PyTorch)
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x имеет форму: [batch_size, seq_len, input_size]
#         out, _ = self.gru(x)  # out имеет форму [batch_size, seq_len, hidden_size]
#         # Берём выход последнего временного шага
#         out = out[:, -1, :]  # [batch_size, hidden_size]
#         out = self.fc(out)  # [batch_size, output_size]
#         out = self.sigmoid(out)
#         return out
#
#
# # 4. Гиперпараметры и настройки обучения
# SEQ_LEN = 20
# INPUT_SIZE = 5  # segment_code, pressure, temperature, vibration, flow_rate
# HIDDEN_SIZE = 64
# NUM_LAYERS = 2
# OUTPUT_SIZE = 1
# DROPOUT = 0.2
# LEARNING_RATE = 0.001
# BATCH_SIZE = 32
# NUM_EPOCHS = 10
#
# # Пути к данным
# TRAIN_CSV = "../data/sensor_train_data.csv"
# VAL_CSV = "../data/sensor_val_data.csv"
#
# # Создаём датасеты и DataLoader'ы
# train_dataset = SensorDataset(TRAIN_CSV, seq_len=SEQ_LEN)
# val_dataset = SensorDataset(VAL_CSV, seq_len=SEQ_LEN)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# # Инициализируем модель, функцию потерь и оптимизатор
# model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
#
# # 5. Функция для вычисления точности (threshold 0.5)
# def accuracy(predictions, labels):
#     preds = (predictions >= 0.5).float()
#     return (preds == labels.unsqueeze(1)).float().mean().item()
#
#
# # 6. Обучающий цикл
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# for epoch in range(1, NUM_EPOCHS + 1):
#     model.train()
#     train_loss = 0.0
#     train_acc = 0.0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item() * inputs.size(0)
#         train_acc += accuracy(outputs, labels) * inputs.size(0)
#
#     train_loss /= len(train_dataset)
#     train_acc /= len(train_dataset)
#
#     # Валидация
#     model.eval()
#     val_loss = 0.0
#     val_acc = 0.0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels.unsqueeze(1))
#             val_loss += loss.item() * inputs.size(0)
#             val_acc += accuracy(outputs, labels) * inputs.size(0)
#
#     val_loss /= len(val_dataset)
#     val_acc /= len(val_dataset)
#
#     print(f"Epoch {epoch}/{NUM_EPOCHS}: "
#           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
#           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#
# # Сохраняем обученную модель
# os.makedirs("saved_models", exist_ok=True)
# MODEL_PATH = "saved_models/gru_model.pt"
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"Модель сохранена в {MODEL_PATH}")
