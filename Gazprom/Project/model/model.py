import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time
import psutil
import os


def failure_condition(seq, pressure_threshold=9.0, vibration_threshold=0.08, growth_window=5,
                      pressure_growth_percent=0.2):
    """
    Определяет отказ установки по комплексным условиям, анализируя всю последовательность seq (длиной seq_len).

    Сценарии отказа:
      1) Сильный рост давления за всю последовательность: (P_end - P_start) / P_start > pressure_growth_percent.
      2) Непрерывный рост температуры в последних growth_window наблюдениях при том, что давление постоянно выше 9.5.
      3) За последние growth_window шагов давление постоянно выше pressure_threshold, а расход падает (flow_rate на последнем шаге меньше, чем на первом).
      4) За последние growth_window шагов вибрация превышает vibration_threshold и температура непрерывно растёт.
      5) Дополнительные сценарии (утечка, механическая проблема, блокировка/перегруз), но без тривиального условия "pressure > 10.0".

    Параметры:
      seq                    : pandas.DataFrame длиной seq_len с колонками pressure, temperature, vibration, flow_rate и др.
      pressure_threshold     : порог давления для проверки уменьшения расхода (по умолчанию 9.0)
      vibration_threshold    : порог вибрации для проверки (по умолчанию 0.08)
      growth_window          : число последних шагов для проверки непрерывного роста признаков (по умолчанию 5)
      pressure_growth_percent: процентный рост давления за всю последовательность, выше которого считается отказ (по умолчанию 20%)

    Возвращает 1, если обнаружен сценарий отказа, иначе 0.
    """
    p_start = seq.iloc[0]['pressure']
    p_end = seq.iloc[-1]['pressure']
    pressure_growth_ratio = (p_end - p_start) / max(p_start, 0.00001)
    cond_pressure_growth = (pressure_growth_ratio > pressure_growth_percent)

    last_window = seq.iloc[-growth_window:] if len(seq) >= growth_window else seq
    cond_temp_rise = True
    for i in range(len(last_window) - 1):
        if last_window.iloc[i + 1]['temperature'] <= last_window.iloc[i]['temperature']:
            cond_temp_rise = False
            break
    cond_pressure_high = (last_window['pressure'] > 9.5).all()
    cond_temp_and_high_pressure = cond_temp_rise and cond_pressure_high

    cond_pressure_high_window = (last_window['pressure'] > pressure_threshold).all()
    cond_flow_down = False
    if cond_pressure_high_window and len(last_window) > 1:
        flow_start = last_window.iloc[0]['flow_rate']
        flow_end = last_window.iloc[-1]['flow_rate']
        cond_flow_down = (flow_end < flow_start)
    cond_pressure_flow = cond_pressure_high_window and cond_flow_down

    cond_vibration_high = (last_window['vibration'] > vibration_threshold).all()
    cond_temp_rise_for_vib = True
    for i in range(len(last_window) - 1):
        if last_window.iloc[i + 1]['temperature'] <= last_window.iloc[i]['temperature']:
            cond_temp_rise_for_vib = False
            break
    cond_vibration_and_temp = cond_vibration_high and cond_temp_rise_for_vib

    row_last = seq.iloc[-1]
    cond_old_1 = (row_last['pressure'] < 6.0) and (row_last['flow_rate'] > 280.0)
    cond_old_2 = (row_last['vibration'] > 0.09) and (row_last['pressure'] < 6.5)
    cond_old_3 = (row_last['pressure'] > 9.5) and (row_last['flow_rate'] < 120.0)
    cond_old_4 = ((row_last['pressure'] < 6.2) or (row_last['pressure'] > 9.3)) and (
                (row_last['vibration'] > 0.08) or (row_last['flow_rate'] > 270.0))

    failure_flag = (cond_pressure_growth or
                    cond_temp_and_high_pressure or
                    cond_pressure_flow or
                    cond_vibration_and_temp or
                    cond_old_1 or cond_old_2 or cond_old_3 or cond_old_4)
    return int(failure_flag)


def create_sequences(data, seq_len=20):
    """
    Формирует последовательности данных (X) и метки (y), анализируя временные ряды для каждой установки.
    Для формирования метки передаётся вся последовательность в функцию failure_condition.
    """
    X, y = [], []
    for segment, group in data.groupby('segment_code'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        for i in range(len(group) - seq_len + 1):
            seq = group.iloc[i:i + seq_len]
            label = failure_condition(seq)
            features = seq[['segment_code', 'pressure', 'temperature', 'vibration', 'flow_rate']].values.astype(
                np.float32)
            features[:, 0] = (features[:, 0] - 1) / 9.0
            features[:, 1] = (features[:, 1] - 5.0) / 5.0
            features[:, 2] = (features[:, 2] - 15.0) / 15.0
            features[:, 3] = (features[:, 3] - 0.02) / (0.1 - 0.02)
            features[:, 4] = (features[:, 4] - 100.0) / 200.0
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


def build_model(seq_length, num_features, num_layers=2, neurons=[64, 32], dropout_rate=0.2):
    """
    Создаёт модель на базе GRU для бинарной классификации (отказ/без отказа).
    """
    model = models.Sequential()
    model.add(layers.GRU(neurons[0], return_sequences=(num_layers > 1), input_shape=(seq_length, num_features)))
    model.add(layers.Dropout(dropout_rate))
    for i in range(1, num_layers):
        return_seq = (i < num_layers - 1)
        model.add(layers.GRU(neurons[i], return_sequences=return_seq))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


train_df = pd.read_csv("../data/sensor_train_data.csv")
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
train_df['segment_code'] = train_df['segment_id'].apply(lambda x: int(x[1:]))

val_df = pd.read_csv("../data/sensor_val_data.csv")
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
val_df['segment_code'] = val_df['segment_id'].apply(lambda x: int(x[1:]))

seq_length = 20
X_train, y_train = create_sequences(train_df, seq_len=seq_length)
X_val, y_val = create_sequences(val_df, seq_len=seq_length)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)

num_features = 5
num_layers = 2
neurons = [64, 32]
dropout_rate = 0.2
EPOCHS = 30
BATCH_SIZE = 32

model = build_model(seq_length, num_features, num_layers=num_layers, neurons=neurons, dropout_rate=dropout_rate)
loss_function = 'binary_crossentropy'
metrics_list = ['accuracy', tf.keras.metrics.AUC(name='auc')]
model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=metrics_list
)

model.summary()


class MetricsLogger(callbacks.Callback):
    """
    Колбэк для записи метрик обучения и валидации по эпохам в CSV-файл.
    """

    def __init__(self, log_file):
        super(MetricsLogger, self).__init__()
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write("epoch,loss,accuracy,auc,val_loss,val_accuracy,val_auc\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch + 1},{logs.get('loss'):.4f},{logs.get('accuracy'):.4f},{logs.get('auc'):.4f},"
                    f"{logs.get('val_loss'):.4f},{logs.get('val_accuracy'):.4f},{logs.get('val_auc'):.4f}\n")


metrics_logger = MetricsLogger("logs/training_log.txt")

train_eval_before = model.evaluate(X_train, y_train, verbose=0)
val_eval_before = model.evaluate(X_val, y_val, verbose=0)
print("\nBaseline evaluation:")
print(
    f"Training: loss = {train_eval_before[0]:.4f}, accuracy = {train_eval_before[1]:.4f}, auc = {train_eval_before[2]:.4f}")
print(
    f"Validation: loss = {val_eval_before[0]:.4f}, accuracy = {val_eval_before[1]:.4f}, auc = {val_eval_before[2]:.4f}")

start_time = time.time()
process = psutil.Process(os.getpid())

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[metrics_logger]
)

end_time = time.time()
training_time = end_time - start_time
memory_used = process.memory_info().rss / (1024 * 1024)

model.save('saved_models/gru_model.h5')
print("\nМодель сохранена в файл 'gru_model.h5'")

loaded_model = tf.keras.models.load_model('saved_models/gru_model.h5')
print("\nЗагруженная модель успешно восстановлена.")

train_eval_after = loaded_model.evaluate(X_train, y_train, verbose=0)
val_eval_after = loaded_model.evaluate(X_val, y_val, verbose=0)

print("\nFinal evaluation:")
print(
    f"Training: loss = {train_eval_after[0]:.4f}, accuracy = {train_eval_after[1]:.4f}, auc = {train_eval_after[2]:.4f}")
print(f"Validation: loss = {val_eval_after[0]:.4f}, accuracy = {val_eval_after[1]:.4f}, auc = {val_eval_after[2]:.4f}")

with open("logs/total_log.txt", "w") as log_file:
    log_file.write("Type: GRU-based RNN (Sequential model)\n")
    log_file.write(f"Count layers: {num_layers}\n")
    log_file.write(f"Count neurons: {neurons}\n")
    log_file.write(f"Loss function: {loss_function}\n")
    log_file.write(f"Count epochs: {EPOCHS}\n")
    log_file.write("Metrics: " + ", ".join([str(m) for m in metrics_list]) + "\n")
    log_file.write(f"Memory used during training: {memory_used:.2f} MB\n")
    log_file.write(f"Total training time: {training_time:.2f} seconds\n")
    log_file.write("\nBaseline metrics:\n")
    log_file.write(f"Loss before training (Train): {train_eval_before[0]:.4f}\n")
    log_file.write(f"Loss before training (Validation): {val_eval_before[0]:.4f}\n")
    log_file.write("\nFinal metrics:\n")
    log_file.write(f"Loss after training (Train): {train_eval_after[0]:.4f}\n")
    log_file.write(f"Loss after training (Validation): {val_eval_after[0]:.4f}\n")
    log_file.write(
        f"Training metrics: loss = {train_eval_after[0]:.4f}, accuracy = {train_eval_after[1]:.4f}, auc = {train_eval_after[2]:.4f}\n")
    log_file.write(
        f"Validation metrics: loss = {val_eval_after[0]:.4f}, accuracy = {val_eval_after[1]:.4f}, auc = {val_eval_after[2]:.4f}\n")

print("\nИстория обучения:")
for epoch in range(len(history.history['loss'])):
    print(f"Эпоха {epoch + 1:2d}: loss = {history.history['loss'][epoch]:.4f}, "
          f"accuracy = {history.history['accuracy'][epoch]:.4f}, auc = {history.history['auc'][epoch]:.4f} | "
          f"val_loss = {history.history['val_loss'][epoch]:.4f}, "
          f"val_accuracy = {history.history['val_accuracy'][epoch]:.4f}, "
          f"val_auc = {history.history['val_auc'][epoch]:.4f}")

train_eval_final = model.evaluate(X_train, y_train, verbose=0)
val_eval_final = model.evaluate(X_val, y_val, verbose=0)
print("\nОценка на обучающем наборе:")
print(f"loss = {train_eval_final[0]:.4f}, accuracy = {train_eval_final[1]:.4f}, auc = {train_eval_final[2]:.4f}")
print("\nОценка на валидационном наборе:")
print(f"loss = {val_eval_final[0]:.4f}, accuracy = {val_eval_final[1]:.4f}, auc = {val_eval_final[2]:.4f}")