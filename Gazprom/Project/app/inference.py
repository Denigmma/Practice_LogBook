import numpy as np
import tensorflow as tf


def load_model(model_path: str):
    """
    Загружает модель из указанного пути.
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_input(data: list, seq_len: int = 20):
    """
    Преобразует входные данные в numpy-массив для модели.

    Ожидается, что data — список длиной seq_len,
    где каждая запись — список из 5 признаков:
    [segment_code, pressure, temperature, vibration, flow_rate].

    Выполняется нормализация:
      - segment_code: [1, 10] → [0, 1]
      - pressure: [5.0, 10.0] → [0, 1]
      - temperature: [15.0, 30.0] → [0, 1]
      - vibration: [0.02, 0.1] → [0, 1]
      - flow_rate: [100, 300] → [0, 1]

    Возвращает данные в форме (1, seq_len, 5) для батча из одного примера.
    """
    data = np.array(data, dtype=np.float32)
    if data.shape != (seq_len, 5):
        raise ValueError(f"Ожидалась форма ({seq_len}, 5), но получена {data.shape}")
    # Применяем нормализацию:
    data[:, 0] = (data[:, 0] - 1) / 9.0
    data[:, 1] = (data[:, 1] - 5.0) / 5.0
    data[:, 2] = (data[:, 2] - 15.0) / 15.0
    data[:, 3] = (data[:, 3] - 0.02) / (0.1 - 0.02)
    data[:, 4] = (data[:, 4] - 100.0) / 200.0
    # Добавляем размерность батча
    data = np.expand_dims(data, axis=0)
    return data


def predict(model, input_data: list):
    """
    Выполняет предсказание вероятности отказа.

    :param model: загруженная модель TensorFlow/Keras.
    :param input_data: список длиной 20, где каждая запись содержит 5 признаков.
    :return: вероятность отказа (float).
    """
    preprocessed_data = preprocess_input(input_data)
    prediction = model.predict(preprocessed_data)
    return float(prediction[0][0])
