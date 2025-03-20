# import time
# import pandas as pd
# import requests
#
# # URL API, к которому будут отправляться запросы
# API_URL = "http://127.0.0.1:8000/predict"
# # Путь к CSV-файлу с данными
# CSV_FILE = "../data/sensor_val_data.csv"
# # Размер последовательности (скользящего окна)
# WINDOW_SIZE = 20
# # Выбираем установку, данные по которой будем отправлять (например, "A1")
# SEGMENT = "A1"
#
#
# def simulate_segment_data(csv_file: str, segment: str, window_size: int):
#     """
#     Функция имитирует поток данных для выбранной установки.
#     Из CSV-файла считываются данные, фильтруются по установке,
#     затем формируется скользящее окно данных и отправляется на API.
#
#     :param csv_file: путь к CSV-файлу с данными.
#     :param segment: идентификатор установки, например, "A1".
#     :param window_size: количество строк в последовательности (например, 20).
#     """
#     # Загружаем данные из CSV-файла
#     df = pd.read_csv(csv_file)
#
#     # Фильтруем данные по выбранной установке
#     df_segment = df[df["segment_id"] == segment].copy()
#
#     # Преобразуем временные метки в datetime и сортируем
#     df_segment["timestamp"] = pd.to_datetime(df_segment["timestamp"])
#     df_segment.sort_values("timestamp", inplace=True)
#     df_segment.reset_index(drop=True, inplace=True)
#
#     # Преобразуем segment_id в числовой код: "A1" -> 1, "A2" -> 2, ...
#     df_segment["segment_code"] = df_segment["segment_id"].apply(lambda x: int(x[1:]))
#
#     num_rows = len(df_segment)
#     total_windows = num_rows - window_size + 1
#
#     print(f"Запущена симуляция для установки {segment}. Всего окон: {total_windows}")
#
#     # Перебираем все возможные окна (скользящее окно)
#     for i in range(total_windows):
#         window_data = []
#         # Формируем окно из последовательных строк
#         for j in range(i, i + window_size):
#             row = df_segment.iloc[j]
#             # Преобразуем значения в стандартные типы Python: int и float
#             features = [
#                 int(row["segment_code"]),
#                 float(row["pressure"]),
#                 float(row["temperature"]),
#                 float(row["vibration"]),
#                 float(row["flow_rate"])
#             ]
#             window_data.append(features)
#
#         payload = {"data": window_data}
#
#         try:
#             response = requests.post(API_URL, json=payload)
#             if response.status_code == 200:
#                 result = response.json()
#                 print(f"Окно {i + 1}/{total_windows} - Предсказание: {result}")
#             else:
#                 print(f"Окно {i + 1}/{total_windows} - Ошибка {response.status_code}: {response.text}")
#         except Exception as e:
#             print(f"Окно {i + 1}/{total_windows} - Исключение: {e}")
#
#         # Задержка в 1 секунду для имитации поступления данных
#         time.sleep(1)
#
#
# if __name__ == "__main__":
#     simulate_segment_data(CSV_FILE, SEGMENT, WINDOW_SIZE)


import time
import pandas as pd
import requests

# URL API для предсказания и обновления
PREDICT_URL = "http://127.0.0.1:8000/predict"
UPDATE_URL = "http://127.0.0.1:8000/update"
# Путь к CSV-файлу с данными
CSV_FILE = "../data/sensor_test_data.csv"
# Размер окна (количество показаний для формирования последовательности)
WINDOW_SIZE = 20


def simulate_all_sensors(csv_file: str, window_size: int):
    """
    Имитация потока данных для всех установок.
    Накапливаются данные для каждой установки.
    Как только окно длиной window_size заполнено, отправляется запрос на предсказание,
    и затем отправляется обновление с полученным результатом на эндпоинт /update.
    """
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    sensors = df['segment_id'].unique()
    sensor_windows = {sensor: [] for sensor in sensors}

    unique_times = df['timestamp'].unique()
    print(f"Запущена симуляция для установок: {', '.join(sensors)}")

    for current_time in unique_times:
        current_rows = df[df['timestamp'] == current_time]
        for idx, row in current_rows.iterrows():
            sensor = row['segment_id']
            features = [
                int(row['segment_id'][1:]),  # "A1" -> 1
                float(row['pressure']),
                float(row['temperature']),
                float(row['vibration']),
                float(row['flow_rate'])
            ]
            sensor_windows[sensor].append(features)

        # Для каждой установки, если окно заполнено, выполняем предсказание
        for sensor in sensors:
            if len(sensor_windows[sensor]) >= window_size:
                window_data = sensor_windows[sensor][-window_size:]
                payload = {"data": window_data}
                try:
                    # Отправляем запрос на предсказание
                    response = requests.post(PREDICT_URL, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        probability = result.get("failure_probability", 0)
                        print(f"Sensor {sensor} at {current_time} - Предсказание: {probability}")
                        # Отправляем обновление на дашборд
                        update_payload = {"sensor": sensor, "probability": probability}
                        update_response = requests.post(UPDATE_URL, json=update_payload)
                        if update_response.status_code != 200:
                            print(f"Ошибка обновления для {sensor}: {update_response.text}")
                    else:
                        print(f"Sensor {sensor} at {current_time} - Ошибка: {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"Sensor {sensor} at {current_time} - Исключение: {e}")
        time.sleep(1)


if __name__ == "__main__":
    simulate_all_sensors(CSV_FILE, WINDOW_SIZE)
