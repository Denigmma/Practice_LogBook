import time
import pandas as pd
import requests
import os

PREDICT_URL = "http://127.0.0.1:8000/predict"
UPDATE_URL = "http://127.0.0.1:8000/update"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "../data/sensor_test_data.csv")
WINDOW_SIZE = 20

def simulate_all_sensors(csv_file: str, window_size: int):
    """
    Имитация потока данных для всех установок.
    Накапливаются данные для каждой установки.
    Когда окно длиной window_size заполнено, отправляется запрос на предсказание,
    а затем обновляется информация на дашборде с указанием времени, взятого из датасета.
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
                int(row['segment_id'][1:]),
                float(row['pressure']),
                float(row['temperature']),
                float(row['vibration']),
                float(row['flow_rate'])
            ]
            sensor_windows[sensor].append(features)

        for sensor in sensors:
            if len(sensor_windows[sensor]) >= window_size:
                window_data = sensor_windows[sensor][-window_size:]
                payload = {"data": window_data}
                try:
                    response = requests.post(PREDICT_URL, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        probability = result.get("failure_probability", 0)
                        print(f"Sensor {sensor} at {current_time} - Предсказание: {probability}")
                        update_payload = {
                            "sensor": sensor,
                            "probability": probability,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                        }
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
