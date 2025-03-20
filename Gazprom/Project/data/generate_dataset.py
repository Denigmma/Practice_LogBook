import csv
import random
from datetime import datetime, timedelta

# Параметры генерации
NUM_SECONDS = 500  # 500 секунд
SEGMENTS = [f"A{i}" for i in range(1, 11)]  # Установки от A1 до A10
START_TIME = datetime(2025, 3, 15, 11, 0, 0)

k=0

# Диапазоны для признаков
PRESSURE_RANGE = (5.0, 10.0)  # давление в барах
TEMPERATURE_RANGE = (15.0, 30.0)  # температура в °C
VIBRATION_RANGE = (0.02, 0.1)  # уровень вибрации в мм/с
FLOW_RATE_RANGE = (100.0, 300.0)  # расход газа в м³/ч

# OUTPUT_FILE = "sensor_train_data.csv"
# OUTPUT_FILE = "sensor_val_data.csv"
OUTPUT_FILE = "sensor_test_data.csv"


with open(OUTPUT_FILE, mode="w", newline="") as csvfile:
    fieldnames = ['timestamp', 'segment_id', 'pressure', 'temperature', 'vibration', 'flow_rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Для каждой секунды генерируем данные для всех установок
    for second in range(NUM_SECONDS):
        current_time = START_TIME + timedelta(seconds=second)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        for segment_id in SEGMENTS:
            k+=0.01
            # pressure = round(random.uniform(*PRESSURE_RANGE), 2)
            # temperature = round(random.uniform(*TEMPERATURE_RANGE), 1)
            # vibration = round(random.uniform(*VIBRATION_RANGE), 3)
            # flow_rate = round(random.uniform(*FLOW_RATE_RANGE), 1)
            pressure = round(min(*PRESSURE_RANGE)+k, 2)
            temperature = round(min(*TEMPERATURE_RANGE)*k, 1)
            vibration = round(min(*VIBRATION_RANGE)*k, 3)
            flow_rate = round(min(*FLOW_RATE_RANGE)*k, 1)

            writer.writerow({
                'timestamp': timestamp,
                'segment_id': segment_id,
                'pressure': pressure,
                'temperature': temperature,
                'vibration': vibration,
                'flow_rate': flow_rate
            })

print(f"Датасет успешно сохранён в {OUTPUT_FILE} с {NUM_SECONDS * len(SEGMENTS)} строками.")
