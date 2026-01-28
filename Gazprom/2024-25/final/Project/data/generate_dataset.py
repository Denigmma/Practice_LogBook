import csv
import random
from datetime import datetime, timedelta

# Параметры генерации
NUM_SECONDS = 500  # общее число секунд
SEGMENTS = [f"A{i}" for i in range(1, 11)]  # установки A1...A10
START_TIME = datetime(2025, 3, 15, 11, 0, 0)

seed = 42
random.seed(seed)

# Выбор выходного файла
OUTPUT_FILE = f"sensor_test_data.csv"
# OUTPUT_FILE = f"sensor_val_data.csv"
# OUTPUT_FILE = f"sensor_train_data.csv"

# Задаём базовые (начальные) значения для каждого сегмента
segment_params = {}
for segment in SEGMENTS:
    base_pressure = random.uniform(7.0, 9.0)         # нормальное давление
    base_temperature = random.uniform(20.0, 25.0)      # нормальная температура
    base_vibration = random.uniform(0.03, 0.07)        # базовый уровень вибрации
    base_flow_rate = random.uniform(150.0, 250.0)      # нормальный расход газа
    segment_params[segment] = {
        'pressure': base_pressure,
        'temperature': base_temperature,
        'vibration': base_vibration,
        'flow_rate': base_flow_rate,
    }

# Задаём drift (плавное смещение) для каждого сегмента
drift_params = {}
for segment in SEGMENTS:
    drift_pressure = random.uniform(-0.005, 0.005)
    drift_temperature = random.uniform(-0.01, 0.01)
    drift_vibration = random.uniform(-0.001, 0.001)
    drift_flow_rate = random.uniform(-0.2, 0.2)
    drift_params[segment] = {
        'pressure': drift_pressure,
        'temperature': drift_temperature,
        'vibration': drift_vibration,
        'flow_rate': drift_flow_rate,
    }

# Состояние для имитации выхода из строя (failure mode) для каждой установки
failure_status = {segment: {"active": False, "remaining": 0, "mode": None} for segment in SEGMENTS}

# Вероятность запуска предписания для выхода из строя для каждой установки в каждую секунду
failure_trigger_probability = 0.0005


with open(OUTPUT_FILE, mode="w", newline="") as csvfile:
    fieldnames = ['timestamp', 'segment_id', 'pressure', 'temperature', 'vibration', 'flow_rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Генерация данных для каждой секунды
    for second in range(NUM_SECONDS):
        current_time = START_TIME + timedelta(seconds=second)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        for segment in SEGMENTS:
            if failure_status[segment]["active"]:
                mode = failure_status[segment]["mode"]
                current_pressure = segment_params[segment]['pressure'] + mode["pressure_step"]
                current_temperature = segment_params[segment]['temperature'] + mode["temperature_step"]
                current_vibration = segment_params[segment]['vibration'] + mode["vibration_step"]
                current_flow_rate = segment_params[segment]['flow_rate'] + mode["flow_rate_step"]

                failure_status[segment]["remaining"] -= 1
                if failure_status[segment]["remaining"] <= 0:
                    failure_status[segment]["active"] = False
                    failure_status[segment]["mode"] = None

                # Ограничиваем значения, чтобы они не стали отрицательными
                current_pressure = max(current_pressure, 0)
                current_temperature = max(current_temperature, 0)
                current_vibration = max(current_vibration, 0)
                current_flow_rate = max(current_flow_rate, 0)

                segment_params[segment]['pressure'] = current_pressure
                segment_params[segment]['temperature'] = current_temperature
                segment_params[segment]['vibration'] = current_vibration
                segment_params[segment]['flow_rate'] = current_flow_rate

            else:
                current_pressure = segment_params[segment]['pressure'] + drift_params[segment]['pressure'] + random.gauss(0, 0.02)
                current_temperature = segment_params[segment]['temperature'] + drift_params[segment]['temperature'] + random.gauss(0, 0.05)
                current_vibration = segment_params[segment]['vibration'] + drift_params[segment]['vibration'] + random.gauss(0, 0.001)
                current_flow_rate = segment_params[segment]['flow_rate'] + drift_params[segment]['flow_rate'] + random.gauss(0, 0.5)

                if random.random() < failure_trigger_probability:
                    failure_duration = random.randint(5, 10)

                    # sign_pressure = random.choice([1, -1])
                    # failure_pressure_step = sign_pressure * random.uniform(0.4, 0.6)
                    # sign_flow_rate = random.choice([1, -1])
                    # failure_flow_rate_step = sign_flow_rate * random.uniform(3.0, 8.0)
                    # sign_temperature = random.choice([1, -1])
                    # failure_temperature_step = sign_temperature * random.uniform(0.1, 0.3)
                    # sign_vibration = random.choice([1, -1])
                    # failure_vibration_step = sign_vibration * random.uniform(0.001, 0.003)

                    failure_pressure_step = random.uniform(0.4, 0.6)
                    failure_flow_rate_step = -random.uniform(3.0, 8.0)
                    failure_temperature_step = random.uniform(0.1, 0.3)
                    failure_vibration_step = random.uniform(0.001, 0.003)

                    failure_status[segment]["active"] = True
                    failure_status[segment]["remaining"] = failure_duration
                    failure_status[segment]["mode"] = {
                        "pressure_step": failure_pressure_step,
                        "flow_rate_step": failure_flow_rate_step,
                        "temperature_step": failure_temperature_step,
                        "vibration_step": failure_vibration_step
                    }
                    current_pressure = segment_params[segment]['pressure'] + failure_status[segment]["mode"]["pressure_step"]
                    current_temperature = segment_params[segment]['temperature'] + failure_status[segment]["mode"]["temperature_step"]
                    current_vibration = segment_params[segment]['vibration'] + failure_status[segment]["mode"]["vibration_step"]
                    current_flow_rate = segment_params[segment]['flow_rate'] + failure_status[segment]["mode"]["flow_rate_step"]
                    failure_status[segment]["remaining"] -= 1

                    # Ограничиваем значения, чтобы они не стали отрицательными
                    current_pressure = max(current_pressure, 0)
                    current_temperature = max(current_temperature, 0)
                    current_vibration = max(current_vibration, 0)
                    current_flow_rate = max(current_flow_rate, 0)

                    segment_params[segment]['pressure'] = current_pressure
                    segment_params[segment]['temperature'] = current_temperature
                    segment_params[segment]['vibration'] = current_vibration
                    segment_params[segment]['flow_rate'] = current_flow_rate
                else:
                    current_pressure = max(current_pressure, 0)
                    current_temperature = max(current_temperature, 0)
                    current_vibration = max(current_vibration, 0)
                    current_flow_rate = max(current_flow_rate, 0)

                    segment_params[segment]['pressure'] = current_pressure
                    segment_params[segment]['temperature'] = current_temperature
                    segment_params[segment]['vibration'] = current_vibration
                    segment_params[segment]['flow_rate'] = current_flow_rate

            writer.writerow({
                'timestamp': timestamp,
                'segment_id': segment,
                'pressure': round(current_pressure, 2),
                'temperature': round(current_temperature, 1),
                'vibration': round(current_vibration, 3),
                'flow_rate': round(current_flow_rate, 1)
            })

print(f"Датасет успешно сохранён в {OUTPUT_FILE} с {NUM_SECONDS * len(SEGMENTS)} строками.")