import pandas as pd

# Загрузите Excel-файл
df = pd.read_excel("time.xlsx")

# Предполагается, что данные находятся в первом столбце
# Извлекаем числа из строк и преобразуем в числа с плавающей точкой
df['Time'] = df.iloc[:, 0].str.extract(r"(\d+\.\d+)").astype(float)

# Находим сумму времени в секундах
total_time_seconds = df['Time'].sum()

# Преобразуем в минуты и секунды
minutes = int(total_time_seconds // 60)
seconds = total_time_seconds % 60

# Форматируем вывод
print(f"Сумма времени: {total_time_seconds:.2f} секунд")
print(f"Это: {minutes} минут(ы) и {seconds:.2f} секунд(ы)")
