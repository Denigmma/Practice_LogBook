import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Шаг 1: Чтение данных
train_data = pd.read_csv('restaurants_train.txt', sep='\t', header=None, names=['winner', 'r1', 'r2', 'd1', 'd2'])
test_data = pd.read_csv('restaurants.in', sep='\t', header=None, names=['r', 'd'])

# Шаг 2: Обработка данных
# Заменяем -1 на NaN для рейтингов
train_data['r1'] = train_data['r1'].replace(-1, np.nan)
train_data['r2'] = train_data['r2'].replace(-1, np.nan)
# Удаление строк с NaN
train_data.dropna(inplace=True)


# Создаем матрицы признаков и целевую переменную
X = []
y = []

for index, row in train_data.iterrows():
    r1, d1 = row['r1'], row['d1']
    r2, d2 = row['r2'], row['d2']

    if not np.isnan(r1) and not np.isnan(r2):
        X.append([r1, d1, r2, d2])
        y.append(row['winner'])

X = np.array(X)
y = np.array(y)


# Шаг 3: Обучение модели
model = GradientBoostingRegressor()
model.fit(X, y)

# Шаг 4: Оценка для тестовых данных
scores = []
for index, row in test_data.iterrows():
    r, d = row['r'], row['d']
    scores.append(model.predict([[r, d, 0, 0]])[0])  # Для простоты, подставим 0 для r2, d2

# Шаг 5: Форматирование вывода
scores = np.array(scores)
sorted_indices = np.argsort(scores)
output = {idx: score for idx, score in zip(sorted_indices, scores[sorted_indices])}

for idx in sorted(output.keys()):
    print(f"{idx}\t{output[idx]}")
