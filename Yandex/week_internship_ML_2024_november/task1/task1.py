import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Шаг 1: Загрузка обучающего набора данных
train_data = pd.read_csv('train.tsv', sep='\t', header=None)
X_train = train_data.iloc[:, :-1].values  # Первые 100 колонок - признаки
y_train = train_data.iloc[:, -1].values   # Последняя колонка - целевая переменная

# Шаг 2: Обучение линейной модели
model = LinearRegression()
model.fit(X_train, y_train)

# Шаг 3: Загрузка тестового набора данных
test_data = pd.read_csv('test.tsv', sep='\t', header=None)
X_test = test_data.values

# Шаг 4: Предсказание и сохранение результата
predictions = model.predict(X_test)

# Округление предсказаний до 8 знаков после запятой
predictions = np.round(predictions, 8)

# Сохранение результата в файл answer.tsv
pd.DataFrame(predictions).to_csv('answer.tsv', sep='\t', header=False, index=False)
