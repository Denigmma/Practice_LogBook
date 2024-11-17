import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

base = os.path.dirname(__file__)
csv_path = os.path.join(base, '../data_sets/')

data = pd.read_csv(csv_path + 'organisations.csv')
features = pd.read_csv(csv_path + 'features.csv')
rubrics = pd.read_csv(csv_path + 'rubrics.csv')

clean_data = data.dropna(subset=['average_bill'])
clean_data = clean_data[clean_data['average_bill'] <= 2500]

clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)

# Обработка столбца 'city'
city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
city_train = city_encoder.fit_transform(clean_data_train[['city']])
city_test = city_encoder.transform(clean_data_test[['city']])

# Обработка столбца 'rating'
rating_train = clean_data_train[['rating']].values
rating_test = clean_data_test[['rating']].values

# Обработка столбцов 'rubrics_id' и 'features_id'
rubrics_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
features_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

rubrics_train = rubrics_encoder.fit_transform(clean_data_train[['rubrics_id']])
rubrics_test = rubrics_encoder.transform(clean_data_test[['rubrics_id']])

features_train = features_encoder.fit_transform(clean_data_train[['features_id']])
features_test = features_encoder.transform(clean_data_test[['features_id']])

# Создание матрицы из всех обработанных данных
X_train = np.hstack((city_train, rating_train, rubrics_train, features_train))
X_test = np.hstack((city_test, rating_test, rubrics_test, features_test))

# Обучение CatBoost модели
clf = CatBoostRegressor()
clf.fit(X_train, clean_data_train['average_bill'])

# Предсказываем на тестовой выборке
predictions = clf.predict(X_test)

# Сохраняем предсказания в формате .csv
predictions_df = pd.DataFrame({'index': clean_data_test.index, 'prediction': predictions})
predictions_df.to_csv('predictions.csv', index=False)

# Оценка модели на обучающей выборке
train_predictions = clf.predict(X_train)
rmse_train = mean_squared_error(clean_data_train['average_bill'], train_predictions, squared=False)

# Оценка модели на тестовой выборке
rmse_test = mean_squared_error(clean_data_test['average_bill'], predictions, squared=False)

# Выводим результаты
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
