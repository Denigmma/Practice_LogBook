import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Функция для загрузки данных
def load_data(filename):
    # Используем регулярное выражение для разделения по одному или нескольким пробелам
    df = pd.read_csv(filename, header=None, sep=r'\s+', on_bad_lines='skip')
    return df

# Пример загрузки данных
train_data = load_data('train.csv')
print(train_data.head())



# Преобразование данных в нужный формат
def preprocess_data(df):
    # Преобразуем данные в numpy массивы
    X = df.iloc[:, :-2].values  # Все столбцы кроме последних двух
    target_x = df.iloc[:, -2].values
    target_y = df.iloc[:, -1].values
    return X, target_x, target_y


# Модель предсказания траектории
def train_model(X_train, target_x_train, target_y_train):
    model_x = LinearRegression()
    model_y = LinearRegression()

    model_x.fit(X_train, target_x_train)
    model_y.fit(X_train, target_y_train)

    return model_x, model_y


def predict_trajectory(model_x, model_y, X_test):
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)
    return pred_x, pred_y


def compute_ADE(target_x, target_y, pred_x, pred_y):
    T = len(target_x)
    error = np.sqrt((target_x - pred_x) ** 2 + (target_y - pred_y) ** 2)
    return np.mean(error)


# Загрузка тренировочных данных
train_data = load_data('train.csv')
X_train, target_x_train, target_y_train = preprocess_data(train_data)

# Тренировка модели
model_x, model_y = train_model(X_train, target_x_train, target_y_train)

# Загрузка тестовых данных
test_data = load_data('test.csv')
X_test, _, _ = preprocess_data(test_data)

# Получение предсказаний для тестовых данных
pred_x, pred_y = predict_trajectory(model_x, model_y, X_test)

# Формирование и вывод предсказаний в нужном формате
predictions = np.column_stack((pred_x, pred_y))

# Сохраняем предсказания в нужном формате
for pred in predictions:
    print(" ".join(map(str, pred)))

