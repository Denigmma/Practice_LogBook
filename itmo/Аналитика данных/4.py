# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# ========== 1. Загрузка данных ==========
df = pd.read_csv("dataset.csv")

# ========== 2. Удаление ID-признаков ==========
df = df.drop(columns=["ID", "Delivery_person_ID"], errors="ignore")

# ========== 3. Удаление выбросов по расстоянию ==========
df = df[df["distance_meters"] <= 100_000].copy()

# ======================================================
# ЧАСТЬ 1. БАЗОВАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ
# ======================================================

features_basic = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_meters"
]
target = "Time_taken_min"

X = df[features_basic]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=15
)

# Средний возраст доставщиков (train)
mean_age_train = round(X_train["Delivery_person_Age"].mean(), 3)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
mae_basic = round(mean_absolute_error(y_test, y_pred), 3)

print("БАЗОВАЯ МОДЕЛЬ")
print("Средний возраст (train):", mean_age_train)
print("MAE (test):", mae_basic)

# ======================================================
# ЧАСТЬ 2. ВСЕ ПРИЗНАКИ + MinMaxScaler
# ======================================================

X_full = df.drop(columns=[target])
y_full = df[target]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=15
)

# Масштабирование
scaler = MinMaxScaler()
X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

# Средний возраст после масштабирования
age_index = X_train2.columns.get_loc("Delivery_person_Age")
mean_age_scaled = round(X_train2_scaled[:, age_index].mean(), 3)

# Обучение модели
model2 = LinearRegression()
model2.fit(X_train2_scaled, y_train2)

# Оценка
y_pred2 = model2.predict(X_test2_scaled)
mae_scaled = round(mean_absolute_error(y_test2, y_pred2), 3)

print("\nМОДЕЛЬ С МАСШТАБИРОВАНИЕМ")
print("Средний возраст после MinMaxScaler (train):", mean_age_scaled)
print("MAE (test):", mae_scaled)
