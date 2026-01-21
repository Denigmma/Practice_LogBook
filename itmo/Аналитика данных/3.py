# -*- coding: utf-8 -*-

import pandas as pd

# 1) Считываем данные
path = "dataset.csv"
df = pd.read_csv(path)

# 2) Удаляем признаки ID и Delivery_person_ID
df = df.drop(columns=["ID", "Delivery_person_ID"], errors="ignore")

# 3) Убираем заказы, где distance_meters > 100000
df = df[df["distance_meters"] <= 100_000].copy()

# 4) Считаем ответы
rows_cnt = df.shape[0]
max_dist = round(df["distance_meters"].max(), 3)

# 5) Выводим ответы
print("Количество строк после фильтрации:", rows_cnt)
print("Максимальное distance_meters (до тысячных):", f"{max_dist:.3f}")
