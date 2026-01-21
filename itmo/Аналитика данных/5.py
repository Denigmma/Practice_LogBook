# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

df = df.drop(columns=["ID", "Delivery_person_ID"], errors="ignore")
df = df[df["distance_meters"] <= 100_000].copy()

df["is_long_Delivery"] = (df["Time_taken_min"] > 30).astype(int)
df = df.drop(columns=["Time_taken_min"])

X = df.drop(columns=["is_long_Delivery"])
y = df["is_long_Delivery"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

share_fast = round((y_train == 0).mean(), 3)

model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = round(accuracy_score(y_test, y_pred), 3)

print("Доля быстрых доставок (train):", share_fast)
print("Accuracy на test:", accuracy)
