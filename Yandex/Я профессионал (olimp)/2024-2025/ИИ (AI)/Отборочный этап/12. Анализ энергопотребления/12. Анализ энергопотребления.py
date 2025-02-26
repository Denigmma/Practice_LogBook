import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
random_data = pd.read_csv('random.csv')
# print("Train data:")
# print(train_data.head())
# print("\nTest data:")
# print(test_data.head())


#заполнение пропущенных значений
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

#нормализация данных
scaler = StandardScaler()

#нормализуем только столбцы 'G (mkS)' и 'B (mkS)' в train и test
train_data[['G (mkS)', 'B (mkS)']] = scaler.fit_transform(train_data[['G (mkS)', 'B (mkS)']])
test_data[['G (mkS)', 'B (mkS)']] = scaler.transform(test_data[['G (mkS)', 'B (mkS)']])

#разделение на признаки и целевые переменные для обучения
X_train = train_data[['G (mkS)', 'B (mkS)']]  # Признаки (G и B)
y_train = train_data.drop(columns=['G (mkS)', 'B (mkS)'])  # Целевые переменные (классы приборов)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

X_test = test_data[['G (mkS)', 'B (mkS)']]  #признаки для теста
y_test_pred = model.predict(X_test)

submission = pd.DataFrame(y_test_pred, columns=random_data.columns)

submission.to_csv('submission.csv', index=False)

print("Prediction results saved to 'submission.csv'")
