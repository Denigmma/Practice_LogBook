import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#категориальные столбцы
categorical_cols = ['mapName', 'ctTeam', 'tTeam', 'ctBuyType', 'tBuyType']

#словарь для LabelEncoder
label_encoders = {}

#категориальные признаки
for col in categorical_cols:
    #объединяем данные train и test для обучения кодировщика на всех категориях
    all_categories = pd.concat([train_data[col], test_data[col]]).unique()
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(all_categories)  #обучаем LabelEncoder на всех данных

    #трансформируем train и test
    train_data[col] = label_encoders[col].transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])

#выделяем целевую переменную и признаки
X = train_data.drop(columns=['winnerSide'])
y = train_data['winnerSide']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

val_accuracy = model.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

test_predictions = model.predict(test_data)

submission = pd.DataFrame({'winnerSide': test_predictions})
submission.to_csv("submission.csv", index=False)
