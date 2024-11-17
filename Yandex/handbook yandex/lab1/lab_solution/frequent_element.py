import numpy as np
import pandas as pd
import os
import statistics
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from scipy.stats import mode
from scipy import stats
from collections import Counter
from collections import defaultdict
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import tqdm

base = os.path.dirname(__file__)
csv_path = os.path.join(base, '../data_sets/')

data = pd.read_csv(csv_path + 'organisations.csv')
features = pd.read_csv(csv_path + 'features.csv')
rubrics = pd.read_csv(csv_path + 'rubrics.csv')

def most_frequent(nums):
    # Используем np.bincount для подсчета вхождений элементов
    counts = np.bincount(nums)
    # Находим индекс максимального значения в counts, который соответствует самому частому элементу
    most_common_element = np.argmax(counts)
    return most_common_element


# task 1
# Удаляем строки, где значение average_bill отсутствует (NaN)
data = data.dropna(subset=['average_bill'])
# Преобразуем столбец 'average_bill' в массив целых чисел
nums = data['average_bill'].astype(int).values
print("Самый частый средний чек:",most_frequent(nums))
print("самый большой чек", max(nums))


# task 2
# Удаляем строки, где значение average_bill отсутствует (NaN) или превышает 2500
data_cleaned = data.dropna(subset=['average_bill'])
data_cleaned = data_cleaned[data_cleaned['average_bill'] <= 2500]
print(f'Количество заведений после очистки: {len(data_cleaned)}')

# task 3
data_cleaned = data.dropna(subset=['average_bill', 'city'])
average_bill_msk = data_cleaned[data_cleaned['city'] == 'msk']['average_bill']
average_bill_spb = data_cleaned[data_cleaned['city'] == 'spb']['average_bill']
mean_msk = statistics.mean(average_bill_msk)
mean_spb = statistics.mean(average_bill_spb)

difference = round(abs(mean_spb-mean_msk))
nums = data['average_bill'].astype(int).values
print("Сред ариф чек:",statistics.mean(nums))
print('Разность между средними арифметическими average_bill в кафе msk/spb:',difference)

# -----------------------------------------------------------------------------------------------------------------------
# введение в ML

clean_data = data.dropna(subset=['average_bill'])
clean_data = clean_data[clean_data['average_bill'] <= 2500]

clean_data_train, clean_data_test = train_test_split(clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        # Сохраняем среднее значение таргета (y)
        self.mean_ = np.mean(y)

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        # Возвращаем среднее значение для всех предсказаний
        return np.full(shape=(X.shape[0],), fill_value=self.mean_)


class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        # Находим самый частый класс
        mode_result = stats.mode(y, keepdims=True)
        self.most_frequent_ = mode_result.mode[0]  # Извлекаем первый элемент из массива


    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        # Возвращаем самый частый класс для всех предсказаний
        return np.full(shape=(X.shape[0],), fill_value=self.most_frequent_)


# обучение
reg = MeanRegressor()
reg.fit(y=clean_data_train['average_bill'])

clf = MostFrequentClassifier()
clf.fit(y=clean_data_train['average_bill'])

# примеры работы моделек, предсказания на тестовых данных
reg_predictions = reg.predict(X=clean_data_test)
clf_predictions = clf.predict(X=clean_data_test)

print('Mean prediction:', reg_predictions[0])  # Например, вывод одного из предсказаний
print('Most frequent class prediction:', clf_predictions[0])



#  регрессор, для каждого заведения предсказывающий среднее значение в том же городе


class CityMeanRegressor(RegressorMixin):
    '''
    Класс CityMeanRegressor предназначен для предсказания среднего чека для
        каждого заведения на основе города, в котором оно находится.

        Метод fit:
        - Использует данные X (содержащие город) и y (средний чек).
        - Сохраняет среднее значение среднего чека для каждого уникального города в словарь self.city_means_.
        - Для каждого города вычисляется среднее значение чеков на основе обучающей выборки.

        Метод predict:
        - На основе города каждого заведения возвращает предсказанное среднее значение чека.
        - Если данные по городу отсутствуют, возвращает среднее по всем городам.
    '''

    def fit(self, X, y):
        # Сохраняем среднее значение для каждого города
        self.city_means_ = {}
        cities = X['city'].unique()  # Получаем уникальные города
        for city in cities:
            # Рассчитываем среднее значение среднего чека для каждого города
            city_mask = X['city'] == city
            self.city_means_[city] = np.mean(y[city_mask])

    def predict(self, X):
        # Предсказываем среднее значение для города, в котором находится заведение
        predictions = np.zeros(len(X))
        for i, city in enumerate(X['city']):
            predictions[i] = self.city_means_.get(city, np.mean(list(self.city_means_.values())))
            # Если нет данных для города, используем среднее по всем городам
        return predictions

# создание модели и обучении на данных города и чеков в этом городе
city_reg = CityMeanRegressor()
city_reg.fit(clean_data_train[['city']], clean_data_train['average_bill'])

# Вывод среднего значения для каждого города
print("Среднее значение среднего чека для каждого города:")
for city, mean_value in city_reg.city_means_.items():
    print(" ",f"{city}: {mean_value}")

# Предсказания для тестовых данных
predictions = city_reg.predict(clean_data_test[['city']])

# Вычисление RMSE (метрики качества)
rmse = np.sqrt(mean_squared_error(clean_data_test['average_bill'], predictions))
print("RMSE for CityMeanRegressor:", rmse)



# добавим в учет типы заведений, чтобы предсказывать по заведению
# медиану средних чеков среди тех в обучающей выборке, у которых с ним одинаковые modified_rubrics и город

# массив - словарь с ключем - rubric_id
rubrics_map = dict(zip(rubrics['rubric_id'], rubrics['rubric_name']))

# Подсчет количества каждой рубрики в обучающей выборке
rubric_counts = Counter(clean_data_train['rubrics_id'])

limit = 100
# замена малочисленных рубрик на 'other'
def modify_rubrics(rubric_id):
    if rubric_counts[rubric_id] >= limit:
        return rubric_id
    else:
        return 'other'

# Применение функции к данным
clean_data_train['modified_rubrics'] = clean_data_train['rubrics_id'].apply(modify_rubrics)
clean_data_test['modified_rubrics'] = clean_data_test['rubrics_id'].apply(modify_rubrics)

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        # Группируем данные по городу и рубрике и вычисляем медиану для каждого сочетания
        self.medians_ = clean_data_train.groupby(['city', 'modified_rubrics'])['average_bill'].median().to_dict()

    def predict(self, X):
        predictions = []
        for i, row in X.iterrows():
            key = (row['city'], row['modified_rubrics'])
            # Если сочетание город + рубрика есть, берем медиану, иначе возвращаем общее среднее
            predictions.append(self.medians_.get(key, clean_data_train['average_bill'].median()))
        return np.array(predictions)


clf = RubricCityMedianClassifier()

# Обучаем классификатор на обучающей выборке
clf.fit(X=clean_data_train[['city', 'modified_rubrics']], y=clean_data_train['average_bill'])

# 1. Предсказания на обучающей выборке
train_preds = clf.predict(clean_data_train[['city', 'modified_rubrics']])

# 2. Предсказания на тестовой выборке
test_preds = clf.predict(clean_data_test[['city', 'modified_rubrics']])

# 3. RMSE на обучающей выборке
train_rmse = np.sqrt(mean_squared_error(clean_data_train['average_bill'], train_preds))
print(f"Train RMSE: {train_rmse}")

# 4. Balanced Accuracy Score на обучающей выборке
train_balanced_accuracy = balanced_accuracy_score(clean_data_train['average_bill'], train_preds)
print(f"Train Balanced Accuracy Score: {train_balanced_accuracy}")

# 5. RMSE на тестовой выборке
test_rmse = np.sqrt(mean_squared_error(clean_data_test['average_bill'], test_preds))
print(f"Test RMSE: {test_rmse}")

# 6. Balanced Accuracy Score на тестовой выборке
test_balanced_accuracy = balanced_accuracy_score(clean_data_test['average_bill'], test_preds)
print(f"Test Balanced Accuracy Score: {test_balanced_accuracy}")


### След задание: Создадим классификатор, использующий одновременно rubrics_id и features_id

# Конкатенация rubrics_id и features_id с разделителем 'q'
clean_data_train['modified_features'] = clean_data_train['rubrics_id'] + 'q' + clean_data_train['features_id']
clean_data_test['modified_features'] = clean_data_test['rubrics_id'] + 'q' + clean_data_test['features_id']

# Подсчет частоты встречаемости комбинаций в обучающей выборке
train_features_counts = Counter(clean_data_train['modified_features'])

# Замена редких комбинаций в тестовой выборке на 'other'
clean_data_test['modified_features'] = clean_data_test['modified_features'].apply(
    lambda x: x if x in train_features_counts else 'other'
)

# Классификатор на основе медианы
class RubricFeatureMedianClassifier:
    def fit(self, X=None, y=None):
        # Сохраняем медианы по modified_features
        self.medians_ = clean_data_train.groupby('modified_features')['average_bill'].median()
        # Сохраняем глобальную медиану
        self.global_median_ = np.median(clean_data_train['average_bill'])

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            feature = row['modified_features']
            if feature in self.medians_:
                predictions.append(self.medians_.loc[feature])
            else:
                predictions.append(self.global_median_)
        return np.array(predictions)

# Обучаем классификатор
clf = RubricFeatureMedianClassifier()
clf.fit()

# 1. Предсказания на обучающей выборке
train_preds = clf.predict(clean_data_train)

# 2. Предсказания на тестовой выборке
test_preds = clf.predict(clean_data_test)

# 3. RMSE на обучающей выборке
train_rmse = np.sqrt(mean_squared_error(clean_data_train['average_bill'], train_preds))
print(f"Train RMSE: {train_rmse:.2f}")

# 4. Balanced Accuracy Score на обучающей выборке
train_balanced_accuracy = balanced_accuracy_score(clean_data_train['average_bill'].round(), train_preds.round())
print(f"Train Balanced Accuracy Score: {train_balanced_accuracy:.2f}")

# 5. RMSE на тестовой выборке
test_rmse = np.sqrt(mean_squared_error(clean_data_test['average_bill'], test_preds))
print(f"Test RMSE: {test_rmse:.2f}")

# 6. Balanced Accuracy Score на тестовой выборке
test_balanced_accuracy = balanced_accuracy_score(clean_data_test['average_bill'].round(), test_preds.round())
print(f"Test Balanced Accuracy Score: {test_balanced_accuracy:.2f}")

# Сохранение предсказаний для тестовой выборки в .csv файл
# Округляем предсказания до целого числа
test_preds = np.round(clf.predict(clean_data_test))

# Сохраняем предсказания вместе с индексами заведений в .csv файл
predictions_with_index = pd.DataFrame({
    'index': clean_data_test.index,  # Индексы заведений из исходного датасета
    'prediction': test_preds         # Предсказания классификатора
})

# Сохраняем результат в файл
predictions_with_index.to_csv('test_predictions.csv', index=False, header=False)

