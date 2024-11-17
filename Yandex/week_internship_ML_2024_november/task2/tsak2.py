import sys
import numpy as np
from sklearn.svm import LinearSVC


def main():
    # Считываем данные
    input_data = sys.stdin.read().strip().splitlines()
    n, m = map(int, input_data[0].split())
    data = [list(map(float, line.split())) for line in input_data[1:]]

    # Разделяем на признаки и метки классов
    X = np.array([row[:m] for row in data])
    y = np.array([row[m] for row in data])

    # Обучаем линейный SVM
    model = LinearSVC(C=1e5, max_iter=10000, tol=1e-5)
    model.fit(X, y)

    # Получаем разделяющий вектор
    weights = model.coef_[0]

    # Выводим результат
    print(" ".join(map(str, weights)))


if __name__ == "__main__":
    main()
