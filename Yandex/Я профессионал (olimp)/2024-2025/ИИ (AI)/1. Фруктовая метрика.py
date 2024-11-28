### answer = 0.71
import numpy as np

probabilities = [0.85, 0.55, 0.65, 0.40, 0.95, 0.75, 0.50, 0.60, 0.30, 0.80]
true_labels = ["Спелый", "Спелый", "Неспелый", "Спелый", "Спелый", "Неспелый", "Спелый", "Спелый", "Неспелый", "Спелый"]

true_labels_numeric = [1 if label == "Спелый" else 0 for label in true_labels]

thresholds = np.arange(0, 1.1, 0.1)

# вычисление Precision при каждом пороге
def calculate_precision(threshold, probs, true_labels):
    """
    - threshold: порог вероятности для классификации.
    - probs: список предсказанных вероятностей.
    - true_labels: список истинных меток (1 для Спелый, 0 для Неспелый).
    """
    #классификация: предсказание положительного класса (1) при вероятности >= threshold
    predicted_positive = []
    for p in probs:
        if p >= threshold:
            predicted_positive.append(1)
        else:
            predicted_positive.append(0)

    #подсчёт истинно положительных (True Positives)
    tp = 0
    for pred, true in zip(predicted_positive, true_labels):
        if pred == 1 and true == 1:
            tp += 1

    #подсчёт ложноположительных (False Positives)
    fp = 0
    for pred, true in zip(predicted_positive, true_labels):
        if pred == 1 and true == 0:
            fp += 1

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    return precision


precisions = [calculate_precision(t, probabilities, true_labels_numeric) for t in thresholds]

average_precision = sum(precisions) / len(precisions)

print(f"{average_precision:.2f}")