import math

def combinations(n, k):
    if k > n or k < 0:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def calculate_probabilities():
    p_professional = 0.95  # вероятность попадания профессионала
    p_amateur = 0.15  # вероятность попадания любителя

    for n in range(1, 100):  # от 1 до 10
        for k in range(0, n + 1):  # от 0 до n
            # Вероятность того, что профессионал не выиграет
            p_professional_not_win = sum(
                combinations(n, j) * (p_professional ** j) * ((1 - p_professional) ** (n - j))
                for j in range(k)
            )
            # Вероятность того, что любитель выиграет
            p_amateur_win = sum(
                combinations(n, j) * (p_amateur ** j) * ((1 - p_amateur) ** (n - j))
                for j in range(k, n + 1)
            )

            # Проверка условий
            if p_professional_not_win < 0.01 and p_amateur_win <= 0.1:
                print(f"Подходящие значения: n = {n}, k = {k}")
                print(f"Вероятность того, что профессионал не выиграет: {p_professional_not_win:.6f}")
                print(f"Вероятность того, что любитель выиграет: {p_amateur_win:.6f}\n")


# Запуск функции
calculate_probabilities()
