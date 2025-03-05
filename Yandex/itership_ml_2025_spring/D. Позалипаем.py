from itertools import product
import numpy as np

# Вероятности активности в зависимости от настроения
prob_G = {'L': 0.6, 'M': 0.3, 'H': 0.1}
prob_B = {'L': 0.2, 'M': 0.3, 'H': 0.5}

# Вероятности перехода между состояниями
p_GG = p_BB = 0.7  # Вероятность остаться в том же состоянии
p_GB = p_BG = 0.3  # Вероятность смены состояния

# Последовательность наблюдений активности
observed = "LMMHMLMMLLHMMHL"
n = len(observed)

# Матрицы вероятностей
dp = np.zeros((2, n))  # dp[0][i] - вероятность G, dp[1][i] - вероятность B
backtrack = np.zeros((2, n), dtype=int)  # Для восстановления пути

# Инициализация (предполагаем равные вероятности первого дня)
dp[0][0] = 0.5 * prob_G[observed[0]]
dp[1][0] = 0.5 * prob_B[observed[0]]

# Динамическое программирование (алгоритм Витерби)
for i in range(1, n):
    for cur_state, prob_cur, prev_G, prev_B in [(0, prob_G, p_GG, p_BG), (1, prob_B, p_BB, p_GB)]:
        prob_from_G = dp[0][i - 1] * prev_G * prob_cur[observed[i]]
        prob_from_B = dp[1][i - 1] * prev_B * prob_cur[observed[i]]

        if prob_from_G > prob_from_B:
            dp[cur_state][i] = prob_from_G
            backtrack[cur_state][i] = 0
        else:
            dp[cur_state][i] = prob_from_B
            backtrack[cur_state][i] = 1

# Восстановление последовательности состояний
result = []
state = 0 if dp[0][-1] > dp[1][-1] else 1  # Начинаем с наиболее вероятного состояния последнего дня
for i in range(n - 1, -1, -1):
    result.append('G' if state == 0 else 'B')
    state = backtrack[state][i]

print("".join(result[::-1]))
