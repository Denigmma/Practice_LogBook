'''
N. MDE

Используйте оценку матожидания и оценку дисперсии из предыдущей задачи для дизайна эксперимента.
Предположим, мы хотим "увидеть" эффект в 1% с вероятностью ошибки первого рода 5% и вероятностью ошибки второго рода 20%.
Разбиение на тестовую и контрольную группу у нас 1к3 (контроль в 3 раза больше теста). Предположим, что дисперсии в тестовой и контрольной группах одинаковые.
Сколько всего пользователей должно быть в эксперименте?

Формат вывода
Выведите натуральное число.

Примечания
Округлите до целого наверх, используя math.ceil.
'''


import pandas as pd
import math
from scipy.stats import norm
import os

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'M. Оценки', 'synthetic_gmv_data_1.1.csv')

df = pd.read_csv(data_path)
df_user = df.groupby('user_id', as_index=False)['gmv'].sum()

mu = df_user['gmv'].mean()
sigma = df_user['gmv'].std(ddof=1)

alpha = 0.05
beta = 0.20
power = 1 - beta
effect_rel = 0.01
delta = mu * effect_rel
ratio = 3

z_alpha = norm.ppf(1 - alpha/2)
z_beta = norm.ppf(power)

n_test = ((z_alpha + z_beta)**2 * sigma**2 * (1 + 1/ratio)) / delta**2
n_control = n_test * ratio
n_total = math.ceil(n_test + n_control)

print(n_total)


# answer: 201706