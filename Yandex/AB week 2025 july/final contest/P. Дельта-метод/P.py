'''
P. Дельта-метод
На тех же данных теперь проверим гипотезу равенства среднего чека.

Посчитайте T-статистику, найдите P-value.

Используйте данные из файла synthetic_gmv_data_1.2.csv (https://github.com/dakhakimova/YSDA_ABweek/blob/476cbc4a49e1f4dfcdb376d69239b6103fbad932/synthetic_gmv_data_1.2.csv)

Формат вывода
В ответе выведите два числа через пробел. Целую и дробную часть разделяйте точкой.

Пример ответа: -0.046 0.476

Примечания
Округлите до 3-го знака после точки.

Используте распределение Стьюдента (вместо нормального), для степеней свободы используйте упрощенную формулу:
n+m−2 (количество уников теста + количество уников контроля - 2). Оценивайте дисперсию, используя Дельта-метод.
'''


import pandas as pd
import os
from scipy.stats import t

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'O. t-test', 'synthetic_gmv_data_1.2.csv')

df = pd.read_csv(data_path)
df_user = df.groupby(['user_id', 'group_name'], as_index=False)['gmv'].sum()

control = df_user.loc[df_user['group_name'] == 'control', 'gmv']
test = df_user.loc[df_user['group_name'] == 'test',    'gmv']

n_c = control.size
n_t = test.size

mean_c = control.mean()
mean_t = test.mean()

var_c = control.var(ddof=1)
var_t = test.var(ddof=1)

# оценка стандартной ошибки разницы средних по Δ‑методу
se_diff = (var_t / n_t + var_c / n_c) ** 0.5

t_stat = (mean_t - mean_c) / se_diff
dfree  = n_t + n_c - 2

# двусторонний p‑value по Стьюденту
p_value = 2 * t.sf(abs(t_stat), dfree)

print(f"{t_stat:.3f} {p_value:.3f}")


# answer: 2.360 0.018