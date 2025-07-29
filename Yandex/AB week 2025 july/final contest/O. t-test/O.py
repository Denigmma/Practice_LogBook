'''
O. t‑test

Вам даны данные следующего формата: для каждого события зафиксирован идентификатор пользователя (user_id), сумма транзакции (gmv), группа эксперимента (group_name).
Сагрегируйте данные до пользователя (тип агрегации: сумма). Мы проверяем гипотезу равенства средних поюзерных gmv.
Посчитайте T‑статистику. Найдите P‑value, используя T‑test.
Используйте данные из файла synthetic_gmv_data_1.2.csv. (https://github.com/dakhakimova/YSDA_ABweek/blob/476cbc4a49e1f4dfcdb376d69239b6103fbad932/synthetic_gmv_data_1.2.csv)

Формат вывода
В ответе выведите два числа через пробел. Целую и дробную часть разделяйте точкой.
Пример ответа: 1.990 2.014

Примечания
Округлите до 3‑го знака после точки.
'''


import pandas as pd
import os
from scipy.stats import ttest_ind

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'synthetic_gmv_data_1.2.csv')

df = pd.read_csv(data_path)
df_user = df.groupby(['user_id', 'group_name'], as_index=False)['gmv'].sum()

control = df_user.loc[df_user['group_name'] == 'control', 'gmv']
test = df_user.loc[df_user['group_name'] == 'test', 'gmv']

t_stat, p_value = ttest_ind(test, control, equal_var=False)
print(f"{t_stat:.3f} {p_value:.3f}")


# answer: 2.360 0.018