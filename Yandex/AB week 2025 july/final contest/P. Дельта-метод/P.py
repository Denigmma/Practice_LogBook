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

import numpy as np
import pandas as pd
import os
from scipy import stats

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'O. t-test', 'synthetic_gmv_data_1.2.csv')

df = pd.read_csv(data_path)
df_user = df.groupby(['user_id', 'group_name'], as_index=False)['gmv'].sum()


def safe_divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return np.nan


def delta_var(numerator, denominator):
    """
    Функция для расчета дисперсии дельта-методом, numerator - вектор числитель, denominator - вектор знаменатель
    """
    x = numerator
    y = denominator
    n = len(x)
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0][1]
    delta_var = safe_divide(
        safe_divide(var_x, mu_y ** 2) - 2 * cov_xy * safe_divide(mu_x, mu_y ** 3) + var_y * safe_divide(mu_x ** 2,
                                                                                                        mu_y ** 4), n)
    return delta_var


def esttimate_tt_and_p(x_num, x_denom, y_num, y_denom):
    n = len(x_num)
    m = len(y_num)
    test_var = delta_var(x_num, x_denom)
    control_var = delta_var(y_num, y_denom)
    sigma = np.sqrt(test_var + control_var)
    delta_estimator = safe_divide(np.mean(x_num), np.mean(x_denom)) - safe_divide(np.mean(y_num), np.mean(y_denom))
    tt = safe_divide(delta_estimator, sigma)
    pvalue = 2 * stats.t.sf(np.abs(tt), n + m - 2)

    return tt, pvalue


df_control = df[df['group_name'] == 'control'].drop(columns='group_name')
df_test = df[df['group_name'] == 'test'].drop(columns='group_name')

df_control = df_control.groupby('user_id').agg(
    total_gmv=('gmv', 'sum'),
    count=('gmv', 'size')
).reset_index()

df_test = df_test.groupby('user_id').agg(
    total_gmv=('gmv', 'sum'),
    count=('gmv', 'size')
).reset_index()

x_num = df_control['total_gmv']
x_denom = df_control['count']
y_num = df_test['total_gmv']
y_denom = df_test['count']

tt, pvalue = esttimate_tt_and_p(x_num, x_denom, y_num, y_denom)
tt.item(), pvalue.item()

print(f"{tt:.3f} {pvalue:.3f}")

# answer -2.344 0.019