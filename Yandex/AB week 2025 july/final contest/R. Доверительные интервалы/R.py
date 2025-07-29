'''
R. Доверительные интервалы

На тех же данных постройте доверительные интервалы для:
Δ gmv - разницы средних gmv
Δ gmv, % - процентного изменения средних gmv
Δ aov - разницы средних чеков
Δ aov, % - процентного изменения средних чеков
Используйте данные из файла synthetic_gmv_data_1.2.csv

Формат вывода
В ответе выведите 4 замкнутых интервала через пробел. Целую и дробную часть чисел разделяйте точкой.

Пример ответа:
[0.239, 0.179] [1.332, 2.007] [2.019, 2.025] [0.808, 2.004]

Примечания
Округление до 3-го знака после точки.

Везде используте распределение Стьюдента (вместо нормального), для степеней свободы используйте упрощенную формулу:
n+m−2 (количество уников теста + количество уников контроля - 2).
'''


import os
import pandas as pd
import numpy as np
from scipy import stats

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'O. t-test', 'synthetic_gmv_data_1.2.csv')
df = pd.read_csv(data_path)

user_metrics = (
    df
    .groupby(['user_id', 'group_name'])
    .agg(
        total_gmv=('gmv', 'sum'),
        orders=('gmv', 'count'),
        aov=('gmv', 'mean')
    )
    .reset_index()
)

ctl = user_metrics[user_metrics.group_name == 'control']
tst = user_metrics[user_metrics.group_name == 'test']
a=0.05

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


def delta_gmv(x_num, y_num):
    n = len(x_num)
    m = len(y_num)

    mu_x = np.mean(x_num)
    mu_y = np.mean(y_num)
    delta = mu_x - mu_y

    sigma = (x_num.var(ddof=1) / n + y_num.var(ddof=1) / m) ** 0.5

    quantile = stats.t.ppf(1 - a / 2, n + m - 2)

    return (f"[{round(delta - quantile * sigma, 3)}, {round(delta + quantile * sigma, 3)}]")


def delta_gmv_percent(x_num, y_num):
    n = len(x_num)
    m = len(y_num)

    mu_x = np.mean(x_num)
    mu_y = np.mean(y_num)

    delta = 100 * (mu_x - mu_y) / mu_y

    first = x_num.var(ddof=1) / n
    second = (mu_x ** 2 * y_num.var(ddof=1)) / (mu_y ** 2 * m)
    sigma = ((first + second) / mu_y ** 2) ** 0.5

    quantile = stats.norm.ppf(1 - a / 2, loc=0, scale=1)

    return (f"[{round(delta - 100 * quantile * sigma, 3)}, {round(delta + 100 * quantile * sigma, 3)}]")


def delta_aov(x_num, x_denom, y_num, y_denom):
    delta = safe_divide(x_num.mean(), x_denom.mean()) - safe_divide(y_num.mean(), y_denom.mean())

    sigma = (delta_var(x_num, x_denom) + delta_var(y_num, y_denom)) ** 0.5

    quantile = stats.norm.ppf(1 - a / 2, loc=0, scale=1)

    return (f"[{round(delta - quantile * sigma, 3)}, {round(delta + quantile * sigma, 3)}]")


def delta_aov_percent(x_num, x_denom, y_num, y_denom):
    mu_x_num = x_num.mean()
    mu_x_denom = x_denom.mean()

    mu_y_num = y_num.mean()
    mu_y_denom = y_denom.mean()

    Rt = safe_divide(mu_x_num, mu_x_denom)
    Rc = safe_divide(mu_y_num, mu_y_denom)

    Rt_var = delta_var(x_num, x_denom)
    Rc_var = delta_var(y_num, y_denom)

    delta = 100 * (Rt - Rc) / Rc

    sigma = ((Rt_var / Rc ** 2) + (Rc_var * Rt ** 2) / Rc ** 4) ** 0.5

    quantile = stats.norm.ppf(1 - a / 2, loc=0, scale=1)
    return (f"[{round(delta - 100 * quantile * sigma, 3)}, {round(delta + 100 * quantile * sigma, 3)}]")


gmv_interval = delta_gmv(tst['total_gmv'], ctl['total_gmv'])
gmv_percent_interval = delta_gmv_percent(tst['total_gmv'], ctl['total_gmv'])
aov_interval = delta_aov(tst['total_gmv'], tst['orders'], ctl['total_gmv'], ctl['orders'])
aov_percent_interval = delta_aov_percent(tst['total_gmv'], tst['orders'], ctl['total_gmv'], ctl['orders'])

print(gmv_interval,gmv_percent_interval, aov_interval, aov_percent_interval)


# answer: [3.975, 42.891] [0.138, 1.508] [0.652, 7.313] [0.092, 1.045]