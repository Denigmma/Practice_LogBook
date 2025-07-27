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
from scipy.stats import t

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, '..', 'O. t-test', 'synthetic_gmv_data_1.2.csv')
df = pd.read_csv(data_path)

user_metrics = (
    df
    .groupby(['user_id', 'group_name'])
    .agg(
        total_gmv = ('gmv', 'sum'),
        orders    = ('gmv', 'count'),
        aov       = ('gmv', 'mean')
    )
    .reset_index()
)

ctl = user_metrics[user_metrics.group_name == 'control']
tst = user_metrics[user_metrics.group_name == 'test']

def ci_diff(x1, x2, alpha=0.05):
    """
    Двухвыборочный CI для разницы средних (x2.mean() - x1.mean())
    с df = n1+n2-2 и пуленой дисперсией.
    Возвращает (mean_diff, lower, upper, se_diff, df)
    """
    n1, n2 = len(x1), len(x2)
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.var(ddof=1), x2.var(ddof=1)
    df_tot = n1 + n2 - 2
    # pooled variance
    s_pooled = ((n1-1)*s1 + (n2-1)*s2) / df_tot
    se = np.sqrt(s_pooled * (1/n1 + 1/n2))
    t_crit = t.ppf(1 - alpha/2, df_tot)
    diff = m2 - m1
    lower = diff - t_crit * se
    upper = diff + t_crit * se
    return diff, lower, upper, se, df_tot

# CI для разницы GMV
diff_gmv, lo_gmv, hi_gmv, se_gmv, df_gmv = ci_diff(ctl.total_gmv, tst.total_gmv)


# CI для процентного изменения GMV относительно контроля
pct_diff_gmv = diff_gmv / ctl.total_gmv.mean() * 100
lo_pct_gmv  = lo_gmv   / ctl.total_gmv.mean() * 100
hi_pct_gmv  = hi_gmv   / ctl.total_gmv.mean() * 100


# CI для разницы AOV
diff_aov, lo_aov, hi_aov, se_aov, df_aov = ci_diff(ctl.aov, tst.aov)


# CI для процентного изменения AOV
pct_diff_aov = diff_aov / ctl.aov.mean() * 100
lo_pct_aov   = lo_aov   / ctl.aov.mean() * 100
hi_pct_aov   = hi_aov   / ctl.aov.mean() * 100


print(
    f"[{lo_gmv:.3f}, {hi_gmv:.3f}] "
    f"[{lo_pct_gmv:.3f}, {hi_pct_gmv:.3f}] "
    f"[{lo_aov:.3f}, {hi_aov:.3f}] "
    f"[{lo_pct_aov:.3f}, {hi_pct_aov:.3f}]"
)


# answer: [4.090, 42.777] [0.144, 1.502] [1.575, 7.610] [0.225, 1.087]